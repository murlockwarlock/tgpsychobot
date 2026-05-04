from __future__ import annotations

from uuid import uuid4

from sqlalchemy import delete, func, select

from ..api import MaxApiClient
from ..keyboards import admin_button_editor_keyboard, admin_buttons_keyboard
from ..legacy import Content, ContentMedia, async_session_maker
from ..storage import MaxContentMedia, StateStore


EXCLUDED_KEYS = {"test_intro", "secret_test_outro", "disclaimer", "test_results", "test_button"}


async def _load_buttons(session) -> list[Content]:
    return (
        await session.execute(
            select(Content)
            .where(Content.button_title != None, Content.key.not_in(EXCLUDED_KEYS))
            .order_by(Content.sort_order.asc(), Content.key.asc())
        )
    ).scalars().all()


async def _normalize_sort_order(session) -> None:
    buttons = await _load_buttons(session)
    for index, button in enumerate(buttons, start=1):
        button.sort_order = index


async def show_buttons(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        buttons = await _load_buttons(session)
    text = (
        "🎛️ <b>Кнопки главного меню</b>\n\n"
        "Здесь можно менять видимость, названия и порядок пользовательских кнопок."
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_buttons_keyboard(buttons))


async def show_button_editor(client: MaxApiClient, chat_id: int, button_key: str) -> None:
    async with async_session_maker() as session:
        button = await session.get(Content, button_key)
    if not button:
        await client.send_message(chat_id=chat_id, text="Кнопка не найдена.")
        return
    text = (
        f"<b>{button.button_title or button.key}</b>\n\n"
        f"<b>ID:</b> <code>{button.key}</code>\n"
        f"<b>Видимость:</b> {'да' if button.is_visible else 'нет'}\n"
        f"<b>Порядок:</b> {button.sort_order or 0}\n\n"
        f"<b>Текст контента:</b>\n<pre><code>{(button.text_content or 'Не задан').replace('<', '&lt;').replace('>', '&gt;')[:2500]}</code></pre>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_button_editor_keyboard(button.key, bool(button.is_visible)))


async def start_edit_title(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, button_key: str) -> None:
    await states.set(user_id, chat_id, "admin_button_edit_title", {"button_key": button_key})
    await client.send_message(chat_id=chat_id, text="Введите новое название кнопки.")


async def save_title(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    title = text.strip()
    if not title:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    snapshot = await states.get(user_id)
    button_key = snapshot.data.get("button_key") if snapshot else None
    if not button_key:
        await client.send_message(chat_id=chat_id, text="Состояние кнопки потеряно.")
        return
    async with async_session_maker() as session:
        button = await session.get(Content, button_key)
        if not button:
            await client.send_message(chat_id=chat_id, text="Кнопка не найдена.")
            return
        button.button_title = title
        await session.commit()
    await states.clear(user_id)
    await show_button_editor(client, chat_id, button_key)


async def start_create_button(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_button_create_title", {})
    await client.send_message(chat_id=chat_id, text="Введите название новой кнопки.")


async def create_button(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    title = text.strip()
    if not title:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    async with async_session_maker() as session:
        max_order = await session.scalar(select(func.max(Content.sort_order))) or 0
        key = f"btn_{uuid4().hex[:10]}"
        session.add(
            Content(
                key=key,
                button_title=title,
                is_visible=True,
                text_content=f"Это текст для новой кнопки «{title}». Отредактируйте его в разделе контента.",
                sort_order=max_order + 1,
            )
        )
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Кнопка «{title}» создана.")
    await show_button_editor(client, chat_id, key)


async def toggle_visibility(client: MaxApiClient, chat_id: int, button_key: str) -> None:
    async with async_session_maker() as session:
        button = await session.get(Content, button_key)
        if button:
            button.is_visible = not bool(button.is_visible)
            await session.commit()
    await show_button_editor(client, chat_id, button_key)


async def move_button(client: MaxApiClient, chat_id: int, button_key: str, direction: str) -> None:
    async with async_session_maker() as session:
        buttons = await _load_buttons(session)
        current_idx = next((idx for idx, item in enumerate(buttons) if item.key == button_key), None)
        if current_idx is None:
            await client.send_message(chat_id=chat_id, text="Кнопка не найдена.")
            return
        target_idx = current_idx - 1 if direction == "up" else current_idx + 1
        if target_idx < 0 or target_idx >= len(buttons):
            await client.send_message(chat_id=chat_id, text="Перемещение недоступно.")
            return
        buttons[current_idx], buttons[target_idx] = buttons[target_idx], buttons[current_idx]
        for index, button in enumerate(buttons, start=1):
            button.sort_order = index
        await session.commit()
    await show_button_editor(client, chat_id, button_key)


async def delete_button(client: MaxApiClient, chat_id: int, button_key: str) -> None:
    async with async_session_maker() as session:
        button = await session.get(Content, button_key)
        if not button:
            await client.send_message(chat_id=chat_id, text="Кнопка не найдена.")
            return
        await session.execute(delete(ContentMedia).where(ContentMedia.content_key == button_key))
        await session.execute(delete(MaxContentMedia).where(MaxContentMedia.content_key == button_key))
        await session.execute(delete(Content).where(Content.key == button_key))
        await session.flush()
        await _normalize_sort_order(session)
        await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Кнопка удалена.")
    await show_buttons(client, chat_id)
