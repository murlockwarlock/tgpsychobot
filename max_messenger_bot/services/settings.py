from __future__ import annotations

import html

from sqlalchemy import update

from ..api import MaxApiClient
from ..keyboards import gender_keyboard, settings_keyboard
from ..legacy import User, async_session_maker
from ..storage import StateStore


def _settings_text(user: User, prefix: str | None = None) -> str:
    header = f"{prefix}\n\n" if prefix else ""
    length_text = "📏 Короткий" if getattr(user, "response_length", "normal") == "short" else "📏 Обычный"
    return (
        f"{header}<b>⚙️ Настройки</b>\n\n"
        f"<b>Имя:</b> {html.escape(user.name or user.first_name or 'Не указано')}\n"
        f"<b>Пол:</b> {'👨 Мужской' if user.gender == 'male' else ('👩 Женский' if user.gender == 'female' else '❓ Не указан')}\n"
        f"<b>Возраст:</b> {user.age or 'Не указан'}\n"
        f"<b>Длина ответов:</b> {length_text}"
    )


async def show_settings(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
    if not user:
        return
    await client.send_message(chat_id=chat_id, text=_settings_text(user), attachments=settings_keyboard(user))


async def start_change_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "awaiting_new_name", {"is_settings": True})
    await client.send_message(chat_id=chat_id, text="Введите новое имя.")


async def save_name_only(states: StateStore, user_id: int, text: str) -> str | None:
    """Save the user's name during onboarding without showing a settings block. Returns name or None if invalid."""
    user_name = text.strip()
    if not user_name or len(user_name) > 50:
        return None
    async with async_session_maker() as session:
        await session.execute(update(User).where(User.id == user_id).values(name=user_name))
        await session.commit()
    await states.clear(user_id)
    return user_name


async def process_new_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    user_name = text.strip()
    if not user_name or len(user_name) > 50:
        await client.send_message(chat_id=chat_id, text="Введите корректное имя.")
        return
    async with async_session_maker() as session:
        await session.execute(update(User).where(User.id == user_id).values(name=user_name))
        await session.commit()
        user = await session.get(User, user_id)
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=_settings_text(user, f"✅ Имя изменено на <b>{html.escape(user_name)}</b>"), attachments=settings_keyboard(user))


async def start_change_gender(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, *, is_settings: bool = True, initial_prompt: str | None = None) -> None:
    data = {"is_settings": is_settings}
    if initial_prompt:
        data["initial_prompt"] = initial_prompt
        data["is_onboarding"] = True
    await states.set(user_id, chat_id, "awaiting_gender", data)
    await client.send_message(chat_id=chat_id, text="Выберите ваш пол:", attachments=gender_keyboard())


async def save_gender(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, gender: str) -> dict:
    async with async_session_maker() as session:
        await session.execute(update(User).where(User.id == user_id).values(gender=gender))
        await session.commit()
        user = await session.get(User, user_id)
    snapshot = await states.get(user_id)
    await states.clear(user_id)
    if snapshot and snapshot.data.get("is_settings"):
        await client.send_message(chat_id=chat_id, text=_settings_text(user, "✅ Пол обновлён"), attachments=settings_keyboard(user))
    return snapshot.data if snapshot else {}


async def start_change_age(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, *, is_settings: bool = True) -> None:
    await states.set(user_id, chat_id, "awaiting_age", {"is_settings": is_settings})
    await client.send_message(chat_id=chat_id, text="Введите возраст числом (например, 25).")


async def save_age(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> dict:
    if not text.strip().isdigit():
        await client.send_message(chat_id=chat_id, text="Возраст нужно ввести числом.")
        return {}
    age = int(text.strip())
    if age < 1 or age > 120:
        await client.send_message(chat_id=chat_id, text="Введите реальный возраст.")
        return {}
    async with async_session_maker() as session:
        await session.execute(update(User).where(User.id == user_id).values(age=str(age)))
        await session.commit()
        user = await session.get(User, user_id)
    snapshot = await states.get(user_id)
    await states.clear(user_id)
    if snapshot and snapshot.data.get("is_settings"):
        await client.send_message(chat_id=chat_id, text=_settings_text(user, f"✅ Возраст установлен: {age}"), attachments=settings_keyboard(user))
    return snapshot.data if snapshot else {}


async def toggle_response_length(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            return
        new_length = "short" if getattr(user, "response_length", "normal") != "short" else "normal"
        await session.execute(update(User).where(User.id == user_id).values(response_length=new_length))
        await session.commit()
        user = await session.get(User, user_id)
    await client.send_message(chat_id=chat_id, text=_settings_text(user, "✅ Длина ответов изменена"), attachments=settings_keyboard(user))

