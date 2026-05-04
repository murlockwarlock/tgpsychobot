from __future__ import annotations

import html

from sqlalchemy import delete, func, select

from ..api import MaxApiClient
from ..keyboards import admin_secret_questions_keyboard, admin_test_links_keyboard, admin_test_menu_keyboard
from ..legacy import Content, SecretTestQuestion, TestConfig, async_session_maker
from ..storage import StateStore


async def show_menu(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        if not config:
            config = TestConfig(id=1)
            session.add(config)
            await session.commit()
    await client.send_message(
        chat_id=chat_id,
        text="🧩 <b>Управление разделом теста</b><br><br>Здесь настраивается пользовательский сценарий теста.",
        attachments=admin_test_menu_keyboard(config),
    )


async def toggle_status(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.is_enabled = not config.is_enabled
        btn = await session.get(Content, "test_button")
        if btn:
            btn.is_visible = config.is_enabled
        await session.commit()
    await show_menu(client, chat_id)


async def show_links(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
    text = (
        "🔗 <b>Настройка ссылок</b><br><br>"
        f"👤 <b>Admin Username:</b> @{html.escape(config.admin_username or '')}<br>"
        f"🚀 <b>Марафон URL:</b> {html.escape(config.marathon_url or '')}"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_test_links_keyboard())


async def start_set_admin_username(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_set_admin_username", {})
    await client.send_message(chat_id=chat_id, text="Введите username администратора без @.")


async def save_admin_username(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.admin_username = text.strip().replace("@", "")
        await session.commit()
    await states.clear(user_id)
    await show_links(client, chat_id)


async def start_set_marathon_url(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_set_marathon_url", {})
    await client.send_message(chat_id=chat_id, text="Введите новую ссылку на марафон.")


async def save_marathon_url(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.marathon_url = text.strip()
        await session.commit()
    await states.clear(user_id)
    await show_links(client, chat_id)


async def show_secret_questions(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        items = (await session.execute(select(SecretTestQuestion).order_by(SecretTestQuestion.sort_order.asc()))).scalars().all()
    await client.send_message(
        chat_id=chat_id,
        text="🔐 <b>Секретные вопросы</b><br><br>Добавляйте вопросы для второго блока теста.",
        attachments=admin_secret_questions_keyboard(items),
    )


async def start_add_secret_question(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_add_secret_question", {})
    await client.send_message(chat_id=chat_id, text="Введите текст нового секретного вопроса.")


async def save_secret_question(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    question_text = text.strip()
    if not question_text:
        await client.send_message(chat_id=chat_id, text="Текст вопроса не может быть пустым.")
        return
    async with async_session_maker() as session:
        count = await session.scalar(select(func.count(SecretTestQuestion.id))) or 0
        session.add(SecretTestQuestion(text=question_text, sort_order=count + 1))
        await session.commit()
    await states.clear(user_id)
    await show_secret_questions(client, chat_id)


async def delete_secret_question(client: MaxApiClient, chat_id: int, question_id: int) -> None:
    async with async_session_maker() as session:
        await session.execute(delete(SecretTestQuestion).where(SecretTestQuestion.id == question_id))
        await session.commit()
    await show_secret_questions(client, chat_id)


async def start_edit_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_set_prompt", {})
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        current = config.test_system_prompt or "Не задан."
    preview = current[:3000] + ("\n\n[...]" if len(current) > 3000 else "")
    await client.send_message(
        chat_id=chat_id,
        text=f"<b>Текущий промпт теста:</b><br><pre><code>{html.escape(preview)}</code></pre><br>Отправьте новый текст промпта.",
    )


async def save_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.test_system_prompt = text
        await session.commit()
    await states.clear(user_id)
    await show_menu(client, chat_id)

