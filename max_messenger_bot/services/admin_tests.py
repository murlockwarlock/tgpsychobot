from __future__ import annotations

import html
import io
from types import SimpleNamespace

from sqlalchemy import delete, func, select

from ..api import MaxApiClient
from ..keyboards import admin_secret_questions_keyboard, admin_test_links_keyboard, admin_test_menu_keyboard
from ..legacy import Content, SecretTestQuestion, TestConfig, TestQuestion, async_session_maker
from ..models import IncomingMessage
from ..storage import StateStore
from file_parser import parse_formulas_file, parse_questions_file
from universal_tests import json_dumps, json_loads, validate_test_definition


async def show_menu(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        if not config:
            config = TestConfig(id=1)
            session.add(config)
            await session.commit()
        question_count = await session.scalar(select(func.count(TestQuestion.id))) or 0
        formula_count = len(json_loads(getattr(config, "formulas_json", None), []))
    await client.send_message(
        chat_id=chat_id,
        text=(
            "🧩 <b>Управление разделом теста</b>\n\n"
            f"Вопросов: <b>{question_count}</b>\n"
            f"Формул: <b>{formula_count}</b>\n\n"
            "Настройки общие для Telegram и MAX в этой базе."
        ),
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


async def toggle_progress(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.show_progress = not bool(getattr(config, "show_progress", True))
        await session.commit()
    await show_menu(client, chat_id)


async def toggle_formulas(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        if not getattr(config, "formulas_json", None):
            await client.send_message(chat_id=chat_id, text="Сначала загрузите формулы.")
            return
        config.formulas_enabled = not bool(getattr(config, "formulas_enabled", False))
        if not config.formulas_enabled and config.interpretation_input_mode == "formulas":
            config.interpretation_input_mode = "all"
        await session.commit()
    await show_menu(client, chat_id)


async def toggle_input_mode(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        current = getattr(config, "interpretation_input_mode", "all") or "all"
        next_mode = {"all": "selected", "selected": "formulas", "formulas": "all"}.get(current, "all")
        if next_mode == "formulas" and not getattr(config, "formulas_json", None):
            next_mode = "all"
            await client.send_message(chat_id=chat_id, text="Режим «только формулы» пропущен: формулы не загружены.")
        config.interpretation_input_mode = next_mode
        await session.commit()
    await show_menu(client, chat_id)


async def toggle_separate_prompt(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        if not config.separate_result_prompt_enabled and not (getattr(config, "result_system_prompt", None) or "").strip():
            await client.send_message(chat_id=chat_id, text="Сначала задайте промпт результата.")
            return
        config.separate_result_prompt_enabled = not bool(config.separate_result_prompt_enabled)
        await session.commit()
    await show_menu(client, chat_id)


async def start_selected_variables(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
    available = [question.variable_name or f"answer_{index + 1:02d}" for index, question in enumerate(questions)]
    selected = json_loads(getattr(config, "interpretation_selected_variables", None), [])
    await states.set(user_id, chat_id, "admin_test_set_selected_vars", {"available": available})
    await client.send_message(
        chat_id=chat_id,
        text=(
            "<b>Выбранные переменные</b>\n\n"
            f"Сейчас: <code>{html.escape(', '.join(selected) or 'не заданы')}</code>\n"
            f"Доступны: <code>{html.escape(', '.join(available) or 'нет')}</code>\n\n"
            "Отправьте переменные через запятую или <code>-</code>, чтобы очистить."
        ),
    )


async def save_selected_variables(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    available = set(snapshot.data.get("available", [])) if snapshot else set()
    selected = [] if text.strip() == "-" else [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    unknown = sorted(set(selected) - available)
    if unknown:
        await client.send_message(chat_id=chat_id, text="Неизвестные переменные: " + ", ".join(unknown))
        return
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.interpretation_selected_variables = json_dumps(selected)
        await session.commit()
    await states.clear(user_id)
    await show_menu(client, chat_id)


async def start_upload(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, kind: str) -> None:
    state_name = "admin_test_upload_questions" if kind == "questions" else "admin_test_upload_formulas"
    await states.set(user_id, chat_id, state_name, {})
    target = "вопросами" if kind == "questions" else "формулами"
    await client.send_message(
        chat_id=chat_id,
        text=f"Отправьте файл <b>.xlsx</b>, <b>.csv</b> или <b>.txt</b> с {target}. Текущие данные изменятся только после успешной проверки.",
    )


def _attachment_filename(message: IncomingMessage) -> str:
    for attachment in message.attachments or []:
        payload = attachment.get("payload") or {}
        for source in (payload, attachment):
            for key in ("name", "filename", "file_name", "title"):
                if source.get(key):
                    return str(source[key])
    return "uploaded"


def _question_preview(data: dict) -> SimpleNamespace:
    return SimpleNamespace(
        text=data.get("text", ""),
        category=data.get("category", "general"),
        is_reverse=data.get("is_reverse", False),
        comment=data.get("comment"),
        variable_name=data.get("variable_name"),
        allow_custom_answer=data.get("allow_custom_answer", False),
        buttons_layout=data.get("buttons_layout", "vertical"),
        answer_options_json=data.get("answer_options_json"),
    )


async def receive_test_file(client: MaxApiClient, states: StateStore, message: IncomingMessage, kind: str) -> None:
    filename = _attachment_filename(message)
    extension = filename.rsplit('.', 1)[-1].lower()
    try:
        raw = await client.download_attachment(message.media_token, message.media_url)
        if extension not in {"xlsx", "csv", "txt"} and raw.startswith(b"PK"):
            filename = f"{filename}.xlsx"
            extension = "xlsx"
        if extension not in {"xlsx", "csv", "txt"}:
            await client.send_message(chat_id=message.chat_id, text="Не удалось определить формат. У файла должно быть расширение .xlsx, .csv или .txt.")
            return
        if kind == "formulas":
            formulas = await parse_formulas_file(io.BytesIO(raw), filename)
            async with async_session_maker() as session:
                questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
                errors = validate_test_definition(questions, formulas)
                if not formulas:
                    errors.insert(0, "В файле не найдено формул.")
                if errors:
                    await client.send_message(chat_id=message.chat_id, text="Формулы не сохранены:\n- " + "\n- ".join(errors[:15]))
                    return
                config = await session.get(TestConfig, 1)
                config.formulas_json = json_dumps(formulas)
                config.formulas_enabled = True
                await session.commit()
            result_text = f"✅ Загружено и включено формул: {len(formulas)}."
        else:
            parsed = await parse_questions_file(io.BytesIO(raw), filename)
            questions_data = parsed.get("questions", [])
            formulas = parsed.get("formulas", [])
            previews = [_question_preview(item) for item in questions_data]
            errors = validate_test_definition(previews, formulas)
            if not questions_data:
                errors.insert(0, "В файле не найдено вопросов.")
            if errors:
                await client.send_message(chat_id=message.chat_id, text="Тест не изменён:\n- " + "\n- ".join(errors[:15]))
                return
            async with async_session_maker() as session:
                await session.execute(delete(TestQuestion))
                for index, item in enumerate(questions_data):
                    session.add(TestQuestion(sort_order=index, **item))
                config = await session.get(TestConfig, 1)
                config.formulas_json = json_dumps(formulas) if formulas else None
                config.formulas_enabled = bool(formulas)
                await session.commit()
            result_text = f"✅ Загружено вопросов: {len(questions_data)}. Формул: {len(formulas)}."
        await states.clear(message.sender.user_id)
        await client.send_message(chat_id=message.chat_id, text=result_text)
        await show_menu(client, message.chat_id)
    except Exception as exc:
        await client.send_message(chat_id=message.chat_id, text=f"Файл не применён: {html.escape(str(exc))}")


async def show_links(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
    text = (
        "🔗 <b>Настройка ссылок</b>\n\n"
        f"👤 <b>Admin Username:</b> @{html.escape(config.admin_username or '')}\n"
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
        text="🔐 <b>Секретные вопросы</b>\n\nДобавляйте вопросы для второго блока теста.",
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
        text=f"<b>Текущий промпт теста:</b>\n<pre><code>{html.escape(preview)}</code></pre>\nОтправьте новый текст промпта сообщением или загрузите <b>.txt/.md</b> файл.",
    )


async def start_edit_result_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_set_result_prompt", {})
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        current = config.result_system_prompt or "Не задан."
    preview = current[:3000] + ("\n\n[...]" if len(current) > 3000 else "")
    await client.send_message(
        chat_id=chat_id,
        text=(
            "<b>Отдельный промпт интерпретации</b>\n\n"
            f"<pre><code>{html.escape(preview)}</code></pre>\n"
            "Отправьте новый текст или .txt/.md файл. После интерпретации результат всё равно будет передан в вызвавший промпт."
        ),
    )


async def save_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.test_system_prompt = text
        await session.commit()
    await states.clear(user_id)
    await show_menu(client, chat_id)


async def save_result_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    if not text.strip():
        await client.send_message(chat_id=chat_id, text="Промпт не может быть пустым.")
        return
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.result_system_prompt = text.strip()
        await session.commit()
    await states.clear(user_id)
    await show_menu(client, chat_id)
