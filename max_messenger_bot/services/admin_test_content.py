from __future__ import annotations

import html
import math

from sqlalchemy import delete, func, select

from ..api import MaxApiClient
from ..keyboards import (
    admin_case_studies_keyboard,
    admin_case_study_editor_keyboard,
    admin_test_question_categories_keyboard,
    admin_test_question_editor_keyboard,
    admin_test_questions_keyboard,
    callback_button,
    inline_keyboard,
)
from ..legacy import CaseStudy, TestQuestion, async_session_maker
from ..storage import StateStore


PAGE_SIZE = 10

CATEGORY_NAMES = {
    "body": "Отношение к телу",
    "face": "Отношение к лицу",
    "age": "Отношение к возрасту",
    "health": "Отношение к здоровью",
    "abilities": "Отношение к способностям",
    "relations": "Отношения с окружающими",
    "success": "Успешность и реализация",
}


def _reverse_keyboard(prefix: str) -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("Прямой вопрос", f"{prefix}_direct")],
            [callback_button("Обратный вопрос", f"{prefix}_reverse")],
        ]
    )


async def _normalize_question_sort_order(session) -> None:
    questions = (
        await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc(), TestQuestion.id.asc()))
    ).scalars().all()
    for index, question in enumerate(questions, start=1):
        question.sort_order = index


async def list_questions(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        questions = (
            await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc(), TestQuestion.id.asc()))
        ).scalars().all()
    text = "❓ <b>Вопросы теста</b><br/><br/>Выберите вопрос для редактирования." if questions else "❓ <b>Вопросы теста</b><br/><br/>Вопросов пока нет."
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_test_questions_keyboard(questions))


async def show_question_editor(client: MaxApiClient, chat_id: int, question_id: int) -> None:
    async with async_session_maker() as session:
        question = await session.get(TestQuestion, question_id)
    if not question:
        await client.send_message(chat_id=chat_id, text="Вопрос не найден.")
        return
    text = (
        f"<b>Вопрос #{question.id}</b><br/><br/>"
        f"<b>Порядок:</b> {question.sort_order}<br/>"
        f"<b>Категория:</b> {html.escape(CATEGORY_NAMES.get(question.category, question.category))}<br/>"
        f"<b>Тип:</b> {'обратный' if question.is_reverse else 'прямой'}<br/><br/>"
        f"<pre><code>{html.escape(question.text)}</code></pre>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_test_question_editor_keyboard(question.id, bool(question.is_reverse)))


async def start_create_question(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_question_create_text", {})
    await client.send_message(chat_id=chat_id, text="Введите текст нового вопроса теста.")


async def save_new_question_text(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    question_text = text.strip()
    if not question_text:
        await client.send_message(chat_id=chat_id, text="Текст вопроса не может быть пустым.")
        return
    await states.set(user_id, chat_id, "admin_test_question_create_category", {"text": question_text})
    await client.send_message(
        chat_id=chat_id,
        text="Выберите категорию для нового вопроса.",
        attachments=admin_test_question_categories_keyboard("admin_test_question_create_category"),
    )


async def choose_new_question_category(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, category: str) -> None:
    if category not in CATEGORY_NAMES:
        await client.send_message(chat_id=chat_id, text="Неизвестная категория.")
        return
    snapshot = await states.get(user_id)
    if not snapshot or "text" not in snapshot.data:
        await client.send_message(chat_id=chat_id, text="Состояние создания вопроса потеряно.")
        return
    data = dict(snapshot.data)
    data["category"] = category
    await states.set(user_id, chat_id, "admin_test_question_create_reverse", data)
    await client.send_message(
        chat_id=chat_id,
        text="Выберите тип вопроса.",
        attachments=_reverse_keyboard("admin_test_question_create_reverse"),
    )


async def create_question(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, reverse_mode: str) -> None:
    is_reverse = reverse_mode == "reverse"
    snapshot = await states.get(user_id)
    if not snapshot:
        await client.send_message(chat_id=chat_id, text="Состояние создания вопроса потеряно.")
        return
    data = dict(snapshot.data)
    async with async_session_maker() as session:
        total = await session.scalar(select(func.count(TestQuestion.id))) or 0
        question = TestQuestion(
            text=data["text"],
            category=data["category"],
            is_reverse=is_reverse,
            sort_order=total + 1,
        )
        session.add(question)
        await session.commit()
        question_id = question.id
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Вопрос создан.")
    await show_question_editor(client, chat_id, question_id)


async def start_edit_question_text(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, question_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_question_edit_text", {"question_id": question_id})
    await client.send_message(chat_id=chat_id, text="Введите новый текст вопроса.")


async def save_question_text(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    question_text = text.strip()
    if not question_text:
        await client.send_message(chat_id=chat_id, text="Текст вопроса не может быть пустым.")
        return
    snapshot = await states.get(user_id)
    question_id = snapshot.data.get("question_id") if snapshot else None
    if not question_id:
        await client.send_message(chat_id=chat_id, text="Состояние вопроса потеряно.")
        return
    async with async_session_maker() as session:
        question = await session.get(TestQuestion, question_id)
        if not question:
            await client.send_message(chat_id=chat_id, text="Вопрос не найден.")
            return
        question.text = question_text
        await session.commit()
    await states.clear(user_id)
    await show_question_editor(client, chat_id, question_id)


async def show_question_category_menu(client: MaxApiClient, chat_id: int, question_id: int) -> None:
    async with async_session_maker() as session:
        question = await session.get(TestQuestion, question_id)
    if not question:
        await client.send_message(chat_id=chat_id, text="Вопрос не найден.")
        return
    await client.send_message(
        chat_id=chat_id,
        text="Выберите новую категорию вопроса.",
        attachments=admin_test_question_categories_keyboard(f"admin_test_question_set_category_{question_id}", question.category),
    )


async def set_question_category(client: MaxApiClient, chat_id: int, question_id: int, category: str) -> None:
    if category not in CATEGORY_NAMES:
        await client.send_message(chat_id=chat_id, text="Неизвестная категория.")
        return
    async with async_session_maker() as session:
        question = await session.get(TestQuestion, question_id)
        if not question:
            await client.send_message(chat_id=chat_id, text="Вопрос не найден.")
            return
        question.category = category
        await session.commit()
    await show_question_editor(client, chat_id, question_id)


async def toggle_question_reverse(client: MaxApiClient, chat_id: int, question_id: int) -> None:
    async with async_session_maker() as session:
        question = await session.get(TestQuestion, question_id)
        if question:
            question.is_reverse = not bool(question.is_reverse)
            await session.commit()
    await show_question_editor(client, chat_id, question_id)


async def start_edit_question_sort(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, question_id: int) -> None:
    await states.set(user_id, chat_id, "admin_test_question_edit_sort", {"question_id": question_id})
    await client.send_message(chat_id=chat_id, text="Введите новый номер позиции вопроса.")


async def save_question_sort(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        target_position = int(text.strip())
        if target_position <= 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое положительное число.")
        return
    snapshot = await states.get(user_id)
    question_id = snapshot.data.get("question_id") if snapshot else None
    if not question_id:
        await client.send_message(chat_id=chat_id, text="Состояние вопроса потеряно.")
        return
    async with async_session_maker() as session:
        questions = (
            await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc(), TestQuestion.id.asc()))
        ).scalars().all()
        target = next((item for item in questions if item.id == question_id), None)
        if not target:
            await client.send_message(chat_id=chat_id, text="Вопрос не найден.")
            return
        questions = [item for item in questions if item.id != question_id]
        insert_index = min(max(target_position - 1, 0), len(questions))
        questions.insert(insert_index, target)
        for index, item in enumerate(questions, start=1):
            item.sort_order = index
        await session.commit()
    await states.clear(user_id)
    await show_question_editor(client, chat_id, question_id)


async def delete_question(client: MaxApiClient, chat_id: int, question_id: int) -> None:
    async with async_session_maker() as session:
        question = await session.get(TestQuestion, question_id)
        if not question:
            await client.send_message(chat_id=chat_id, text="Вопрос не найден.")
            return
        await session.execute(delete(TestQuestion).where(TestQuestion.id == question_id))
        await session.flush()
        await _normalize_question_sort_order(session)
        await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Вопрос удалён.")
    await list_questions(client, chat_id)


async def list_case_studies(client: MaxApiClient, chat_id: int, page: int) -> None:
    async with async_session_maker() as session:
        total = await session.scalar(select(func.count(CaseStudy.id))) or 0
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        page = max(0, min(page, total_pages - 1))
        cases = (
            await session.execute(
                select(CaseStudy).order_by(CaseStudy.id.desc()).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
            )
        ).scalars().all()
    text = (
        f"📖 <b>Кейсы и истории</b><br/><br/>Страница {page + 1}/{total_pages}.<br/>"
        "Эта база используется для подбора релевантной истории по результатам теста."
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_case_studies_keyboard(cases, page, total_pages))


async def show_case_study_editor(client: MaxApiClient, chat_id: int, case_id: int, page: int | None = None) -> None:
    async with async_session_maker() as session:
        case_study = await session.get(CaseStudy, case_id)
    if not case_study:
        await client.send_message(chat_id=chat_id, text="Кейс не найден.")
        return
    preview = case_study.text[:3000] + ("..." if len(case_study.text) > 3000 else "")
    text = f"<b>Кейс #{case_study.id}</b><br/><br/><pre><code>{html.escape(preview)}</code></pre>"
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_case_study_editor_keyboard(case_study.id, page))


async def start_create_case_study(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_case_study_create_text", {})
    await client.send_message(chat_id=chat_id, text="Отправьте текст нового кейса одним сообщением.")


async def save_new_case_study(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    case_text = text.strip()
    if not case_text:
        await client.send_message(chat_id=chat_id, text="Текст кейса не может быть пустым.")
        return
    async with async_session_maker() as session:
        item = CaseStudy(text=case_text)
        session.add(item)
        await session.commit()
        case_id = item.id
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Кейс создан.")
    await show_case_study_editor(client, chat_id, case_id)


async def start_edit_case_study(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, case_id: int, page: int | None) -> None:
    await states.set(user_id, chat_id, "admin_case_study_edit_text", {"case_id": case_id, "page": page})
    await client.send_message(chat_id=chat_id, text="Введите новый текст кейса.")


async def save_case_study_text(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    case_text = text.strip()
    if not case_text:
        await client.send_message(chat_id=chat_id, text="Текст кейса не может быть пустым.")
        return
    snapshot = await states.get(user_id)
    case_id = snapshot.data.get("case_id") if snapshot else None
    page = snapshot.data.get("page") if snapshot else None
    if not case_id:
        await client.send_message(chat_id=chat_id, text="Состояние кейса потеряно.")
        return
    async with async_session_maker() as session:
        case_study = await session.get(CaseStudy, case_id)
        if not case_study:
            await client.send_message(chat_id=chat_id, text="Кейс не найден.")
            return
        case_study.text = case_text
        await session.commit()
    await states.clear(user_id)
    await show_case_study_editor(client, chat_id, case_id, page)


async def delete_case_study(client: MaxApiClient, chat_id: int, case_id: int, page: int = 0) -> None:
    async with async_session_maker() as session:
        item = await session.get(CaseStudy, case_id)
        if not item:
            await client.send_message(chat_id=chat_id, text="Кейс не найден.")
            return
        await session.delete(item)
        await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Кейс удалён.")
    await list_case_studies(client, chat_id, page)
