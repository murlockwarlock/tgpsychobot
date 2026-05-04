from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import delete, func, select, update

from ..ai import AIServiceError, get_ai_response_direct
from ..api import MaxApiClient
from ..keyboards import case_study_keyboard, final_test_keyboard, secret_test_keyboard, test_answers_keyboard
from ..legacy import CaseStudy, Content, Message as DBMessage, SecretTestQuestion, TestConfig, TestQuestion, TestSession, User, async_session_maker
from ..logging_utils import get_ai_logger, get_bot_logger
from ..storage import StateStore


log = get_bot_logger("tests")
ai_log = get_ai_logger("tests")

CATEGORY_NAMES = {
    "body": "Отношение к телу",
    "face": "Отношение к лицу",
    "age": "Отношение к возрасту",
    "health": "Отношение к здоровью",
    "abilities": "Отношение к способностям",
    "relations": "Отношения с окружающими",
    "success": "Успешность/Реализация целей",
}

CATEGORY_KEYWORDS = {
    "body": ["тело", "внешность", "вес", "фигура"],
    "face": ["лицо", "красота", "зеркало", "внешность"],
    "age": ["возраст", "старение", "молодость"],
    "health": ["здоровье", "болезнь", "самочувствие", "энергия"],
    "abilities": ["способности", "талант", "ум", "компетентность"],
    "relations": ["отношения", "люди", "общение", "партнер", "семья"],
    "success": ["успех", "карьера", "деньги", "реализация", "цели"],
}


def _progress_bar(current: int, total: int, length: int = 10) -> str:
    percent = int((current / total) * 100) if total else 0
    filled = int(percent / (100 / length)) if total else 0
    return f"{'█' * filled}{'░' * (length - filled)} {percent}% ({current}/{total})"


def _build_test_diagram(questions: list[TestQuestion], answers: list[int]) -> tuple[str, list[tuple[str, float]]]:
    categories: dict[str, dict[str, int]] = {}
    total_score = 0
    total_max = 0
    for question, answer in zip(questions, answers):
        category = question.category
        categories.setdefault(category, {"score": 0, "max": 0})
        final_score = 6 - answer if question.is_reverse else answer
        categories[category]["score"] += final_score
        categories[category]["max"] += 5
        total_score += final_score
        total_max += 5

    ranking: list[tuple[str, float]] = []
    lines = ["📊 <b>ТВОЯ КАРТА САМООЦЕНКИ</b><br/>"]
    for category, values in categories.items():
        percent = values["score"] / values["max"] if values["max"] else 0
        ranking.append((category, percent))
        filled = int(percent * 10)
        status = " ✅ Сильная сторона" if percent >= 0.75 else (" ⚠️ Зона внимания" if percent <= 0.45 else "")
        lines.append(f"<br/><b>{CATEGORY_NAMES.get(category, category)}</b><br/>{'█' * filled}{'░' * (10 - filled)} {values['score']}/{values['max']}{status}")

    total_percent = int((total_score / total_max) * 100) if total_max else 0
    lines.append(f"<br/><br/><b>ОБЩИЙ ИТОГ:</b> {total_score}/{total_max} ({total_percent}%)")
    ranking.sort(key=lambda item: item[1])
    return "".join(lines), ranking


def _pick_relevant_case(case_studies: list[CaseStudy], ranking: list[tuple[str, float]]) -> CaseStudy | None:
    if not case_studies:
        return None
    weakest_categories = [item[0] for item in ranking[:3]]
    for category in weakest_categories:
        keywords = CATEGORY_KEYWORDS.get(category, []) + [CATEGORY_NAMES.get(category, "").lower()]
        for case in case_studies:
            lowered = (case.text or "").lower()
            if any(keyword and keyword in lowered for keyword in keywords):
                return case
    return case_studies[0]


def _render_case_block(case_study: CaseStudy | None, ranking: list[tuple[str, float]]) -> str:
    if not case_study:
        return "Подходящий кейс пока не найден. Администратор может добавить истории в разделе кейсов."
    weakest = ", ".join(CATEGORY_NAMES.get(category, category) for category, _ in ranking[:2]) or "общей самооценкой"
    excerpt = (case_study.text or "").strip()
    if len(excerpt) > 1800:
        excerpt = excerpt[:1800].rstrip() + "..."
    safe_excerpt = excerpt.replace("<", "&lt;").replace(">", "&gt;")
    return (
        f"<b>История-зеркало</b><br/><br/>"
        f"Ниже кейс, который ближе всего к вашим текущим зонам внимания: {weakest}.<br/><br/>"
        f"<pre><code>{safe_excerpt}</code></pre>"
    )


def _fallback_ai_summary(ranking: list[tuple[str, float]]) -> str:
    if not ranking:
        return (
            "<b>Разбор результата</b><br/><br/>"
            "Результаты уже сохранены. Продолжайте к секретному блоку вопросов, чтобы получить более точную картину."
        )
    weakest = ", ".join(CATEGORY_NAMES.get(category, category) for category, _ in ranking[:2])
    strongest = ", ".join(CATEGORY_NAMES.get(category, category) for category, _ in ranking[-2:])
    return (
        "<b>Разбор результата</b><br/><br/>"
        f"Сейчас стоит уделить больше внимания зонам: {weakest}.<br/>"
        f"При этом у вас есть ресурс в сферах: {strongest}.<br/><br/>"
        "Секретный блок поможет точнее понять внутренние причины и перейти от общей картины к конкретным шагам."
    )


async def _build_ai_summary(
    user: User | None,
    test_config: TestConfig | None,
    diagram_text: str,
    ranking: list[tuple[str, float]],
    case_study: CaseStudy | None,
) -> str:
    if not user or not test_config or not test_config.test_system_prompt:
        return _fallback_ai_summary(ranking)

    case_context = case_study.text if case_study else "Подходящий кейс не найден."
    weakest = ", ".join(CATEGORY_NAMES.get(category, category) for category, _ in ranking[:3]) or "общая самооценка"
    user_name = user.name or user.first_name or "пользователь"
    user_gender = "Женский" if user.gender == "female" else "Мужской" if user.gender == "male" else "Не определен"
    user_prompt = (
        f"Пользователь: {user_name}\n"
        f"Возраст: {user.age or 'Не указан'}\n"
        f"Пол: {user_gender}\n"
        f"Зоны внимания: {weakest}\n\n"
        f"Диаграмма результатов:\n{diagram_text}\n\n"
        f"Кейс из базы:\n{case_context}\n\n"
        "Дай теплую и конкретную расшифровку результата в 2-4 коротких абзацах."
        " Объясни, как слабые зоны влияют на жизнь, и мягко подведи к секретному блоку вопросов."
    )
    try:
        return await get_ai_response_direct(user.id, test_config.test_system_prompt, user_prompt)
    except AIServiceError:
        ai_log.exception("Test AI summary failed user_id=%s", user.id)
        return _fallback_ai_summary(ranking)
    except Exception:
        ai_log.exception("Unexpected test AI summary failure user_id=%s", user.id)
        return _fallback_ai_summary(ranking)


async def start_test(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        if not questions:
            log.warning("Test start requested without questions user_id=%s chat_id=%s", user_id, chat_id)
            await client.send_message(chat_id=chat_id, text="Вопросы теста не загружены.")
            return
        await session.execute(delete(TestSession).where(TestSession.user_id == user_id))
        session.add(TestSession(user_id=user_id, current_question_index=0, answers=""))
        await session.commit()
    await _send_question(client, chat_id, user_id, 0)


async def _send_question(client: MaxApiClient, chat_id: int, user_id: int, index: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
    question = questions[index]
    suffix = "ен" if user and user.gender == "male" else "на"
    neutral = "Нейтрален" if user and user.gender == "male" else "Нейтральна"
    legend = (
        f"1 — Совершенно не соглас{suffix}<br/>"
        f"2 — Скорее не соглас{suffix}<br/>"
        f"3 — {neutral}<br/>"
        f"4 — Скорее соглас{suffix}<br/>"
        f"5 — Полностью соглас{suffix}"
    )
    text = (
        f"<b>Вопрос {index + 1}/{len(questions)}</b><br/>"
        f"{_progress_bar(index, len(questions))}<br/><br/>"
        f"<b>{question.text}</b><br/><br/>{legend}"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=test_answers_keyboard())


async def process_answer(client: MaxApiClient, chat_id: int, user_id: int, answer_value: int) -> None:
    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id)
        if not test_session or test_session.is_finished:
            await client.send_message(chat_id=chat_id, text="Сессия теста не найдена.")
            return
        answers = [int(item) for item in test_session.answers.split(",") if item]
        answers.append(answer_value)
        test_session.answers = ",".join(str(item) for item in answers)
        test_session.current_question_index += 1
        total_questions = await session.scalar(select(func.count()).select_from(TestQuestion)) or 0
        await session.commit()

    if test_session.current_question_index < total_questions:
        await _send_question(client, chat_id, user_id, test_session.current_question_index)
        return

    async with async_session_maker() as session:
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        answers = [int(item) for item in test_session.answers.split(",") if item]
        await session.execute(update(TestSession).where(TestSession.user_id == user_id).values(is_finished=True))
        await session.commit()

    await client.send_message(chat_id=chat_id, text="Диаграмма готова. Ниже можно открыть результат и историю-зеркало.", attachments=case_study_keyboard())


async def show_results(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id)
        test_config = await session.get(TestConfig, 1)
        content = await session.get(Content, "test_results")
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        case_studies = (await session.execute(select(CaseStudy).order_by(CaseStudy.id.desc()))).scalars().all()
        user = await session.get(User, user_id)
    if content and content.text_content:
        await client.send_message(chat_id=chat_id, text=content.text_content)
    answers_raw = [int(item) for item in (test_session.answers or "").split(",") if item.isdigit()] if test_session else []
    ranking: list[tuple[str, float]] = []
    diagram_text = ""
    if answers_raw and questions:
        diagram_text, ranking = _build_test_diagram(questions, answers_raw)
        await client.send_message(chat_id=chat_id, text=diagram_text)
    case_study = _pick_relevant_case(case_studies, ranking)
    await client.send_message(chat_id=chat_id, text=_render_case_block(case_study, ranking))
    ai_summary = await _build_ai_summary(user, test_config, diagram_text, ranking, case_study)
    await client.send_message(chat_id=chat_id, text=ai_summary)
    async with async_session_maker() as session:
        if ai_summary:
            session.add(
                DBMessage(
                    user_id=user_id,
                    role="assistant",
                    content=ai_summary,
                    dialogue_id=user.current_dialogue_id if user else 1,
                    topic_id=user.current_topic_id if user else None,
                )
            )
            await session.commit()
    text = "Секретный блок вопросов уже доступен. После него можно продолжить диалог с ботом по результатам теста."
    marathon_url = test_config.marathon_url if test_config else "https://max.ru"
    await client.send_message(chat_id=chat_id, text=text, attachments=secret_test_keyboard(marathon_url))


async def start_secret_test(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        questions = (await session.execute(select(SecretTestQuestion).order_by(SecretTestQuestion.sort_order.asc()))).scalars().all()
    if not questions:
        await client.send_message(chat_id=chat_id, text="Вопросы секретного теста ещё не добавлены.")
        return
    text = "<b>🔐 Секретный блок вопросов</b><br/><br/>Ответьте одним сообщением:<br/><br/>"
    for index, question in enumerate(questions, start=1):
        text += f"{index}. {question.text}<br/><br/>"
    await states.set(user_id, chat_id, "secret_test_answering", {})
    await client.send_message(chat_id=chat_id, text=text)


async def save_secret_answers(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        await session.execute(update(TestSession).where(TestSession.user_id == user_id).values(secret_answers=text))
        content = await session.get(Content, "secret_test_outro")
        config = await session.get(TestConfig, 1)
        await session.commit()
    await states.clear(user_id)
    await client.send_message(
        chat_id=chat_id,
        text=(content.text_content if content and content.text_content else "Спасибо за ответы!"),
        attachments=final_test_keyboard(config.marathon_url if config else "https://max.ru"),
    )
