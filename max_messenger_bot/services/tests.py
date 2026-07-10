from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import delete, func, select, update

from ..ai import AIServiceError, get_ai_response, get_ai_response_direct
from ..api import MaxApiClient
from ..keyboards import case_study_keyboard, final_test_keyboard, secret_test_keyboard, universal_test_answers_keyboard
from ..legacy import CaseStudy, Content, Message as DBMessage, SecretTestQuestion, TestConfig, TestQuestion, TestSession, User, async_session_maker
from ..logging_utils import get_ai_logger, get_bot_logger
from ..storage import StateStore
from universal_tests import (
    build_result_handoff_prompt,
    build_prompt_payload,
    build_question_text,
    calculate_formulas,
    get_answer_options,
    json_dumps,
    json_loads,
    is_universal_test_report,
    make_answer_record,
    make_option_answer_record,
    make_text_answer_record,
    parse_answers,
    question_buttons_are_horizontal,
    serialize_answers,
)


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
    lines = ["📊 <b>ТВОЯ КАРТА САМООЦЕНКИ</b>\n"]
    for category, values in categories.items():
        percent = values["score"] / values["max"] if values["max"] else 0
        ranking.append((category, percent))
        filled = int(percent * 10)
        status = " ✅ Сильная сторона" if percent >= 0.75 else (" ⚠️ Зона внимания" if percent <= 0.45 else "")
        lines.append(f"\n<b>{CATEGORY_NAMES.get(category, category)}</b>\n{'█' * filled}{'░' * (10 - filled)} {values['score']}/{values['max']}{status}")

    total_percent = int((total_score / total_max) * 100) if total_max else 0
    lines.append(f"\n\n<b>ОБЩИЙ ИТОГ:</b> {total_score}/{total_max} ({total_percent}%)")
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
        f"<b>История-зеркало</b>\n\n"
        f"Ниже кейс, который ближе всего к вашим текущим зонам внимания: {weakest}.\n\n"
        f"<pre><code>{safe_excerpt}</code></pre>"
    )


def _fallback_ai_summary(ranking: list[tuple[str, float]]) -> str:
    if not ranking:
        return (
            "<b>Разбор результата</b>\n\n"
            "Результаты уже сохранены. Продолжайте к секретному блоку вопросов, чтобы получить более точную картину."
        )
    weakest = ", ".join(CATEGORY_NAMES.get(category, category) for category, _ in ranking[:2])
    strongest = ", ".join(CATEGORY_NAMES.get(category, category) for category, _ in ranking[-2:])
    return (
        "<b>Разбор результата</b>\n\n"
        f"Сейчас стоит уделить больше внимания зонам: {weakest}.\n"
        f"При этом у вас есть ресурс в сферах: {strongest}.\n\n"
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


async def start_test(client: MaxApiClient, chat_id: int, user_id: int, states: StateStore | None = None) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        config = await session.get(TestConfig, 1)
        if config and not config.is_enabled and not bool(getattr(user, "is_admin", False)):
            await client.send_message(chat_id=chat_id, text="Тестирование сейчас отключено.")
            return
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        if not questions:
            log.warning("Test start requested without questions user_id=%s chat_id=%s", user_id, chat_id)
            await client.send_message(chat_id=chat_id, text="Вопросы теста не загружены.")
            return
        await session.execute(delete(TestSession).where(TestSession.user_id == user_id))
        session.add(TestSession(
            user_id=user_id,
            current_question_index=0,
            answers="[]",
            invocation_topic_id=user.current_topic_id if user else None,
            invocation_dialogue_id=user.current_dialogue_id if user else 1,
            invocation_platform="max",
        ))
        await session.commit()
    if states:
        await states.set(user_id, chat_id, "test_answering", {})
    await _send_question(client, chat_id, user_id, 0)


async def _send_question(client: MaxApiClient, chat_id: int, user_id: int, index: int) -> None:
    async with async_session_maker() as session:
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        config = await session.get(TestConfig, 1)
    question = questions[index]
    options = get_answer_options(question)
    text = build_question_text(question, index, len(questions), getattr(config, "show_progress", True) if config else True)
    keyboard = universal_test_answers_keyboard(options, question_buttons_are_horizontal(question), index) if options else None
    await client.send_message(chat_id=chat_id, text=text, attachments=keyboard)


async def process_answer(client: MaxApiClient, chat_id: int, user_id: int, answer_payload: int | str, states: StateStore | None = None) -> None:
    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id, with_for_update=True)
        if not test_session or test_session.is_finished:
            await client.send_message(chat_id=chat_id, text="Сессия теста не найдена.")
            return
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        question_index = test_session.current_question_index
        if question_index >= len(questions):
            await client.send_message(chat_id=chat_id, text="Вопросы теста уже закончились.")
            return
        question = questions[question_index]
        options = get_answer_options(question)
        if isinstance(answer_payload, str) and answer_payload.startswith("test_opt_"):
            try:
                answer_record = make_option_answer_record(question, question_index, answer_payload)
            except ValueError as exc:
                await client.send_message(chat_id=chat_id, text=str(exc))
                return
        else:
            answer_text = str(answer_payload)
            numeric_value = float(answer_text) if answer_text.isdigit() else None
            answer_record = make_answer_record(question, question_index, answer_text, numeric_value)

        answers = parse_answers(test_session.answers)
        answers.append(answer_record)
        test_session.answers = serialize_answers(answers)
        test_session.current_question_index += 1
        await session.commit()
        next_index = test_session.current_question_index
        total_questions = len(questions)

    if next_index < total_questions:
        await _send_question(client, chat_id, user_id, next_index)
        return

    await _finish_universal_test(client, chat_id, user_id, states)
    await client.send_message(chat_id=chat_id, text="Диаграмма готова. Ниже можно открыть результат и историю-зеркало.", attachments=case_study_keyboard())


async def process_text_answer(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id, with_for_update=True)
        if not test_session or test_session.is_finished:
            await states.clear(user_id)
            await client.send_message(chat_id=chat_id, text="Сессия теста не найдена.")
            return
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        question_index = test_session.current_question_index
        if question_index >= len(questions):
            await _finish_universal_test(client, chat_id, user_id, states)
            return
        question = questions[question_index]
        try:
            answer_record = make_text_answer_record(question, question_index, text)
        except ValueError as exc:
            await client.send_message(chat_id=chat_id, text=str(exc))
            return
        answers = parse_answers(test_session.answers)
        answers.append(answer_record)
        test_session.answers = serialize_answers(answers)
        test_session.current_question_index += 1
        await session.commit()
        next_index = test_session.current_question_index

    if next_index < len(questions):
        await _send_question(client, chat_id, user_id, next_index)
    else:
        await _finish_universal_test(client, chat_id, user_id, states)
        await client.send_message(chat_id=chat_id, text="Диаграмма готова. Ниже можно открыть результат и историю-зеркало.", attachments=case_study_keyboard())


async def _finish_universal_test(client: MaxApiClient, chat_id: int, user_id: int, states: StateStore | None = None) -> None:
    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id)
        test_config = await session.get(TestConfig, 1)
        questions = (await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))).scalars().all()
        answers = parse_answers(test_session.answers if test_session else "[]")
        formulas = json_loads(test_config.formulas_json, []) if test_config and test_config.formulas_enabled else []
        try:
            formula_results = calculate_formulas(answers, formulas) if formulas else {}
        except Exception:
            ai_log.exception("Universal Max test formula calculation failed user_id=%s", user_id)
            formula_results = {}
        report_text = build_prompt_payload(questions, answers, formula_results, mode="all")
        input_mode = getattr(test_config, "interpretation_input_mode", "all") if test_config else "all"
        selected_variables = json_loads(getattr(test_config, "interpretation_selected_variables", None), []) if test_config else []
        prompt_payload = build_prompt_payload(
            questions,
            answers,
            formula_results,
            mode=input_mode,
            selected_variables=selected_variables,
        )
        topic_id = getattr(test_session, "invocation_topic_id", None) if test_session else None
        dialogue_id = getattr(test_session, "invocation_dialogue_id", None) if test_session else None
        user = await session.get(User, user_id)
        await session.execute(
            update(TestSession)
            .where(TestSession.user_id == user_id)
            .values(
                answers=report_text,
                formula_results=json_dumps(formula_results) if formula_results else None,
                is_finished=True,
            )
        )
        await session.commit()
    if states:
        await states.clear(user_id)

    if not user or not test_config:
        return

    preliminary = None
    try:
        if getattr(test_config, "separate_result_prompt_enabled", False) and getattr(test_config, "result_system_prompt", None):
            preliminary = await get_ai_response_direct(
                user_id,
                test_config.result_system_prompt,
                f"Интерпретируй результаты теста.\n\n{prompt_payload}\n\n"
                "Не показывай технические имена переменных и формул. Не придумывай максимальные баллы, "
                "знаменатели или нормы, которых нет во входных данных.",
            )
        profile_name = getattr(user, "name", None) or getattr(user, "first_name", None) or None
        final_prompt = build_result_handoff_prompt(prompt_payload, preliminary, profile_name)
        final_text = await get_ai_response(
            user_id,
            final_prompt,
            topic_id_override=topic_id,
            dialogue_id_override=dialogue_id,
        )
    except Exception:
        ai_log.exception("Universal Max test final interpretation failed user_id=%s", user_id)
        final_text = preliminary or "Интерпретация результата сейчас недоступна. Попробуйте открыть результат позже."

    await client.send_message(chat_id=chat_id, text=final_text)
    async with async_session_maker() as session:
        session.add(DBMessage(
            user_id=user_id,
            role="assistant",
            content=final_text,
            dialogue_id=dialogue_id or user.current_dialogue_id,
            topic_id=topic_id,
        ))
        await session.commit()


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
    if test_session and is_universal_test_report(test_session.answers):
        marathon_url = test_config.marathon_url if test_config else "https://max.ru"
        await client.send_message(
            chat_id=chat_id,
            text="Секретный блок вопросов уже доступен. После него можно продолжить диалог с ботом по результатам теста.",
            attachments=secret_test_keyboard(marathon_url),
        )
        return
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
    text = "<b>🔐 Секретный блок вопросов</b>\n\nОтветьте одним сообщением:\n\n"
    for index, question in enumerate(questions, start=1):
        text += f"{index}. {question.text}\n\n"
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
