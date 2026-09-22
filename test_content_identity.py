from types import SimpleNamespace

from sqlalchemy import select

from database import TestQuestion, TestSession
from universal_tests import json_dumps, json_loads


QUESTION_FIELDS = ("id", "text", "category", "is_reverse", "sort_order", "comment", "variable_name", "allow_custom_answer", "buttons_layout", "answer_options_json")


def question_snapshot(questions) -> str:
    from translation_service import translation_cache
    result = []
    for question in questions:
        item = {field: getattr(question, field, None) for field in QUESTION_FIELDS}
        prefix = f"test_question.{question.id}."
        item["localized_snapshot"] = {
            locale + ":" + key: value
            for (locale, key), value in translation_cache.snapshot.translations.items()
            if key.startswith(prefix)
        }
        result.append(item)
    return json_dumps(result)


def translate_question(question, key, locale, *, source=None, fallback=None):
    from translation_service import translate
    snapshot = getattr(question, "localized_snapshot", None)
    if snapshot is not None:
        return (snapshot.get(locale + ":" + key) if locale != "ru" else None) or source or fallback
    return translate(key, locale, source=source, fallback=fallback)


def questions_for_session(test_session, live_questions):
    snapshot = json_loads(getattr(test_session, "question_snapshot", None), None)
    if isinstance(snapshot, list):
        return [SimpleNamespace(**question) for question in snapshot]
    return live_questions


async def preserve_active_test_definitions(session) -> None:
    questions = (await session.scalars(select(TestQuestion).order_by(TestQuestion.sort_order, TestQuestion.id))).all()
    snapshot = question_snapshot(questions)
    active = (await session.scalars(select(TestSession).where(TestSession.is_finished == False, TestSession.question_snapshot == None))).all()
    for test_session in active:
        test_session.question_snapshot = snapshot


async def move_question(session, question_id: int, direction: int) -> None:
    await preserve_active_test_definitions(session)
    questions = (await session.scalars(select(TestQuestion).order_by(TestQuestion.sort_order, TestQuestion.id))).all()
    index = next(index for index, item in enumerate(questions) if item.id == question_id)
    target = index + direction
    if 0 <= target < len(questions):
        questions[index], questions[target] = questions[target], questions[index]
        for order, question in enumerate(questions):
            question.sort_order = order
