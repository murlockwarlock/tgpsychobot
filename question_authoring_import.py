from sqlalchemy import select

from content_authoring import create_resource, ensure_answer_identities, save_content_value
from database import TestQuestion
from test_content_identity import preserve_active_test_definitions


async def apply_question_import(session, rows, locale):
    existing = {question.id: question for question in (await session.scalars(select(TestQuestion))).all()}
    ids = [row.get("id") for row in rows if row.get("id") is not None]
    if len(ids) != len(set(ids)):
        raise ValueError("ID вопросов в файле повторяются.")
    if existing and any(row.get("id") is None for row in rows):
        raise ValueError("Для обновления существующего теста укажите ID вопроса в первой колонке с заголовком ID. Новые вопросы добавляются через «Вопросы теста → Добавить».")
    if any(identity not in existing for identity in ids):
        raise ValueError("Неизвестный ID вопроса. Файл не применён.")
    if existing and any(row.get("answer_options_json") for row in rows):
        raise ValueError("Варианты ответа редактируются в карточке вопроса: файл без ID вариантов не может безопасно изменить их привязки. Уберите колонки вариантов из файла.")
    await preserve_active_test_definitions(session)
    for index, row in enumerate(rows):
        if row.get("id") is not None:
            question = existing[row["id"]]
            await save_content_value(session, "test_question", question, "text", locale, row["text"])
            await save_content_value(session, "test_question", question, "comment", locale, row.get("comment") or "")
        else:
            question = await create_resource(session, "test_question", locale, {"text": row["text"], "comment": row.get("comment") or ""}, {
                "sort_order": index, "category": row["category"], "is_reverse": row["is_reverse"],
                "variable_name": row.get("variable_name"), "allow_custom_answer": row.get("allow_custom_answer", False),
                "buttons_layout": row.get("buttons_layout", "vertical"), "answer_options_json": row.get("answer_options_json"),
            })
            items = await ensure_answer_identities(session, question)
            if locale != "ru":
                for item in items:
                    for field in ("text", "button_text"):
                        value = item[field] or ""
                        await save_content_value(session, "test_question", question, f"option.{item['translation_slot']}.{field}", "ru", "")
                        await save_content_value(session, "test_question", question, f"option.{item['translation_slot']}.{field}", locale, value)
