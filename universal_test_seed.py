from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import delete

from database import TestConfig, TestQuestion, async_session_maker, engine
from file_parser import parse_questions_file
from universal_tests import json_dumps, validate_test_definition


def _preview_question(data: dict) -> SimpleNamespace:
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


async def seed(path: Path) -> tuple[int, int]:
    parsed = await parse_questions_file(io.BytesIO(path.read_bytes()), path.name)
    questions = parsed.get("questions", [])
    formulas = parsed.get("formulas", [])
    if not questions:
        raise ValueError("В файле не найдено вопросов.")

    errors = validate_test_definition([_preview_question(item) for item in questions], formulas)
    if errors:
        raise ValueError("Файл не прошёл проверку:\n- " + "\n- ".join(errors))

    async with async_session_maker() as session:
        async with session.begin():
            await session.execute(delete(TestQuestion))
            for index, item in enumerate(questions):
                session.add(TestQuestion(sort_order=index, **item))
            config = await session.get(TestConfig, 1)
            if not config:
                config = TestConfig(id=1)
                session.add(config)
            config.formulas_json = json_dumps(formulas) if formulas else None
            config.formulas_enabled = bool(formulas)
    return len(questions), len(formulas)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and seed a universal test into the configured database.")
    parser.add_argument("file", type=Path, help="Path to .csv, .txt or .xlsx test file")
    args = parser.parse_args()
    try:
        question_count, formula_count = await seed(args.file)
        print(f"Seed complete: questions={question_count} formulas={formula_count}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
