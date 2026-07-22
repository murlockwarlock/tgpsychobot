import os
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import Base, TestAttempt as DBTestAttempt, User
from result_history import (
    attach_secret_answers,
    build_test_attempt_pages,
    save_test_attempt,
    attempt_to_dict,
)


class TestAttemptHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.sessions() as session:
            session.add(User(id=123, first_name="Иван", current_dialogue_id=1))
            await session.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_saves_every_attempt_and_updates_same_session_idempotently(self):
        first_started = datetime(2026, 7, 22, 8, 0)
        second_started = first_started + timedelta(hours=1)
        async with self.sessions() as session:
            first = await save_test_attempt(
                session,
                user_id=123,
                source_session_created_at=first_started,
                completed_at=first_started + timedelta(minutes=10),
                platform="telegram",
                topic_id=None,
                dialogue_id=1,
                answers=[{"question_number": 1, "question": "Как дела?", "answer": "Хорошо", "numeric_value": 2}],
                report_text="Первый результат",
                formula_results={"Итого": 2},
                interpretation_text="Первая интерпретация",
            )
            await session.commit()
            first_id = first.id

        async with self.sessions() as session:
            updated = await save_test_attempt(
                session,
                user_id=123,
                source_session_created_at=first_started,
                completed_at=first_started + timedelta(minutes=11),
                platform="telegram",
                topic_id=None,
                dialogue_id=1,
                answers=[{"question_number": 1, "question": "Как дела?", "answer": "Отлично", "numeric_value": 3}],
                report_text="Обновлённый результат",
                formula_results={"Итого": 3},
                interpretation_text="Обновлённая интерпретация",
            )
            await save_test_attempt(
                session,
                user_id=123,
                source_session_created_at=second_started,
                completed_at=second_started + timedelta(minutes=5),
                platform="max",
                topic_id=None,
                dialogue_id=2,
                answers=[],
                report_text="Второй результат",
                formula_results={},
                interpretation_text="Вторая интерпретация",
            )
            await session.commit()
            self.assertEqual(updated.id, first_id)

        async with self.sessions() as session:
            count = await session.scalar(select(func.count(DBTestAttempt.id)))
            self.assertEqual(count, 2)

    async def test_builds_readable_pages_and_hides_service_markup(self):
        attempt = DBTestAttempt(
            id=1,
            user_id=123,
            completed_at=datetime(2026, 7, 22, 9, 30),
            platform="telegram",
            dialogue_id=7,
            answers_json='[{"question_number":1,"question":"Ваш выбор?","answer":"Первый","numeric_value":2}]',
            formula_results_json='{"Смысл":9}',
            interpretation_text=(
                "Итоговая интерпретация.\n"
                "[Продолжить](btn:да)\n"
                "[DATA]{\"hidden\":true}[/DATA]"
            ),
        )

        pages = build_test_attempt_pages([attempt], client_name="Иван")
        rendered = "\n".join(page["html"] for page in pages)

        self.assertIn("Расчётные показатели", rendered)
        self.assertIn("Смысл", rendered)
        self.assertIn("Ваш выбор?", rendered)
        self.assertIn("Итоговая интерпретация.", rendered)
        self.assertNotIn("btn:да", rendered)
        self.assertNotIn("hidden", rendered)

    async def test_secret_answers_attach_to_latest_attempt(self):
        async with self.sessions() as session:
            older = DBTestAttempt(user_id=123, completed_at=datetime(2026, 7, 22, 8, 0))
            latest = DBTestAttempt(user_id=123, completed_at=datetime(2026, 7, 22, 9, 0))
            session.add_all([older, latest])
            await session.commit()

        async with self.sessions() as session:
            await attach_secret_answers(session, 123, "Секретный ответ")
            await session.commit()

        async with self.sessions() as session:
            attempts = (await session.execute(select(DBTestAttempt).order_by(DBTestAttempt.completed_at))).scalars().all()
            self.assertIsNone(attempts[0].secret_answers)
            self.assertEqual(attempts[1].secret_answers, "Секретный ответ")

    def test_export_contains_structured_values(self):
        attempt = DBTestAttempt(
            id=5,
            user_id=123,
            completed_at=datetime(2026, 7, 22, 10, 0),
            answers_json='[{"question":"Вопрос","answer":"Ответ"}]',
            formula_results_json='{"Итого":4}',
        )

        exported = attempt_to_dict(attempt)

        self.assertEqual(exported["answers"][0]["answer"], "Ответ")
        self.assertEqual(exported["formula_results"], {"Итого": 4})

    def test_long_markup_is_escaped_and_split_within_telegram_limit(self):
        attempt = DBTestAttempt(
            id=8,
            user_id=123,
            completed_at=datetime(2026, 7, 22, 11, 0),
            answers_json=json.dumps([
                {
                    "question_number": 1,
                    "question": "<Вопрос & проверка>" * 300,
                    "answer": "<script>&" * 1500,
                    "numeric_value": 7,
                }
            ], ensure_ascii=False),
            formula_results_json=json.dumps({"<Формула>": "<&>" * 2000}, ensure_ascii=False),
            interpretation_text="<b>Не разметка</b> & текст " * 1000,
        )

        pages = build_test_attempt_pages(
            [attempt],
            client_name="<Иван & Co>" * 100,
            topic_names={None: "<Тема>" * 100},
        )

        self.assertGreater(len(pages), 5)
        self.assertTrue(all(len(page["html"]) < 3900 for page in pages))
        rendered = "\n".join(page["html"] for page in pages)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;b&gt;Не разметка&lt;/b&gt;", rendered)

    def test_database_initialization_backfills_legacy_result_once(self):
        script = textwrap.dedent("""
            import asyncio
            from datetime import datetime
            from sqlalchemy import func, select
            from database import TestAttempt, TestSession, User, async_session_maker, init_db

            async def main():
                await init_db()
                async with async_session_maker() as session:
                    session.add(User(id=987654321, first_name="Legacy"))
                    session.add(TestSession(
                        user_id=987654321,
                        answers="legacy report",
                        formula_results='{"total":5}',
                        invocation_platform="telegram",
                        is_finished=True,
                        created_at=datetime(2026, 7, 20, 10, 0),
                    ))
                    await session.commit()

                await init_db()
                await init_db()
                async with async_session_maker() as session:
                    count = await session.scalar(
                        select(func.count(TestAttempt.id)).where(TestAttempt.user_id == 987654321)
                    )
                    attempt = await session.scalar(
                        select(TestAttempt).where(TestAttempt.user_id == 987654321)
                    )
                    assert count == 1
                    assert attempt.report_text == "legacy report"
                    assert attempt.formula_results_json == '{"total":5}'

            asyncio.run(main())
        """)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["BOT_TOKEN"] = "test"
            env["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmpdir}/migration.db"
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
