import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import handlers
from database import Base, TestQuestion as DBTestQuestion, TestSession as DBTestSession, User
from handlers import UserStates


def make_message(user_id=10, text="Привет"):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        text=text,
        answer=AsyncMock(),
        delete=AsyncMock(),
    )


async def make_database(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, sessions


@pytest.mark.asyncio
async def test_stale_session_is_finished_and_does_not_capture_normal_chat(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(get_state=AsyncMock(return_value=None), set_state=AsyncMock())
    process_answer = AsyncMock()
    monkeypatch.setattr(handlers, "process_test_text_answer", process_answer)
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestSession(
                    user_id=10,
                    created_at=datetime.utcnow() - timedelta(days=60),
                    current_question_index=0,
                    answers="[]",
                    is_finished=False,
                ),
            ])
            await session.commit()

        assert await handlers._restore_test_state_from_db(make_message(), state, object()) is False
        state.set_state.assert_not_awaited()
        process_answer.assert_not_awaited()

        async with sessions() as session:
            stored = await session.get(DBTestSession, 10)
            assert stored.is_finished is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_stale_message_continues_into_normal_chat_path(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(
        get_state=AsyncMock(return_value=None),
        set_state=AsyncMock(),
        clear=AsyncMock(),
        update_data=AsyncMock(),
    )
    message = make_message()
    message.from_user.username = "tester"
    message.from_user.full_name = "Test User"
    process_answer = AsyncMock()
    schedule_runner = MagicMock()

    @asynccontextmanager
    async def scheduling_lock(user_id):
        yield

    monkeypatch.setattr(handlers, "process_test_text_answer", process_answer)
    monkeypatch.setattr(handlers, "_get_user_locale", AsyncMock(return_value="ru"))
    monkeypatch.setattr(handlers, "_sync_user_birthdate_from_telegram", AsyncMock())
    monkeypatch.setattr(handlers, "_request_profile_onboarding_if_needed", AsyncMock(return_value=False))
    monkeypatch.setattr(handlers, "_get_user_scheduling_lock", scheduling_lock)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers.single_flight, "try_claim", lambda *args: "lease")
    monkeypatch.setattr(handlers, "_ensure_telegram_drain_runner", schedule_runner)
    monkeypatch.setattr(handlers, "user_message_buffers", {})
    monkeypatch.setattr(handlers, "user_message_buffer_leases", {})
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester", accepted_disclaimer=True),
                DBTestSession(
                    user_id=10,
                    created_at=datetime.utcnow() - timedelta(days=60),
                    is_finished=False,
                ),
            ])
            await session.commit()

        await handlers.handle_ai_chat(message, state, object())

        process_answer.assert_not_awaited()
        assert handlers.user_message_buffers[10] == ["Привет"]
        schedule_runner.assert_called_once()
        async with sessions() as session:
            stored = await session.get(DBTestSession, 10)
            assert stored.is_finished is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_null_created_at_session_is_finished_instead_of_becoming_immortal(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(get_state=AsyncMock(return_value=None), set_state=AsyncMock())
    try:
        async with sessions() as session:
            session.add_all([User(id=10, first_name="Tester"), DBTestSession(user_id=10)])
            await session.commit()
            stored = await session.get(DBTestSession, 10)
            stored.created_at = None
            await session.commit()

        assert await handlers._restore_test_state_from_db(make_message(), state, object()) is False
        state.set_state.assert_not_awaited()

        async with sessions() as session:
            stored = await session.get(DBTestSession, 10)
            assert stored.is_finished is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fresh_session_still_restores_after_fsm_loss(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(get_state=AsyncMock(return_value=None), set_state=AsyncMock())
    process_answer = AsyncMock()
    monkeypatch.setattr(handlers, "process_test_text_answer", process_answer)
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestSession(
                    user_id=10,
                    created_at=datetime.utcnow() - timedelta(hours=1),
                    current_question_index=2,
                    answers='[{"value": 1}]',
                    is_finished=False,
                ),
            ])
            await session.commit()

        assert await handlers._restore_test_state_from_db(make_message(), state, object()) is True
        state.set_state.assert_awaited_once_with(UserStates.in_test)
        process_answer.assert_awaited_once()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_finished_session_never_restores(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(get_state=AsyncMock(return_value=None), set_state=AsyncMock())
    process_answer = AsyncMock()
    monkeypatch.setattr(handlers, "process_test_text_answer", process_answer)
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestSession(user_id=10, is_finished=True),
            ])
            await session.commit()

        assert await handlers._restore_test_state_from_db(make_message(), state, object()) is False
        state.set_state.assert_not_awaited()
        process_answer.assert_not_awaited()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_callback_cancel_finishes_session_and_survives_restart(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(clear=AsyncMock(), get_state=AsyncMock(return_value=None), set_state=AsyncMock())
    monkeypatch.setattr(handlers, "_get_user_locale", AsyncMock(return_value="ru"))
    monkeypatch.setattr(handlers.kb, "main_client_keyboard", AsyncMock(return_value="main-menu"))
    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=10),
        message=SimpleNamespace(answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestSession(user_id=10, question_message_id=77, is_finished=False),
            ])
            await session.commit()

        await handlers.process_cancel_test(callback, state, object())

        async with sessions() as session:
            stored = await session.get(DBTestSession, 10)
            assert stored.is_finished is True
            assert stored.question_message_id is None
        state.clear.assert_awaited_once()

        process_answer = AsyncMock()
        monkeypatch.setattr(handlers, "process_test_text_answer", process_answer)
        assert await handlers._restore_test_state_from_db(make_message(), state, object()) is False
        process_answer.assert_not_awaited()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_text", ["/cancel", " /CANCEL ", "/cancel@BotUsername", "отмена", "СТОП", " выйти ", "выход"])
async def test_text_cancel_finishes_session(tmp_path, monkeypatch, cancel_text):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(clear=AsyncMock())
    monkeypatch.setattr(handlers, "_get_user_locale", AsyncMock(return_value="ru"))
    monkeypatch.setattr(handlers.kb, "main_client_keyboard", AsyncMock(return_value="main-menu"))
    message = make_message(text=cancel_text)
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestSession(user_id=10, question_message_id=77, is_finished=False),
            ])
            await session.commit()

        await handlers.process_test_text_answer(message, state, object())

        async with sessions() as session:
            stored = await session.get(DBTestSession, 10)
            assert stored.is_finished is True
            assert stored.question_message_id is None
        state.clear.assert_awaited_once()
        message.answer.assert_awaited_once()
    finally:
        await engine.dispose()


def test_text_cancel_requires_an_exact_command():
    assert handlers._is_test_cancel_text("/cancel") is True
    assert handlers._is_test_cancel_text("/cancel@BotUsername") is True
    assert handlers._is_test_cancel_text("не могу остановиться") is False
    assert handlers._is_test_cancel_text("/cancel later") is False


@pytest.mark.asyncio
async def test_invalid_text_restores_current_option_keyboard_without_mutating_session(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(clear=AsyncMock())
    bot = SimpleNamespace(delete_message=AsyncMock())
    message = make_message(text="свой ответ")
    monkeypatch.setattr(handlers, "_get_user_locale", AsyncMock(return_value="ru"))
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestQuestion(
                    id=1,
                    text="Выберите вариант",
                    category="general",
                    answer_options_json=json.dumps([
                        {"text": "Нет", "value": 0},
                        {"text": "Да", "value": 1},
                    ]),
                    allow_custom_answer=False,
                ),
                DBTestSession(user_id=10, current_question_index=0, answers="[]", is_finished=False),
            ])
            await session.commit()

        await handlers.process_test_text_answer(message, state, bot)

        async with sessions() as session:
            stored = await session.get(DBTestSession, 10)
            assert stored.current_question_index == 0
            assert stored.answers == "[]"
            assert stored.is_finished is False
        markup = message.answer.await_args.kwargs["reply_markup"]
        labels = [button.text for row in markup.inline_keyboard for button in row]
        assert labels[:2] == ["Нет", "Да"]
        assert labels[-1] == "❌ Выйти из теста"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_free_text_question_still_accepts_text(tmp_path, monkeypatch):
    engine, sessions = await make_database(tmp_path, monkeypatch)
    state = SimpleNamespace(clear=AsyncMock())
    bot = SimpleNamespace(delete_message=AsyncMock())
    message = make_message(text="мой ответ")
    monkeypatch.setattr(handlers, "_get_user_locale", AsyncMock(return_value="ru"))
    monkeypatch.setattr(handlers, "finish_test_generation", AsyncMock())
    try:
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestQuestion(
                    id=1,
                    text="Расскажите",
                    category="general",
                    answer_options_json=None,
                    allow_custom_answer=False,
                ),
                DBTestSession(user_id=10, current_question_index=0, answers="[]", is_finished=False),
            ])
            await session.commit()

        await handlers.process_test_text_answer(message, state, bot)

        async with sessions() as session:
            stored = await session.get(DBTestSession, 10)
            assert stored.current_question_index == 1
            assert json.loads(stored.answers)[0]["answer"] == "мой ответ"
            assert stored.is_finished is False
    finally:
        await engine.dispose()
