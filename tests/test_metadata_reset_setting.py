from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
import max_messenger_bot.services.admin_ai as max_admin_ai
from automation_engine import build_runtime_automation_context
from database import (
    AIConfig,
    AutomationConversationState,
    AutomationDialogueState,
    AutomationEvent,
    AutomationMetadataRecord,
    AutomationStepTransition,
    Base,
    CardSpreadState,
    Message,
    TestSession,
    User,
)
from dialogue_reset_policy import start_new_dialogue_scope
from max_messenger_bot.services import common as max_common
from memory_mode import MEMORY_MODE_GLOBAL, MEMORY_MODE_TOPIC


@pytest_asyncio.fixture
async def metadata_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'metadata-reset.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin_ai, "async_session_maker", sessions)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)
    try:
        yield sessions
    finally:
        await engine.dispose()


def _telegram_callback(data: str):
    return SimpleNamespace(
        data=data,
        answer=AsyncMock(),
        message=SimpleNamespace(edit_text=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_metadata_setting_defaults_to_reset_and_reopens_for_tg_and_max(metadata_db):
    async with metadata_db() as session:
        session.add(AIConfig(id=1))
        await session.commit()

    tg_callback = _telegram_callback("admin_ai_metadata_reset")
    await handlers.open_metadata_reset_settings(tg_callback)
    tg_markup = tg_callback.message.edit_text.await_args.kwargs["reply_markup"]
    tg_buttons = [button for row in tg_markup.inline_keyboard for button in row]
    assert [button.callback_data for button in tg_buttons[:2]] == [
        "admin_ai_set_metadata_reset_reset",
        "admin_ai_set_metadata_reset_preserve",
    ]
    assert tg_buttons[0].text.startswith("✅ ")

    tg_save = _telegram_callback("admin_ai_set_metadata_reset_preserve")
    await handlers.set_metadata_reset_settings(tg_save)
    assert tg_save.message.edit_text.await_args.kwargs["reply_markup"].inline_keyboard[1][0].text.startswith("✅ ")

    max_client = SimpleNamespace(send_message=AsyncMock())
    await max_admin_ai.show_common(max_client, 100)
    assert "Метаданные при новом диалоге: <b>Сохранять</b>" in max_client.send_message.await_args.kwargs["text"]
    await max_admin_ai.show_metadata_reset(max_client, 100)
    max_markup = max_client.send_message.await_args.kwargs["attachments"][0]["payload"]["buttons"]
    assert max_markup[1][0]["text"].startswith("✅ ")

    await max_admin_ai.set_metadata_reset(max_client, 100, "reset")
    async with metadata_db() as session:
        config = await session.get(AIConfig, 1)
        assert config.metadata_reset_mode == "reset"


@pytest.mark.asyncio
async def test_preserve_mode_carries_only_scoped_metadata_and_clears_runtime_state(metadata_db):
    async with metadata_db() as session:
        user = User(
            id=701,
            current_dialogue_id=1,
            current_topic_id=7,
            first_name="Profile",
            metadata_json=json.dumps({"account": "keep", "data_history": ["old"]}),
        )
        session.add(user)
        session.add_all([
            Message(user_id=701, dialogue_id=1, topic_id=7, role="user", content="old history"),
            AutomationConversationState(
                user_id=701,
                dialogue_id=1,
                topic_id=7,
                current_step="old_step",
                current_state_json=json.dumps({"temporary": True}),
                metadata_json=json.dumps({"learned": "keep"}),
            ),
            AutomationMetadataRecord(
                user_id=701,
                dialogue_id=1,
                topic_id=7,
                data_json=json.dumps({"learned": "keep"}),
            ),
            AutomationEvent(
                user_id=701,
                dialogue_id=1,
                topic_id=7,
                name="old_event",
                metadata_json=json.dumps({"learned": "keep"}),
            ),
            AutomationStepTransition(
                user_id=701,
                dialogue_id=1,
                topic_id=7,
                current_step="old_step",
            ),
            CardSpreadState(user_id=701, state_json=json.dumps({"pending": [1]})),
            TestSession(user_id=701, current_question_index=2),
        ])
        await session.commit()

        await start_new_dialogue_scope(
            session,
            user,
            7,
            MEMORY_MODE_TOPIC,
            "preserve",
        )
        await session.commit()

    async with metadata_db() as session:
        current_user = await session.get(User, 701)
        assert current_user.current_dialogue_id == 2
        assert current_user.first_name == "Profile"
        assert json.loads(current_user.metadata_json) == {"account": "keep", "data_history": ["old"]}
        assert not (await session.execute(select(Message).where(Message.user_id == 701, Message.dialogue_id == 2))).scalars().all()
        new_state = await session.scalar(
            select(AutomationConversationState).where(
                AutomationConversationState.user_id == 701,
                AutomationConversationState.dialogue_id == 2,
                AutomationConversationState.topic_id == 7,
            )
        )
        assert json.loads(new_state.metadata_json) == {"learned": "keep"}
        assert new_state.current_step is None
        assert json.loads(new_state.current_state_json) == {}
        assert await session.get(CardSpreadState, 701) is None
        assert await session.get(TestSession, 701) is None
        assert await session.scalar(select(AutomationMetadataRecord.id).where(AutomationMetadataRecord.dialogue_id == 1)) is not None
        assert await session.scalar(select(AutomationEvent.id).where(AutomationEvent.dialogue_id == 1)) is not None
        assert await session.scalar(select(AutomationStepTransition.id).where(AutomationStepTransition.dialogue_id == 1)) is not None
        context = await build_runtime_automation_context(
            session,
            user_id=701,
            dialogue_id=2,
            topic_id=7,
            memory_mode=MEMORY_MODE_TOPIC,
        )
        assert '"learned":"keep"' in context
        assert "temporary" not in context


@pytest.mark.asyncio
async def test_reset_mode_starts_empty_global_scope_without_deleting_profile_metadata(metadata_db):
    async with metadata_db() as session:
        user = User(
            id=702,
            current_dialogue_id=4,
            current_topic_id=None,
            metadata_json=json.dumps({"account": "keep"}),
        )
        session.add(user)
        session.add(AutomationDialogueState(
            user_id=702,
            dialogue_id=4,
            metadata_json=json.dumps({"learned": "drop"}),
        ))
        await session.commit()

        await start_new_dialogue_scope(
            session,
            user,
            0,
            MEMORY_MODE_GLOBAL,
            "reset",
        )
        await session.commit()

    async with metadata_db() as session:
        current_user = await session.get(User, 702)
        assert current_user.current_dialogue_id == 5
        assert json.loads(current_user.metadata_json) == {"account": "keep"}
        new_state = await session.scalar(
            select(AutomationDialogueState).where(
                AutomationDialogueState.user_id == 702,
                AutomationDialogueState.dialogue_id == 5,
            )
        )
        assert json.loads(new_state.metadata_json) == {}
        context = await build_runtime_automation_context(
            session,
            user_id=702,
            dialogue_id=5,
            topic_id=None,
            memory_mode=MEMORY_MODE_GLOBAL,
        )
        assert '"metadata":{}' in context
