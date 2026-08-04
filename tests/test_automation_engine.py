import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from automation_engine import apply_service_data_blocks, get_conversation_automation_state
from database import (
    AutomationConversationState,
    AutomationEvent,
    AutomationMetadataRecord,
    AutomationStepTransition,
    Base,
    User,
)
from user_metadata import extract_service_data, load_metadata


class AutomationEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        tables = [
            User.__table__,
            AutomationConversationState.__table__,
            AutomationStepTransition.__table__,
            AutomationMetadataRecord.__table__,
            AutomationEvent.__table__,
        ]
        async with self.engine.begin() as connection:
            await connection.run_sync(lambda conn: Base.metadata.create_all(conn, tables=tables))
        async with self.sessions() as session:
            session.add(User(id=42, first_name="Иван", current_dialogue_id=1, metadata_json="{}"))
            await session.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    @staticmethod
    def _blocks(payload: str):
        _, blocks, invalid = extract_service_data(f"<DATA>{payload}</DATA>")
        assert invalid == 0
        return blocks

    async def test_step_transition_is_written_only_when_step_changes(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            first = await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=7,
                blocks=self._blocks(
                    '{"current_state":{"current_step":"STEP_1"},'
                    '"events":["ANSWER_RECEIVED"],"metadata":{"profile":{"name":"Иван"}}}'
                ),
            )
            await session.commit()
            second = await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=7,
                blocks=self._blocks(
                    '{"current_state":{"current_step":"STEP_1","score":2},'
                    '"metadata":{"profile":{"age":30}}}'
                ),
            )
            await session.commit()

            transition_count = await session.scalar(select(func.count(AutomationStepTransition.id)))
            event_count = await session.scalar(select(func.count(AutomationEvent.id)))
            state = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=7
            )

        self.assertTrue(first.state_changed)
        self.assertFalse(second.state_changed)
        self.assertEqual(transition_count, 1)
        self.assertEqual(event_count, 1)
        self.assertEqual(load_metadata(state.metadata_json), {
            "profile": {"name": "Иван", "age": 30},
        })
        self.assertEqual(load_metadata(state.current_state_json), {
            "current_step": "STEP_1",
            "score": 2,
        })

    async def test_topic_and_dialogue_have_independent_state(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            for dialogue_id, topic_id, step in (
                (1, None, "MAIN_1"),
                (1, 8, "TOPIC_8"),
                (2, None, "MAIN_2"),
            ):
                await apply_service_data_blocks(
                    session,
                    user=user,
                    dialogue_id=dialogue_id,
                    topic_id=topic_id,
                    blocks=self._blocks(f'{{"current_state":{{"current_step":"{step}"}}}}'),
                )
            await session.commit()
            rows = (await session.execute(select(AutomationConversationState))).scalars().all()

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {(row.dialogue_id, row.topic_id, row.current_step) for row in rows},
            {(1, 0, "MAIN_1"), (1, 8, "TOPIC_8"), (2, 0, "MAIN_2")},
        )

    async def test_snapshot_is_a_record_but_does_not_mutate_merged_context(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=None,
                blocks=self._blocks('{"metadata":{"stable":1},"save_mode":"merge"}'),
            )
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=None,
                blocks=self._blocks('{"metadata":{"snapshot":2},"save_mode":"snapshot"}'),
            )
            await session.commit()
            state = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=None
            )
            records = (
                await session.execute(select(AutomationMetadataRecord).order_by(AutomationMetadataRecord.id))
            ).scalars().all()

        self.assertEqual(load_metadata(state.metadata_json), {"stable": 1})
        self.assertEqual([record.save_mode for record in records], ["merge", "snapshot"])
