import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from automation_engine import apply_service_data_blocks, build_runtime_automation_context, get_conversation_automation_state
from database import (
    AIConfig,
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
            AIConfig.__table__,
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
            session.add(AIConfig(id=1, provider="Gemini", gemini_api_key="test", gemini_model="test"))
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

    async def test_runtime_context_exposes_state_without_navigation_command(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=7,
                blocks=self._blocks(
                    '{"current_state":{"current_step":"STEP_2"},"metadata":{"guide":"yoda"}}'
                ),
            )
            await session.commit()
            context = await build_runtime_automation_context(
                session,
                user_id=42,
                dialogue_id=1,
                topic_id=7,
            )

        self.assertIn('"current_step":"STEP_2"', context)
        self.assertIn('"guide":"yoda"', context)
        self.assertNotIn("Продолжай алгоритм", context)

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

    async def test_test_result_ai_path_applies_atomic_data_block(self):
        with patch.dict(
            sys.modules,
            {
                "yookassa": SimpleNamespace(Configuration=object, Payment=object),
                "yookassa.domain": SimpleNamespace(),
                "yookassa.domain.exceptions": SimpleNamespace(
                    BadRequestError=Exception,
                    ForbiddenError=Exception,
                    InternalServerError=Exception,
                    TooManyRequestsError=Exception,
                    UnauthorizedError=Exception,
                ),
                "dateutil": SimpleNamespace(),
                "dateutil.relativedelta": SimpleNamespace(relativedelta=object),
            },
        ):
            import handlers

        provider_response = (
            "Результат теста готов.\n"
            '<DATA>{"current_state":{"current_step":"TEST_RESULT_READY"},'
            '"events":["TEST_COMPLETED"],"save_mode":"merge",'
            '"metadata":{"test":{"score":17}}}</DATA>'
        )
        call = AsyncMock(return_value=provider_response)

        with (
            patch.object(handlers, "async_session_maker", self.sessions),
            patch.object(handlers, "_call_gemini_api", call),
        ):
            visible = await handlers.get_ai_response_direct(
                42,
                "Сформируй результат психологического теста.",
                "Ответы пользователя.",
            )

        self.assertEqual(visible, "Результат теста готов.")
        sent_system_prompt = call.await_args.args[4]
        self.assertNotIn("<DATA>", sent_system_prompt)
        async with self.sessions() as session:
            state = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=None
            )
            event_count = await session.scalar(select(func.count(AutomationEvent.id)))

        self.assertEqual(state.current_step, "TEST_RESULT_READY")
        self.assertEqual(load_metadata(state.metadata_json), {"test": {"score": 17}})
        self.assertEqual(event_count, 1)

    async def test_get_ai_response_direct_passes_scoped_dialogue_and_topic(self):
        import handlers

        provider_response = (
            "Промежуточный результат.\n"
            '<DATA>{"current_state":{"current_step":"CALCULATED"},'
            '"metadata":{"calc":{"result_val":42}}}</DATA>'
        )
        call = AsyncMock(return_value=provider_response)

        with (
            patch.object(handlers, "async_session_maker", self.sessions),
            patch.object(handlers, "_call_gemini_api", call),
        ):
            visible = await handlers.get_ai_response_direct(
                42,
                "Системный промпт",
                "Ответы",
                dialogue_id=3,
                topic_id=15,
            )

        self.assertEqual(visible, "Промежуточный результат.")
        async with self.sessions() as session:
            state = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=3, topic_id=15
            )
        self.assertIsNotNone(state)
        self.assertEqual(state.current_step, "CALCULATED")
        self.assertEqual(load_metadata(state.metadata_json), {"calc": {"result_val": 42}})

    async def test_max_get_ai_response_direct_merges_data_blocks(self):
        from max_messenger_bot import ai as max_ai

        provider_response = (
            "MAX результат.\n"
            '<DATA>{"current_state":{"current_step":"MAX_STEP"},'
            '"metadata":{"max_key":"max_val"}}</DATA>'
        )

        with (
            patch.object(max_ai, "async_session_maker", self.sessions),
            patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=provider_response)),
        ):
            visible = await max_ai.get_ai_response_direct(
                42,
                "Системный промпт",
                "Ответы",
                dialogue_id=2,
                topic_id=8,
            )

        self.assertEqual(visible, "MAX результат.")
        async with self.sessions() as session:
            state = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=2, topic_id=8
            )
        self.assertIsNotNone(state)
        self.assertEqual(state.current_step, "MAX_STEP")
        self.assertEqual(load_metadata(state.metadata_json), {"max_key": "max_val"})

