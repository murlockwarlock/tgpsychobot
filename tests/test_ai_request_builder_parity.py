import copy
import json
import os
import unittest
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_request_builder import (
    build_client_runtime_context,
    build_conversational_request_layout,
    build_isolated_request_layout,
    build_temporal_activity_context,
    load_conversational_ai_history,
)
from ai_request_context import (
    AIRequestLayout,
    AIRequestMessage,
    build_anthropic_system,
    build_gemini_contents,
    build_gemini_system_parts,
    build_openai_chat_messages,
    sanitize_ai_request_capture,
)
from automation_engine import apply_service_data_blocks
from database import (
    AIConfig,
    AILog,
    AutomationConversationState,
    AutomationDialogueState,
    Base,
    KnowledgeBase,
    Message as DBMessage,
    Topic,
    User,
)
from user_metadata import extract_service_data


class AIRequestBuilderParityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with self.sessions() as session:
            self.user = User(
                id=1001,
                first_name="Иван",
                current_dialogue_id=1,
                current_topic_id=5,
                metadata_json="{}",
            )
            self.topic = Topic(id=5, name="Отношения", is_active=True, system_prompt="Ты эксперт по отношениям.")
            self.other_topic = Topic(id=6, name="Карьера", is_active=True, system_prompt="Ты карьерный коуч.")
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test",
                openai_model="gpt-5.6-terra",
                system_prompt="Ты эмпатичный психолог.",
                shared_prompt_block="Правила: будь вежлив.",
                service_prompt_block="Служебные инструкции: не выходи из роли.",
                memory_mode="global",
                context_limit_first=2,
                context_limit_recent=4,
            )
            session.add_all([self.user, self.topic, self.other_topic, self.ai_config])
            await session.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def _build(self, session, prompt="Привет", user=None, ai_config=None, **kwargs):
        u = user or await session.get(User, 1001)
        cfg = ai_config or await session.get(AIConfig, 1)
        dialogue_id = kwargs.pop("dialogue_id", u.current_dialogue_id)
        topic_id = kwargs.pop("topic_id", u.current_topic_id)
        return await build_conversational_request_layout(
            session,
            user=u,
            ai_config=cfg,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            current_user_content=prompt,
            **kwargs,
        )

    # 23. Service block TG/MAX: service_prompt_block is present in layout of both platforms
    async def test_service_block_tg_max(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Привет")
            layout_max = await self._build(session, prompt="Привет")
            self.assertIn("Служебные инструкции: не выходи из роли.", layout_tg.shared_instructions)
            self.assertIn("Служебные инструкции: не выходи из роли.", layout_max.shared_instructions)

    # 24. Service block exactly once: service block is not duplicated in layout
    async def test_service_block_exactly_once(self):
        async with self.sessions() as session:
            layout = await self._build(session, prompt="Привет")
            count = sum(
                1 for block in layout.shared_instructions
                if "Служебные инструкции: не выходи из роли." in block
            )
            self.assertEqual(count, 1)

    # 25. Shared block parity: shared_prompt_block rendered identically in TG and MAX
    async def test_shared_block_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Привет")
            layout_max = await self._build(session, prompt="Привет")
            self.assertEqual(layout_tg.shared_instructions, layout_max.shared_instructions)

    # 26. Client runtime parity: client data block rendered identically on both platforms
    async def test_client_runtime_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Вопрос")
            layout_max = await self._build(session, prompt="Вопрос")
            self.assertEqual(layout_tg.runtime_context, layout_max.runtime_context)

    # 27. Automation runtime parity: automation service block rendered identically in TG and MAX
    async def test_automation_runtime_parity(self):
        async with self.sessions() as session:
            user = await session.get(User, 1001)
            _, blocks, _ = extract_service_data('<DATA>{"metadata": {"foo": "bar"}}</DATA>')
            await apply_service_data_blocks(
                session, user=user, dialogue_id=1, topic_id=5, blocks=blocks, memory_mode="global"
            )
            await session.commit()

        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Вопрос")
            layout_max = await self._build(session, prompt="Вопрос")
            self.assertEqual(layout_tg.scenario_context, layout_max.scenario_context)
            self.assertTrue(any("foo" in s for s in layout_tg.scenario_context))

    # 28. Global natural history: dialogue messages in Global passed in natural chronological order
    async def test_global_natural_history(self):
        now = datetime.utcnow()
        async with self.sessions() as session:
            m1 = DBMessage(
                user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Вопрос в теме 5",
                timestamp=now - timedelta(minutes=10)
            )
            m2 = DBMessage(
                user_id=1001, dialogue_id=1, topic_id=5, role="assistant", content="Ответ в теме 5",
                timestamp=now - timedelta(minutes=9)
            )
            m3 = DBMessage(
                user_id=1001, dialogue_id=1, topic_id=6, role="user", content="Вопрос в теме 6",
                timestamp=now - timedelta(minutes=8)
            )
            m4 = DBMessage(
                user_id=1001, dialogue_id=1, topic_id=6, role="assistant", content="Ответ в теме 6",
                timestamp=now - timedelta(minutes=7)
            )
            session.add_all([m1, m2, m3, m4])
            await session.commit()

        async with self.sessions() as session:
            layout = await self._build(session, prompt="Новый вопрос")
            roles = [msg.role for msg in layout.history]
            contents = [msg.content for msg in layout.history]
            self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
            self.assertEqual(contents, ["Вопрос в теме 5", "Ответ в теме 5", "Вопрос в теме 6", "Ответ в теме 6"])

    # 29. No synthetic global memory: string "ГЛОБАЛЬНАЯ ПАМЯТЬ ИЗ ДРУГИХ ТЕМ" completely absent
    async def test_no_synthetic_global_memory(self):
        async with self.sessions() as session:
            layout = await self._build(session, prompt="Тест")
            for block in layout.ordered_instruction_blocks:
                self.assertNotIn("ГЛОБАЛЬНАЯ ПАМЯТЬ ИЗ ДРУГИХ ТЕМ", block)
            for msg in layout.history:
                self.assertNotIn("ГЛОБАЛЬНАЯ ПАМЯТЬ ИЗ ДРУГИХ ТЕМ", str(msg.content))

    # 30. No duplicated cross-topic history: cross-topic messages not duplicated in request_context
    async def test_no_duplicated_cross_topic_history(self):
        async with self.sessions() as session:
            layout = await self._build(session, prompt="Тест")
            self.assertEqual(len(layout.request_context), 0)

    # 31. Topic mode scope: in memory_mode="topic", history is strictly filtered by active_topic_id
    async def test_topic_mode_scope(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.memory_mode = "topic"
            now = datetime.utcnow()
            m1 = DBMessage(
                user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Тема 5", timestamp=now - timedelta(minutes=5)
            )
            m2 = DBMessage(
                user_id=1001, dialogue_id=1, topic_id=6, role="user", content="Тема 6", timestamp=now - timedelta(minutes=4)
            )
            session.add_all([m1, m2])
            await session.commit()

        async with self.sessions() as session:
            layout = await self._build(session, prompt="Вопрос", topic_id=5, memory_mode="topic")
            self.assertEqual(len(layout.history), 1)
            self.assertEqual(layout.history[0].content, "Тема 5")

    # 32. Reset mode scope: in memory_mode="reset", history is strictly filtered by current dialogue_id
    async def test_reset_mode_scope(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.memory_mode = "reset"
            now = datetime.utcnow()
            m_old = DBMessage(
                user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Старый диалог", timestamp=now - timedelta(minutes=5)
            )
            m_new = DBMessage(
                user_id=1001, dialogue_id=2, topic_id=5, role="user", content="Новый диалог", timestamp=now - timedelta(minutes=2)
            )
            session.add_all([m_old, m_new])
            await session.commit()

        async with self.sessions() as session:
            layout = await self._build(session, prompt="Вопрос", dialogue_id=2, memory_mode="reset")
            self.assertEqual(len(layout.history), 1)
            self.assertEqual(layout.history[0].content, "Новый диалог")

    # 33. Request context parity: test context and knowledge base context formed identically
    async def test_request_context_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Вопрос", knowledge_context="Справка по теме")
            layout_max = await self._build(session, prompt="Вопрос", knowledge_context="Справка по теме")
            self.assertEqual(layout_tg.request_context, layout_max.request_context)
            self.assertTrue(any("Справка по теме" in r for r in layout_tg.request_context))

    # 34. Current content parity: user_prompt placed in current_user_content identically
    async def test_current_content_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Что делать?")
            layout_max = await self._build(session, prompt="Что делать?")
            self.assertEqual(layout_tg.current_user_content, layout_max.current_user_content)
            self.assertEqual(layout_tg.current_user_content, "Что делать?")

    # 35. Vision parity: photo request contains identical instructions and metadata in TG and MAX
    async def test_vision_parity(self):
        photo_content = [
            {"type": "text", "text": "Опиши рисунок"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,123"}},
        ]
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt=photo_content)
            layout_max = await self._build(session, prompt=photo_content)
            self.assertEqual(layout_tg.current_user_content, layout_max.current_user_content)
            self.assertEqual(layout_tg.ordered_instruction_blocks, layout_max.ordered_instruction_blocks)

    # 36. Current message identity exclusion before history clipping
    async def test_current_message_identity_exclusion_before_history_clipping(self):
        now = datetime.utcnow()
        async with self.sessions() as session:
            m1 = DBMessage(
                id=501, user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Вопрос 1", timestamp=now - timedelta(minutes=5)
            )
            m2 = DBMessage(
                id=502, user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Вопрос 2 (текущий)", timestamp=now
            )
            session.add_all([m1, m2])
            await session.commit()

        async with self.sessions() as session:
            layout = await self._build(session, prompt="Вопрос 2 (текущий)", exclude_message_id=502)
            msg_contents = [m.content for m in layout.history]
            self.assertNotIn("Вопрос 2 (текущий)", msg_contents)
            self.assertIn("Вопрос 1", msg_contents)

    # 37. Current message does not consume recent slot
    async def test_current_message_does_not_consume_recent_slot(self):
        now = datetime.utcnow()
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.context_limit_first = 0
            cfg.context_limit_recent = 2
            m1 = DBMessage(id=601, user_id=1001, dialogue_id=1, topic_id=5, role="user", content="История 1", timestamp=now - timedelta(minutes=10))
            m2 = DBMessage(id=602, user_id=1001, dialogue_id=1, topic_id=5, role="user", content="История 2", timestamp=now - timedelta(minutes=5))
            m3 = DBMessage(id=603, user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Текущее", timestamp=now)
            session.add_all([m1, m2, m3])
            await session.commit()

        async with self.sessions() as session:
            layout = await self._build(session, prompt="Текущее", exclude_message_id=603, limit_first=0, limit_recent=2)
            msg_contents = [m.content for m in layout.history]
            self.assertIn("История 1", msg_contents)
            self.assertIn("История 2", msg_contents)
            self.assertEqual(len(layout.history), 2)

    # 38. Identical-content user messages preserved: consecutive identical messages not dropped
    async def test_identical_content_user_messages_preserved(self):
        now = datetime.utcnow()
        async with self.sessions() as session:
            m1 = DBMessage(id=701, user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Да", timestamp=now - timedelta(minutes=5))
            m2 = DBMessage(id=702, user_id=1001, dialogue_id=1, topic_id=5, role="assistant", content="Хорошо", timestamp=now - timedelta(minutes=4))
            m3 = DBMessage(id=703, user_id=1001, dialogue_id=1, topic_id=5, role="user", content="Да", timestamp=now - timedelta(minutes=3))
            m4 = DBMessage(id=704, user_id=1001, dialogue_id=1, topic_id=5, role="assistant", content="Понял", timestamp=now - timedelta(minutes=2))
            session.add_all([m1, m2, m3, m4])
            await session.commit()

        async with self.sessions() as session:
            layout = await self._build(session, prompt="И что дальше?")
            user_contents = [m.content for m in layout.history if m.role == "user"]
            self.assertEqual(user_contents, ["Да", "Да"])

    # 39. Shared isolated direct builder layout parity
    async def test_shared_isolated_direct_builder_layout_parity(self):
        async with self.sessions() as session:
            user = await session.get(User, 1001)
            cfg = await session.get(AIConfig, 1)
            layout_tg = await build_isolated_request_layout(
                session,
                user=user,
                ai_config=cfg,
                system_prompt="Ты эксперт",
                user_prompt="Сделай расчет",
                dialogue_id=1,
                topic_id=5,
            )
            layout_max = await build_isolated_request_layout(
                session,
                user=user,
                ai_config=cfg,
                system_prompt="Ты эксперт",
                user_prompt="Сделай расчет",
                dialogue_id=1,
                topic_id=5,
            )
            self.assertEqual(layout_tg.ordered_instruction_blocks, layout_max.ordered_instruction_blocks)
            self.assertEqual(layout_tg.current_user_content, layout_max.current_user_content)
            self.assertEqual(layout_tg.history, layout_max.history)

    # 40. Direct isolated request does not load dialogue history
    async def test_direct_isolated_request_does_not_load_dialogue_history(self):
        async with self.sessions() as session:
            user = await session.get(User, 1001)
            cfg = await session.get(AIConfig, 1)
            layout = await build_isolated_request_layout(
                session,
                user=user,
                ai_config=cfg,
                system_prompt="Эксперт",
                user_prompt="Тестовый расчет",
                dialogue_id=1,
                topic_id=5,
            )
            self.assertEqual(len(layout.history), 0)
            self.assertEqual(layout.history, ())

    # 41. DeepSeek wire parity: outbound wire payload equivalent between TG and MAX adhering to 8 invariants
    async def test_deepseek_wire_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Тест")
            layout_max = await self._build(session, prompt="Тест")
            wire_tg = build_openai_chat_messages(layout_tg)
            wire_max = build_openai_chat_messages(layout_max)
            self.assertEqual(wire_tg, wire_max)

    # 42. Gemini wire parity: outbound wire payload equivalent between TG and MAX
    async def test_gemini_wire_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Тест")
            layout_max = await self._build(session, prompt="Тест")
            sys_tg = build_gemini_system_parts(layout_tg)
            sys_max = build_gemini_system_parts(layout_max)
            contents_tg = build_gemini_contents(layout_tg)
            contents_max = build_gemini_contents(layout_max)
            self.assertEqual(sys_tg, sys_max)
            self.assertEqual(contents_tg, contents_max)

    # 43. KIE wire parity: outbound wire payload for KIE equivalent between TG and MAX
    async def test_kie_wire_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Тест KIE")
            layout_max = await self._build(session, prompt="Тест KIE")
            wire_gemini_tg = build_gemini_contents(layout_tg)
            wire_gemini_max = build_gemini_contents(layout_max)
            self.assertEqual(wire_gemini_tg, wire_gemini_max)

    # 44. Claude wire parity: outbound wire payload (system + messages) equivalent between TG and MAX
    async def test_claude_wire_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Тест Claude")
            layout_max = await self._build(session, prompt="Тест Claude")
            sys_tg = build_anthropic_system(layout_tg)
            sys_max = build_anthropic_system(layout_max)
            msgs_tg = [
                {"role": m.role, "content": m.content} for m in layout_tg.history
            ] + [{"role": "user", "content": layout_tg.current_user_content}]
            msgs_max = [
                {"role": m.role, "content": m.content} for m in layout_max.history
            ] + [{"role": "user", "content": layout_max.current_user_content}]
            self.assertEqual(sys_tg, sys_max)
            self.assertEqual(msgs_tg, msgs_max)

    # 45. OpenAI wire parity: outbound wire payload for OpenAI equivalent between TG and MAX
    async def test_openai_wire_parity(self):
        async with self.sessions() as session:
            layout_tg = await self._build(session, prompt="Тест OpenAI")
            layout_max = await self._build(session, prompt="Тест OpenAI")
            wire_tg = build_openai_chat_messages(layout_tg)
            wire_max = build_openai_chat_messages(layout_max)
            self.assertEqual(wire_tg, wire_max)

    # 46. AILog request_payload capture matches wire outbound without secrets
    async def test_ailog_request_payload_capture_matches_wire_outbound(self):
        async with self.sessions() as session:
            layout = await self._build(session, prompt="Вопрос без секретов")
            wire = build_openai_chat_messages(layout)
            raw_payload = {
                "model": "gpt-5.6-terra",
                "messages": wire,
                "api_key": "sk-secret-12345",
            }
            capture = sanitize_ai_request_capture(
                provider="OpenAI",
                endpoint="https://api.openai.com/v1/chat/completions?key=sensitive_key",
                payload=raw_payload,
            )
            # Secrets must be sanitized
            self.assertEqual(capture["payload"]["api_key"], "[REDACTED]")
            self.assertNotIn("sensitive_key", capture["endpoint"])
            # Wire messages structure must match
            self.assertEqual(capture["payload"]["messages"], wire)

    # 47. Image exclusion regression: message starting with "[Изображение]" retained when exclude_message_id=None
    async def test_image_exclusion_retained_when_exclude_id_is_none(self):
        async with self.sessions() as session:
            img_msg = DBMessage(
                id=901,
                user_id=1001,
                dialogue_id=1,
                topic_id=5,
                role="user",
                content="[Изображение] Историческое фото",
                timestamp=datetime(2026, 1, 1, 10, 0, 0),
            )
            session.add(img_msg)
            await session.commit()

            history = await load_conversational_ai_history(
                session,
                user_id=1001,
                dialogue_id=1,
                topic_id=5,
                memory_mode="global",
                exclude_message_id=None,
            )
            self.assertTrue(any(m.content == "[Изображение] Историческое фото" for m in history))

    # 48. Image exclusion regression: only exact row X excluded when exclude_message_id=X
    async def test_image_exclusion_filters_only_exact_id(self):
        async with self.sessions() as session:
            img_msg1 = DBMessage(
                id=902,
                user_id=1001,
                dialogue_id=1,
                topic_id=5,
                role="user",
                content="[Изображение] Первое фото",
                timestamp=datetime(2026, 1, 1, 10, 0, 0),
            )
            img_msg2 = DBMessage(
                id=903,
                user_id=1001,
                dialogue_id=1,
                topic_id=5,
                role="user",
                content="[Изображение] Второе фото",
                timestamp=datetime(2026, 1, 1, 10, 0, 1),
            )
            session.add_all([img_msg1, img_msg2])
            await session.commit()

            history = await load_conversational_ai_history(
                session,
                user_id=1001,
                dialogue_id=1,
                topic_id=5,
                memory_mode="global",
                exclude_message_id=903,
            )
            # 902 is retained, 903 is excluded
            contents = [m.content for m in history]
            self.assertIn("[Изображение] Первое фото", contents)
            self.assertNotIn("[Изображение] Второе фото", contents)

    # 49. Builder ownership regression: production callers do not pass preassembled shared_instructions or scenario_context
    async def test_builder_ownership_invokes_renderers_once_without_precomputed_scenarios(self):
        from unittest.mock import patch
        import prompt_blocks
        import automation_engine

        with patch("ai_request_builder.render_service_prompt", wraps=prompt_blocks.render_service_prompt) as mock_render_service, \
             patch("automation_engine.build_runtime_automation_context", wraps=automation_engine.build_runtime_automation_context) as mock_build_scenario:

            async with self.sessions() as session:
                user = await session.get(User, 1001)
                cfg = await session.get(AIConfig, 1)

                layout = await build_conversational_request_layout(
                    session,
                    user=user,
                    ai_config=cfg,
                    dialogue_id=1,
                    topic_id=5,
                    current_user_content="Тестовый вопрос",
                )

                self.assertEqual(mock_render_service.call_count, 1)
                self.assertEqual(mock_build_scenario.call_count, 1)
                self.assertIsNotNone(layout.scenario_context)
                self.assertTrue(len(layout.shared_instructions) > 0)
