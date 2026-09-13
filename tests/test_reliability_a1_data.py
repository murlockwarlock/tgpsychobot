import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import ai_integration
import handlers
from user_metadata import extract_service_data
from result_history import (
    AIHistoryMessage,
    select_ai_history_messages,
)
from ai_request_builder import build_conversational_request_layout
from database import Base, async_session_maker, init_db, User, Message as DBMessage, AIConfig, AILog, BotGeneralConfig


class TestA1DataSanitization:
    """Covers all 22 required DATA test cases from Section K."""

    def test_01_complete_valid_xml_data(self):
        text = "Здравствуйте!\n<DATA>\n{\"current_state\": {\"stage\": 1}}\n</DATA>"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Здравствуйте!"
        assert len(blocks) == 1
        assert blocks[0].current_state == {"stage": 1}
        assert invalid == 0

    def test_02_complete_valid_square_data(self):
        text = "Здравствуйте!\n[DATA]\n{\"key\": \"value\"}\n[/DATA]"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Здравствуйте!"
        assert len(blocks) == 1
        assert blocks[0].metadata == {"key": "value"}
        assert blocks[0].legacy is True
        assert invalid == 0

    def test_03_complete_invalid_xml_data(self):
        text = "Здравствуйте!\n<DATA>\nnot a valid json\n</DATA>"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Здравствуйте!"
        assert len(blocks) == 0
        assert invalid == 1

    def test_04_complete_invalid_square_data(self):
        text = "Здравствуйте!\n[DATA]\nnot a valid json\n[/DATA]"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Здравствуйте!"
        assert len(blocks) == 0
        assert invalid == 1

    def test_05_visible_text_plus_incomplete_xml_data(self):
        text = "Нормальный ответ пользователю.\n<DATA>\n{\"metadata\":{\"x\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Нормальный ответ пользователю."
        assert len(blocks) == 0
        assert invalid == 1

    def test_06_visible_text_plus_incomplete_xml_data_with_version(self):
        text = "Нормальный ответ пользователю.\n<DATA version=\"1\">\n{\"metadata\":{\"x\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Нормальный ответ пользователю."
        assert len(blocks) == 0
        assert invalid == 1

    def test_07_xml_opener_truncated_before_closing_bracket(self):
        text = "Нормальный ответ пользователю.\n<DATA version=\"1\"\n{\"metadata\":{\"x\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Нормальный ответ пользователю."
        assert len(blocks) == 0
        assert invalid == 1

    def test_08_incomplete_square_data(self):
        text = "Нормальный ответ пользователю.\n[DATA]\n{\"metadata\":{\"x\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Нормальный ответ пользователю."
        assert len(blocks) == 0
        assert invalid == 1

    def test_09_case_insensitive_tags(self):
        for tag in ["<data>", "<DaTa>", "[data]"]:
            text = f"Ответ пользователю.\n{tag}\n{{\"metadata\":{{\"x\":"
            visible, blocks, invalid = extract_service_data(text)
            assert visible == "Ответ пользователю.", f"Failed on {tag}"
            assert len(blocks) == 0
            assert invalid == 1

    def test_10_multiline_partial_json(self):
        text = "Ответ.\n<DATA>\n{\n  \"state\": 1,\n  \"user\": {\n    \"name\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Ответ."
        assert len(blocks) == 0
        assert invalid == 1

    def test_11_incomplete_quoted_json_string(self):
        text = "Ответ.\n<DATA>\n{\"unclosed\": \"text that ends abruptly"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Ответ."
        assert len(blocks) == 0
        assert invalid == 1

    def test_12_incomplete_nested_object_or_list(self):
        text = "Ответ.\n<DATA>\n{\"items\": [{\"id\": 1}, {\"id\": 2, \"sub\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Ответ."
        assert len(blocks) == 0
        assert invalid == 1

    def test_13_opening_json_fence_plus_incomplete_data(self):
        text = "Нормальный ответ.\n```json\n<DATA>\n{\"metadata\":{\"x\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Нормальный ответ."
        assert len(blocks) == 0
        assert invalid == 1

    def test_14_opening_plain_fence_plus_incomplete_data(self):
        text = "Нормальный ответ.\n```\n<DATA>\n{\"metadata\":{\"x\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Нормальный ответ."
        assert len(blocks) == 0
        assert invalid == 1

    def test_15_unrelated_earlier_valid_code_block_remains_intact(self):
        text = (
            "Вот пример функции:\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n"
            "Объяснение готово.\n"
            "```json\n"
            "<DATA>\n"
            "{\"metadata\":{\"x\":"
        )
        visible, blocks, invalid = extract_service_data(text)
        assert "def add(a, b):" in visible
        assert "```python" in visible
        assert visible.endswith("Объяснение готово.")
        assert "<DATA>" not in visible
        assert "{\"metadata\"" not in visible
        assert len(blocks) == 0
        assert invalid == 1

    def test_16_data_only_incomplete_response_yields_empty_clean_text(self):
        text = "<DATA>\n{\"metadata\":{\"x\":"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == ""
        assert len(blocks) == 0
        assert invalid == 1

    def test_17_data_only_complete_response_yields_empty_clean_text(self):
        text = "<DATA>\n{\"current_state\": {\"stage\": 2}}\n</DATA>"
        visible, blocks, invalid = extract_service_data(text)
        assert visible == ""
        assert len(blocks) == 1
        assert invalid == 0

    def test_18_complete_data_followed_by_incomplete_data(self):
        text = (
            "Первая часть ответа.\n"
            "<DATA>\n"
            "{\"current_state\": {\"stage\": 1}}\n"
            "</DATA>\n"
            "Вторая часть ответа.\n"
            "<DATA>\n"
            "{\"metadata\": {\"partial\":"
        )
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Первая часть ответа.\n\nВторая часть ответа."
        assert len(blocks) == 1
        assert blocks[0].current_state == {"stage": 1}
        assert invalid == 1

    def test_19_ordinary_word_data_remains_unchanged(self):
        text = "Это важные DATA сведения для анализа."
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Это важные DATA сведения для анализа."
        assert len(blocks) == 0
        assert invalid == 0

    def test_20_database_tag_remains_unchanged(self):
        text = "Тег <DATABASE> не должен повреждаться."
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Тег <DATABASE> не должен повреждаться."
        assert len(blocks) == 0
        assert invalid == 0

    def test_21_datax_tag_remains_unchanged(self):
        text = "Тег <DATAX> также не должен повреждаться."
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Тег <DATAX> также не должен повреждаться."
        assert len(blocks) == 0
        assert invalid == 0

    def test_22_invalid_count_is_exact(self):
        text = (
            "Начало.\n"
            "<DATA>\n"
            "не json\n"
            "</DATA>\n"
            "Середина.\n"
            "<DATA>\n"
            "{\"обрыв\":"
        )
        visible, blocks, invalid = extract_service_data(text)
        assert visible == "Начало.\n\nСередина."
        assert len(blocks) == 0
        assert invalid == 2

    def test_23_self_closing_data_marker_preserves_text_around_it(self):
        cases = [
            "Ответ до.\n<DATA/>\nОтвет после.",
            "Ответ до.\n<DATA />\nОтвет после.",
            "Ответ до.\n<data/>\nОтвет после.",
            "Ответ до.\n<DaTa />\nОтвет после.",
            "Ответ до.\n<DATA version=\"1\" />\nОтвет после.",
        ]
        for c in cases:
            visible, blocks, invalid = extract_service_data(c)
            assert visible == "Ответ до.\n\nОтвет после.", f"Failed on case: {c}"
            assert len(blocks) == 0
            assert invalid == 1

    def test_24_self_closing_database_datax_remain_unchanged(self):
        t1 = "Ответ до.\n<DATABASE/>\nОтвет после."
        v1, b1, i1 = extract_service_data(t1)
        assert v1 == "Ответ до.\n<DATABASE/>\nОтвет после."
        assert len(b1) == 0
        assert i1 == 0

        t2 = "Ответ до.\n<DATAX/>\nОтвет после."
        v2, b2, i2 = extract_service_data(t2)
        assert v2 == "Ответ до.\n<DATAX/>\nОтвет после."
        assert len(b2) == 0
        assert i2 == 0

    def test_25_four_plus_backticks_markdown_fence_incomplete_data(self):
        # 4 backticks with json
        t1 = "Visible\n````json\n<DATA>\n{\"metadata\":{\"x\":"
        v1, _, i1 = extract_service_data(t1)
        assert v1 == "Visible"
        assert i1 == 1

        # 4 backticks plain
        t2 = "Visible\n````\n<DATA>\n{\"metadata\":{\"x\":"
        v2, _, i2 = extract_service_data(t2)
        assert v2 == "Visible"
        assert i2 == 1

        # 5 backticks
        t3 = "Visible\n`````json\n<DATA>\n{\"metadata\":{\"x\":"
        v3, _, i3 = extract_service_data(t3)
        assert v3 == "Visible"
        assert i3 == 1

    def test_26_four_plus_backticks_earlier_code_block_remains_intact(self):
        code_block = "````python\ndef foo():\n    return 42\n````"
        tail = "\n````json\n<DATA>\n{\"metadata\":{\"x\":"
        text = f"{code_block}{tail}"
        visible, _, invalid = extract_service_data(text)
        assert visible == code_block
        assert invalid == 1

    def test_27_incomplete_data_with_schema_url_attribute(self):
        t = "Visible\n<DATA schema=\"https://example.com\">\n{\"x\":"
        visible, blocks, invalid = extract_service_data(t)
        assert visible == "Visible"
        assert len(blocks) == 0
        assert invalid == 1

    def test_28_incomplete_data_with_path_attribute(self):
        t = "Visible\n<DATA path=\"/foo/bar\">\n{\"x\":"
        visible, blocks, invalid = extract_service_data(t)
        assert visible == "Visible"
        assert len(blocks) == 0
        assert invalid == 1

    def test_29_truncated_data_opener_with_schema_url(self):
        t = "Visible\n<DATA schema=\"https://example.com\""
        visible, blocks, invalid = extract_service_data(t)
        assert visible == "Visible"
        assert len(blocks) == 0
        assert invalid == 1

    def test_30_complete_valid_data_with_slash_containing_attribute(self):
        t = "Visible\n<DATA schema=\"https://example.com\">\n{\"metadata\":{\"key\":\"value\"}}\n</DATA>"
        visible, blocks, invalid = extract_service_data(t)
        assert visible == "Visible"
        assert len(blocks) == 1
        assert blocks[0].metadata == {"key": "value"}
        assert invalid == 0

    def test_31_self_closing_with_schema_url_attribute_preserves_surrounding_text(self):
        t = "Visible before\n<DATA schema=\"https://example.com\" />\nVisible after"
        visible, blocks, invalid = extract_service_data(t)
        assert visible == "Visible before\n\nVisible after"
        assert len(blocks) == 0
        assert invalid == 1


class TestA1ResultHistoryAndIntegration:
    """Covers Section L: raw/clean/history proof, result_history safety, and provider request parity."""

    def test_result_history_does_not_resurrect_raw_data_on_empty_clean_content(self):
        raw_broken = "<DATA>\n{\"current_state\": {\"stage\":"
        clean, _, _ = extract_service_data(raw_broken)
        assert clean == ""

        mock_msg = MagicMock()
        mock_msg.role = "assistant"
        mock_msg.content = raw_broken
        mock_msg.ai_context_content = None
        mock_msg.topic_id = None
        mock_msg.topic = None

        selected = select_ai_history_messages([mock_msg], limit_first=0, limit_recent=10)
        assert len(selected) == 1
        assert selected[0].role == "assistant"
        assert selected[0].content == ""  # NOT resurrected to raw_broken!

    @pytest.mark.asyncio
    async def test_telegram_truncated_data_flow_boundary_contract(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)

        orig_ai_session = ai_integration.async_session_maker
        orig_handlers_session = handlers.async_session_maker
        ai_integration.async_session_maker = sessions
        handlers.async_session_maker = sessions

        try:
            async with sessions() as s:
                user = User(
                    id=2001,
                    first_name="Мария",
                    gender="female",
                    current_dialogue_id=1,
                    current_topic_id=None,
                    metadata_json="{}",
                )
                ai_config = AIConfig(
                    id=1,
                    provider="OpenAI",
                    openai_api_key="sk-test-telegram-a1",
                    openai_model="gpt-5.6-terra",
                    system_prompt="Ты эмпатичный психолог.",
                )
                bot_gen = BotGeneralConfig(id=1)
                s.add(user)
                s.add(ai_config)
                s.add(bot_gen)
                await s.commit()

            broken_response = (
                "Нормальный ответ TG.\n"
                "<DATA>\n"
                '{"metadata":{"x":'
            )

            # 1. Invoke real ai_integration.generate_response() with outbound provider mocked
            response_capture = {}
            with patch.object(ai_integration, "_call_openai_api", AsyncMock(return_value=broken_response)):
                result = await ai_integration.generate_response(
                    user_id=2001,
                    user_prompt="Вопрос TG 1",
                    response_capture=response_capture,
                )

            assert result == "Нормальный ответ TG."

            # 2. Query actual AILog persisted by generate_response()
            async with sessions() as s:
                ai_logs = (
                    await s.execute(select(AILog).where(AILog.user_id == 2001).order_by(AILog.id.desc()))
                ).scalars().all()
                assert len(ai_logs) >= 1
                assert ai_logs[0].raw_response == broken_response
                assert ai_logs[0].clean_text == "Нормальный ответ TG."

            # 3. Drive actual Telegram assistant persistence path via handlers.process_buffered_messages
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=555))
            mock_bot.send_chat_action = AsyncMock()
            mock_bot.delete_message = AsyncMock()

            with patch.object(ai_integration, "_call_openai_api", AsyncMock(return_value=broken_response)):
                await handlers.process_buffered_messages(
                    user_id=2001,
                    bot=mock_bot,
                    isolated_prompt="Вопрос в диалоге TG",
                )

            # Query actual assistant DBMessage
            async with sessions() as s:
                messages = (
                    await s.execute(
                        select(DBMessage).where(DBMessage.user_id == 2001, DBMessage.role == "assistant").order_by(DBMessage.id.desc())
                    )
                ).scalars().all()
                assert len(messages) >= 1
                last_msg = messages[0]
                assert last_msg.content == "Нормальный ответ TG."
                assert "<DATA" not in last_msg.content
                assert '{"metadata"' not in last_msg.content

            # Inspect visible output sent to Telegram user
            sent_texts = [call.kwargs.get("text") or (call.args[1] if len(call.args) > 1 else "") for call in mock_bot.send_message.mock_calls]
            assert any("Нормальный ответ TG." in t for t in sent_texts if t)
            assert all("<DATA" not in t for t in sent_texts if t)
            assert all('{"metadata"' not in t for t in sent_texts if t)

            # 4. Execute/build next actual conversational request via build_conversational_request_layout
            async with sessions() as s:
                user = await s.get(User, 2001)
                ai_config = await s.get(AIConfig, 1)
                layout = await build_conversational_request_layout(
                    s,
                    user=user,
                    ai_config=ai_config,
                    dialogue_id=user.current_dialogue_id,
                    current_user_content="Второй вопрос TG",
                )

            assistant_history = [m.content for m in layout.history if getattr(m, "role", None) == "assistant"]
            assert assistant_history == ["Нормальный ответ TG."]
            assert assistant_history.count("Нормальный ответ TG.") == 1
            assert all("<DATA" not in c for c in assistant_history)
            assert all('{"metadata"' not in c for c in assistant_history)

        finally:
            ai_integration.async_session_maker = orig_ai_session
            handlers.async_session_maker = orig_handlers_session
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_telegram_data_only_truncated_variant_boundary_contract(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)

        orig_ai_session = ai_integration.async_session_maker
        orig_handlers_session = handlers.async_session_maker
        ai_integration.async_session_maker = sessions
        handlers.async_session_maker = sessions

        try:
            async with sessions() as s:
                user = User(
                    id=2001,
                    first_name="Мария",
                    gender="female",
                    current_dialogue_id=1,
                    current_topic_id=None,
                    metadata_json="{}",
                )
                ai_config = AIConfig(
                    id=1,
                    provider="OpenAI",
                    openai_api_key="sk-test-telegram-a1",
                    openai_model="gpt-5.6-terra",
                    system_prompt="Ты эмпатичный психолог.",
                )
                bot_gen = BotGeneralConfig(id=1)
                s.add(user)
                s.add(ai_config)
                s.add(bot_gen)
                await s.commit()

            broken_data_only = (
                "<DATA>\n"
                '{"metadata":{"x":'
            )

            # 1. Invoke real ai_integration.generate_response()
            with patch.object(ai_integration, "_call_openai_api", AsyncMock(return_value=broken_data_only)):
                result = await ai_integration.generate_response(
                    user_id=2001,
                    user_prompt="Вопрос TG DATA-only",
                )

            assert result == ""

            # 2. Query actual AILog
            async with sessions() as s:
                ai_logs = (
                    await s.execute(select(AILog).where(AILog.user_id == 2001).order_by(AILog.id.desc()))
                ).scalars().all()
                assert len(ai_logs) >= 1
                assert ai_logs[0].raw_response == broken_data_only
                assert ai_logs[0].clean_text == ""

            # 3. Drive Telegram conversational flow with broken data-only response
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=556))
            mock_bot.send_chat_action = AsyncMock()
            mock_bot.delete_message = AsyncMock()

            with patch.object(ai_integration, "_call_openai_api", AsyncMock(return_value=broken_data_only)):
                await handlers.process_buffered_messages(
                    user_id=2001,
                    bot=mock_bot,
                    isolated_prompt="Вопрос в диалоге TG DATA-only",
                )

            # 4. Build next actual conversational request layout using repository builder
            async with sessions() as s:
                user = await s.get(User, 2001)
                ai_config = await s.get(AIConfig, 1)
                layout = await build_conversational_request_layout(
                    s,
                    user=user,
                    ai_config=ai_config,
                    dialogue_id=user.current_dialogue_id,
                    current_user_content="Следующий вопрос TG",
                )

            # Prove: empty sanitized assistant content never becomes raw provider history
            for msg in layout.history:
                content = getattr(msg, "content", "")
                assert "<DATA" not in content
                assert '{"metadata"' not in content
                if getattr(msg, "role", None) == "assistant":
                    assert content != broken_data_only

        finally:
            ai_integration.async_session_maker = orig_ai_session
            handlers.async_session_maker = orig_handlers_session
            await engine.dispose()

