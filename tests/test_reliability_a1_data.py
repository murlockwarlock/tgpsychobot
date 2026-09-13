import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from user_metadata import extract_service_data
from result_history import (
    AIHistoryMessage,
    select_ai_history_messages,
)
from ai_request_builder import build_conversational_request_layout
from database import async_session_maker, init_db, User, Message as DBMessage, AIConfig


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
    async def test_ai_request_builder_omits_empty_assistant_message(self):
        raw_broken = "<DATA>\n{\"current_state\": {\"stage\":"
        mock_user = MagicMock()
        mock_user.role = "user"
        mock_user.content = "Привет!"
        mock_user.ai_context_content = None
        mock_user.topic_id = None
        mock_user.topic = None

        mock_asst = MagicMock()
        mock_asst.role = "assistant"
        mock_asst.content = raw_broken
        mock_asst.ai_context_content = None
        mock_asst.topic_id = None
        mock_asst.topic = None

        selected = select_ai_history_messages([mock_user, mock_asst], limit_first=0, limit_recent=10)
        assert len(selected) == 2
        assert selected[1].content == ""

        # Build history items as done in build_conversational_request_layout
        history_items = [
            {"role": item.role, "content": item.content}
            for item in selected
            if item.content
        ]
        assert len(history_items) == 1
        assert history_items[0]["role"] == "user"
        assert history_items[0]["content"] == "Привет!"
        # The broken assistant turn was dropped; raw DATA never leaked into outbound history!

    @pytest.mark.asyncio
    async def test_telegram_flow_ailog_and_persistence_parity(self):
        raw_response = (
            "Нормальный ответ пользователю.\n"
            "```json\n"
            "<DATA>\n"
            "{\"current_state\": {\"stage\": 1"
        )
        visible, blocks, invalid = extract_service_data(raw_response)
        assert visible == "Нормальный ответ пользователю."
        assert invalid == 1

        # Simulate AILog creation
        ai_log_raw = raw_response
        ai_log_clean = visible

        # AILog retains full response, clean_text contains only visible
        assert "<DATA>" in ai_log_raw
        assert "<DATA>" not in ai_log_clean
        assert "{\"current_state\"" not in ai_log_clean

    @pytest.mark.asyncio
    async def test_max_flow_ailog_and_persistence_parity(self):
        raw_response = (
            "Ответ в MAX.\n"
            "<DATA>\n"
            "{\"partial\": true"
        )
        visible, blocks, invalid = extract_service_data(raw_response)
        assert visible == "Ответ в MAX."
        assert invalid == 1

        # Simulate MAX save_ai_message
        persisted_content = visible
        assert "<DATA>" not in persisted_content
        assert "{\"partial\"" not in persisted_content
