import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from max_messenger_bot.models import extract_sent_message_id
from max_messenger_bot.services import common
from max_messenger_bot.ai import AIServiceError


class MockMaxClient:
    def __init__(self, thinking_mid="mid.thinking.123"):
        self.events = []
        self.thinking_mid = thinking_mid

    async def send_message(self, chat_id: int, text: str, attachments=None):
        if "Думаю..." in text or "Анализирую" in text or "Распознаю" in text:
            mid = self.thinking_mid
            self.events.append(("send_message", text, mid))
            if mid:
                return {"message": {"body": {"mid": mid}}}
            else:
                return {}
        elif "Генерирую" in text:
            mid = "mid.gen.456"
            self.events.append(("send_message", text, mid))
            return {"message": {"body": {"mid": mid}}}
        elif "Редактирую" in text:
            mid = "mid.edit.789"
            self.events.append(("send_message", text, mid))
            return {"message": {"body": {"mid": mid}}}
        else:
            mid = f"mid.msg.{len(self.events)}"
            self.events.append(("send_message", text, mid))
            return {"message": {"body": {"mid": mid}}}

    async def edit_message(self, message_id: str | None, text: str, attachments=None):
        assert message_id is not None, "Forbidden: edit_message(None, ...) called!"
        self.events.append(("edit_message", text, message_id))
        return {"message": {"body": {"mid": message_id}}}

    async def delete_message(self, message_id: str | None):
        assert message_id is not None, "Forbidden: delete_message(None) called!"
        self.events.append(("delete_message", message_id))
        return {"ok": True}

    async def upload_file(self, file_type: str, file_path: str):
        self.events.append(("upload_file", file_type))
        return {"token": "file_token_abc"}

    async def send_media_attachment(self, chat_id: int, media_type: str, token: str):
        self.events.append(("send_media_attachment", media_type, token))
        return {"message": {"body": {"mid": "mid.media"}}}


class TestA1MaxSentMessageIdExtractor:
    """Covers cases 1-4 of Section M."""

    def test_01_canonical_message_body_mid(self):
        payload = {"message": {"body": {"mid": "mid.real.max.123"}}}
        assert extract_sent_message_id(payload) == "mid.real.max.123"

    def test_02_legacy_mock_message_mid(self):
        payload = {"message": {"mid": "mid.mock.456"}}
        assert extract_sent_message_id(payload) == "mid.mock.456"

    def test_03_alternative_body_mid(self):
        payload = {"body": {"mid": "mid.body.789"}}
        assert extract_sent_message_id(payload) == "mid.body.789"

    def test_04_root_mid(self):
        assert extract_sent_message_id({"mid": "mid.root.1"}) == "mid.root.1"
        assert extract_sent_message_id({"message_id": "mid.fallback.2"}) is None
        assert extract_sent_message_id({"id": "mid.fallback.3"}) is None
        assert extract_sent_message_id(None) is None
        assert extract_sent_message_id({}) is None
        assert extract_sent_message_id("not_a_dict") is None


@pytest.mark.asyncio
class TestA1MaxTemporaryStatusLifecycle:
    """Covers cases 5-19 of Section M."""

    async def test_05_chat_normal_success(self):
        client = MockMaxClient()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="Ответ ИИ пользователю.")), \
             patch.object(common, "save_ai_message", AsyncMock()):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Привет")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🤖 Думаю...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Ответ ИИ пользователю.", "mid.thinking.123")

    async def test_06_chat_ai_service_error(self):
        client = MockMaxClient()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(side_effect=AIServiceError("timeout"))):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Привет")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🤖 Думаю...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Сервис ИИ временно недоступен. Попробуйте позже.", "mid.thinking.123")

    async def test_07_chat_unexpected_exception(self):
        client = MockMaxClient()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(side_effect=RuntimeError("internal crash"))):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Привет")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🤖 Думаю...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Произошла внутренняя ошибка. Попробуйте позже.", "mid.thinking.123")

    async def test_08_chat_directive_only_gen_img(self):
        client = MockMaxClient()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="GEN_IMG: [котик на лужайке]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", AsyncMock(return_value=b"fake_png")):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Нарисуй котика")

        # 1. send thinking -> 2. edit thinking to generating progress -> 3. upload -> 4. send media -> 5. delete progress
        assert client.events[0] == ("send_message", "🤖 Думаю...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "🖼 Генерирую новое изображение...", "mid.thinking.123")
        assert client.events[2] == ("upload_file", "image")
        assert client.events[3] == ("send_media_attachment", "image", "file_token_abc")
        assert client.events[4] == ("delete_message", "mid.thinking.123")

    async def test_09_test_start_with_mid(self):
        # Case 3: clean_text empty, thinking MID exists => edit thinking message to: Запускаю тест.
        client = MockMaxClient(thinking_mid="mid.thinking.123")
        mock_start_test = AsyncMock()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="[START_TEST]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch("max_messenger_bot.services.tests.start_test", mock_start_test):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Хочу тест")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🤖 Думаю...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Запускаю тест.", "mid.thinking.123")
        mock_start_test.assert_awaited_once()

        # Case 1: clean_text exists, thinking MID exists => edit thinking message with clean_text.
        client2 = MockMaxClient(thinking_mid="mid.thinking.456")
        mock_start_test2 = AsyncMock()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="Отлично, начинаем!\n[START_TEST]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch("max_messenger_bot.services.tests.start_test", mock_start_test2):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client2, chat_id=100, user_id=200, prompt_text="Хочу тест")

        assert len(client2.events) == 2
        assert client2.events[0] == ("send_message", "🤖 Думаю...", "mid.thinking.456")
        assert client2.events[1] == ("edit_message", "Отлично, начинаем!", "mid.thinking.456")
        mock_start_test2.assert_awaited_once()

    async def test_10_test_start_without_mid(self):
        # Case 4: clean_text empty, thinking MID missing => send: Запускаю тест.
        client = MockMaxClient(thinking_mid=None)
        mock_start_test = AsyncMock()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="[START_TEST]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch("max_messenger_bot.services.tests.start_test", mock_start_test):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Хочу тест")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🤖 Думаю...", None)
        assert client.events[1] == ("send_message", "Запускаю тест.", "mid.msg.1")
        mock_start_test.assert_awaited_once()

        # Case 2: clean_text exists, thinking MID missing => send clean_text as normal message.
        client2 = MockMaxClient(thinking_mid=None)
        mock_start_test2 = AsyncMock()
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="Отлично, начинаем!\n[START_TEST]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch("max_messenger_bot.services.tests.start_test", mock_start_test2):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client2, chat_id=100, user_id=200, prompt_text="Хочу тест")

        assert len(client2.events) == 2
        assert client2.events[0] == ("send_message", "🤖 Думаю...", None)
        assert client2.events[1] == ("send_message", "Отлично, начинаем!", "mid.msg.1")
        mock_start_test2.assert_awaited_once()

    async def test_11_vision_normal_success(self):
        client = MockMaxClient()
        mock_analyze = AsyncMock(return_value="На изображении прекрасный закат.")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch("max_messenger_bot.ai.analyze_image", mock_analyze):
            await common.run_ai_dialogue_with_image(client, chat_id=100, user_id=200, image_bytes=b"img", caption="Что тут?")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🤖 Анализирую изображение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "На изображении прекрасный закат.", "mid.thinking.123")

    async def test_12_vision_ai_service_error_final_failure(self):
        client = MockMaxClient()
        mock_analyze = AsyncMock(side_effect=AIServiceError("vision unavailable"))
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch("max_messenger_bot.ai.analyze_image", mock_analyze):
            await common.run_ai_dialogue_with_image(client, chat_id=100, user_id=200, image_bytes=b"img", caption="Что тут?")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🤖 Анализирую изображение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Сервис анализа изображений временно недоступен.", "mid.thinking.123")

    async def test_13_vision_edit_img_only(self):
        client = MockMaxClient()
        mock_analyze = AsyncMock(return_value="EDIT_IMG: [сделай ярче]")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "edit_image", AsyncMock(return_value=b"edited_png")), \
             patch("max_messenger_bot.ai.analyze_image", mock_analyze):
            await common.run_ai_dialogue_with_image(client, chat_id=100, user_id=200, image_bytes=b"img", caption="Отредактируй")

        assert client.events[0] == ("send_message", "🤖 Анализирую изображение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "🎨 Редактирую ваше фото...", "mid.thinking.123")
        assert client.events[2] == ("upload_file", "image")
        assert client.events[3] == ("send_media_attachment", "image", "file_token_abc")
        assert client.events[4] == ("delete_message", "mid.thinking.123")

    async def test_14_vision_gen_img_only(self):
        client = MockMaxClient()
        mock_analyze = AsyncMock(return_value="GEN_IMG: [новый пейзаж]")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", AsyncMock(return_value=b"new_png")), \
             patch("max_messenger_bot.ai.analyze_image", mock_analyze):
            await common.run_ai_dialogue_with_image(client, chat_id=100, user_id=200, image_bytes=b"img", caption="Сделай похожее")

        assert client.events[0] == ("send_message", "🤖 Анализирую изображение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "🖼 Генерирую новое изображение...", "mid.thinking.123")
        assert client.events[2] == ("upload_file", "image")
        assert client.events[3] == ("send_media_attachment", "image", "file_token_abc")
        assert client.events[4] == ("delete_message", "mid.thinking.123")

    async def test_15_image_edit_status_cleanup_with_visible_text(self):
        client = MockMaxClient()
        mock_analyze = AsyncMock(return_value="Вот обработанный снимок.\nEDIT_IMG: [сделай ярче]")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "edit_image", AsyncMock(return_value=b"edited_png")), \
             patch("max_messenger_bot.ai.analyze_image", mock_analyze):
            await common.run_ai_dialogue_with_image(client, chat_id=100, user_id=200, image_bytes=b"img", caption="Сделай")

        assert client.events[0] == ("send_message", "🤖 Анализирую изображение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Вот обработанный снимок.", "mid.thinking.123")
        assert client.events[2] == ("send_message", "🎨 Редактирую ваше фото...", "mid.edit.789")
        assert client.events[3] == ("upload_file", "image")
        assert client.events[4] == ("send_media_attachment", "image", "file_token_abc")
        assert client.events[5] == ("delete_message", "mid.edit.789")

    async def test_16_image_generation_status_cleanup_with_visible_text(self):
        client = MockMaxClient()
        mock_analyze = AsyncMock(return_value="Вот новая картинка.\nGEN_IMG: [пейзаж]")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", AsyncMock(return_value=b"gen_png")), \
             patch("max_messenger_bot.ai.analyze_image", mock_analyze):
            await common.run_ai_dialogue_with_image(client, chat_id=100, user_id=200, image_bytes=b"img", caption="Сделай")

        assert client.events[0] == ("send_message", "🤖 Анализирую изображение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Вот новая картинка.", "mid.thinking.123")
        assert client.events[2] == ("send_message", "🖼 Генерирую новое изображение...", "mid.gen.456")
        assert client.events[3] == ("upload_file", "image")
        assert client.events[4] == ("send_media_attachment", "image", "file_token_abc")
        assert client.events[5] == ("delete_message", "mid.gen.456")

    async def test_17_voice_success(self):
        client = MockMaxClient()
        with patch("max_messenger_bot.ai.transcribe_audio", AsyncMock(return_value="Привет, бот")), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "ensure_access_before_chat", AsyncMock(return_value=False)):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_voice(client, chat_id=100, user_id=200, audio_bytes=b"voice")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🎙 Распознаю голосовое сообщение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "🎙 <i>Привет, бот</i>", "mid.thinking.123")

    async def test_18_voice_ai_service_error(self):
        client = MockMaxClient()
        with patch("max_messenger_bot.ai.transcribe_audio", AsyncMock(side_effect=AIServiceError("transcription err"))):
            await common.run_ai_dialogue_with_voice(client, chat_id=100, user_id=200, audio_bytes=b"voice")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🎙 Распознаю голосовое сообщение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Не удалось распознать аудио. Сервис временно недоступен.", "mid.thinking.123")

    async def test_19_voice_unexpected_exception(self):
        client = MockMaxClient()
        with patch("max_messenger_bot.ai.transcribe_audio", AsyncMock(side_effect=RuntimeError("transcription crash"))):
            await common.run_ai_dialogue_with_voice(client, chat_id=100, user_id=200, audio_bytes=b"voice")

        assert len(client.events) == 2
        assert client.events[0] == ("send_message", "🎙 Распознаю голосовое сообщение...", "mid.thinking.123")
        assert client.events[1] == ("edit_message", "Произошла ошибка при распознавании аудио.", "mid.thinking.123")


@pytest.mark.asyncio
class TestA1MaxImageProgressBestEffort:
    """Proves that image progress status delivery is best-effort and never gates image operations."""

    async def test_chat_gen_img_with_visible_text_progress_send_failure_does_not_gate_image(self):
        client = MockMaxClient(thinking_mid="mid.thinking.123")
        orig_send = client.send_message
        async def failing_send(chat_id, text, **kwargs):
            if "Генерирую" in text:
                raise RuntimeError("MAX transport unavailable for status send")
            return await orig_send(chat_id, text, **kwargs)
        client.send_message = failing_send

        gen_mock = AsyncMock(return_value=b"fake_image_bytes")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="Красивый пейзаж.\nGEN_IMG: [лес]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Нарисуй лес")

        gen_mock.assert_awaited_once_with("лес")
        upload_events = [e for e in client.events if e[0] == "upload_file"]
        send_media_events = [e for e in client.events if e[0] == "send_media_attachment"]
        assert len(upload_events) == 1
        assert len(send_media_events) == 1

    async def test_chat_gen_img_only_progress_edit_failure_does_not_gate_image_and_preserves_mid(self):
        client = MockMaxClient(thinking_mid="mid.thinking.123")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if "Генерирую" in text:
                raise RuntimeError("MAX transport unavailable for status edit")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        gen_mock = AsyncMock(return_value=b"fake_image_bytes")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="GEN_IMG: [лес]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Нарисуй лес")

        gen_mock.assert_awaited_once_with("лес")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.thinking.123" for e in delete_events)

    async def test_hidden_kickoff_gen_img_progress_failure_does_not_gate_image(self):
        client = MockMaxClient(thinking_mid="mid.thinking.kickoff")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if "Генерирую" in text:
                raise RuntimeError("MAX transport edit failed")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        gen_mock = AsyncMock(return_value=b"fake_image_bytes")
        mock_user = MagicMock()
        mock_user.current_dialogue_id = 1
        mock_user.current_topic_id = None
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_user

        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="GEN_IMG: [космос]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_hidden_ai_kickoff(
                client, chat_id=100, user_id=200, synthetic_prompt="[kickoff]",
                expected_dialogue_id=1, expected_topic_id=None
            )

        gen_mock.assert_awaited_once_with("космос")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.thinking.kickoff" for e in delete_events)

    async def test_vision_edit_img_with_visible_text_progress_send_failure_does_not_gate_edit(self):
        client = MockMaxClient(thinking_mid="mid.analyzing.123")
        orig_send = client.send_message
        async def failing_send(chat_id, text, **kwargs):
            if "Редактирую" in text:
                raise RuntimeError("MAX transport send edit status failed")
            return await orig_send(chat_id, text, **kwargs)
        client.send_message = failing_send

        edit_mock = AsyncMock(return_value=b"fake_edited_bytes")
        with patch.object(common, "async_session_maker") as mock_session_maker, \
             patch("max_messenger_bot.ai.analyze_image", AsyncMock(return_value="Результат анализа.\nEDIT_IMG: [сделай ярче]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "edit_image", edit_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_image(
                client, chat_id=100, user_id=200, image_bytes=b"input_bytes", caption="Сделай ярче"
            )

        edit_mock.assert_awaited_once_with("[сделай ярче]", b"input_bytes")
        send_media_events = [e for e in client.events if e[0] == "send_media_attachment"]
        assert len(send_media_events) == 1

    async def test_vision_edit_img_only_progress_edit_failure_does_not_gate_edit_and_preserves_mid(self):
        client = MockMaxClient(thinking_mid="mid.analyzing.123")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if "Редактирую" in text:
                raise RuntimeError("MAX transport edit status failed")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        edit_mock = AsyncMock(return_value=b"fake_edited_bytes")
        with patch.object(common, "async_session_maker") as mock_session_maker, \
             patch("max_messenger_bot.ai.analyze_image", AsyncMock(return_value="EDIT_IMG: [сделай ярче]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "edit_image", edit_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_image(
                client, chat_id=100, user_id=200, image_bytes=b"input_bytes", caption="Сделай ярче"
            )

        edit_mock.assert_awaited_once_with("[сделай ярче]", b"input_bytes")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.analyzing.123" for e in delete_events)

    async def test_vision_gen_img_with_visible_text_progress_send_failure_does_not_gate_gen(self):
        client = MockMaxClient(thinking_mid="mid.analyzing.123")
        orig_send = client.send_message
        async def failing_send(chat_id, text, **kwargs):
            if "Генерирую" in text:
                raise RuntimeError("MAX transport gen status failed")
            return await orig_send(chat_id, text, **kwargs)
        client.send_message = failing_send

        gen_mock = AsyncMock(return_value=b"fake_gen_bytes")
        with patch.object(common, "async_session_maker") as mock_session_maker, \
             patch("max_messenger_bot.ai.analyze_image", AsyncMock(return_value="Текст анализа.\nGEN_IMG: [новый пейзаж]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_image(
                client, chat_id=100, user_id=200, image_bytes=b"input_bytes", caption="Сделай новое"
            )

        gen_mock.assert_awaited_once_with("[новый пейзаж]")
        send_media_events = [e for e in client.events if e[0] == "send_media_attachment"]
        assert len(send_media_events) == 1

    async def test_vision_gen_img_only_progress_edit_failure_does_not_gate_gen_and_preserves_mid(self):
        client = MockMaxClient(thinking_mid="mid.analyzing.456")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if "Генерирую" in text:
                raise RuntimeError("MAX transport gen status edit failed")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        gen_mock = AsyncMock(return_value=b"fake_gen_bytes")
        with patch.object(common, "async_session_maker") as mock_session_maker, \
             patch("max_messenger_bot.ai.analyze_image", AsyncMock(return_value="GEN_IMG: [новый пейзаж]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_image(
                client, chat_id=100, user_id=200, image_bytes=b"input_bytes", caption="Сделай новое"
            )

        gen_mock.assert_awaited_once_with("[новый пейзаж]")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.analyzing.456" for e in delete_events)

    async def test_chat_visible_text_gen_img_first_edit_fails_retains_and_cleans_mid(self):
        client = MockMaxClient(thinking_mid="mid.thinking.chat_retain")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if message_id == "mid.thinking.chat_retain" and "Красивый" in text:
                raise RuntimeError("MAX transport edit failed for visible text")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        gen_mock = AsyncMock(return_value=b"fake_image_bytes")
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="Красивый лес.\nGEN_IMG: [лес]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Нарисуй лес")

        gen_mock.assert_awaited_once_with("лес")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.thinking.chat_retain" for e in delete_events)

    async def test_hidden_kickoff_visible_text_gen_img_first_edit_fails_retains_and_cleans_mid(self):
        client = MockMaxClient(thinking_mid="mid.thinking.kickoff_retain")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if message_id == "mid.thinking.kickoff_retain" and "Привет" in text:
                raise RuntimeError("MAX transport edit failed for kickoff visible text")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        gen_mock = AsyncMock(return_value=b"fake_image_bytes")
        mock_user = MagicMock()
        mock_user.current_dialogue_id = 1
        mock_user.current_topic_id = None
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_user

        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value="Привет!\nGEN_IMG: [космос]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_hidden_ai_kickoff(
                client, chat_id=100, user_id=200, synthetic_prompt="[kickoff]",
                expected_dialogue_id=1, expected_topic_id=None
            )

        gen_mock.assert_awaited_once_with("космос")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.thinking.kickoff_retain" for e in delete_events)

    async def test_vision_edit_img_visible_text_first_edit_fails_retains_and_cleans_mid(self):
        client = MockMaxClient(thinking_mid="mid.analyzing.edit_retain")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if message_id == "mid.analyzing.edit_retain" and "Красивое" in text:
                raise RuntimeError("MAX transport edit failed for vision visible text")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        edit_mock = AsyncMock(return_value=b"fake_edited_bytes")
        with patch.object(common, "async_session_maker") as mock_session_maker, \
             patch("max_messenger_bot.ai.analyze_image", AsyncMock(return_value="Красивое фото.\nEDIT_IMG: [сделай ярче]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "edit_image", edit_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_image(
                client, chat_id=100, user_id=200, image_bytes=b"input_bytes", caption="Сделай ярче"
            )

        edit_mock.assert_awaited_once_with("[сделай ярче]", b"input_bytes")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.analyzing.edit_retain" for e in delete_events)

    async def test_vision_gen_img_visible_text_first_edit_fails_retains_and_cleans_mid(self):
        client = MockMaxClient(thinking_mid="mid.analyzing.gen_retain")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if message_id == "mid.analyzing.gen_retain" and "Анализ" in text:
                raise RuntimeError("MAX transport edit failed for vision gen visible text")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        gen_mock = AsyncMock(return_value=b"fake_gen_bytes")
        with patch.object(common, "async_session_maker") as mock_session_maker, \
             patch("max_messenger_bot.ai.analyze_image", AsyncMock(return_value="Анализ завершен.\nGEN_IMG: [новый закат]")), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_image(
                client, chat_id=100, user_id=200, image_bytes=b"input_bytes", caption="Сделай новое"
            )

        gen_mock.assert_awaited_once_with("[новый закат]")
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert any(e[1] == "mid.analyzing.gen_retain" for e in delete_events)

    async def test_vision_multi_chunk_first_edit_succeeds_later_send_fails_does_not_overwrite_or_delete_visible_chunk(self):
        client = MockMaxClient(thinking_mid="mid.analyzing.first_chunk")
        orig_send = client.send_message
        async def failing_send(chat_id, text, **kwargs):
            if "chunk_two" in text:
                raise RuntimeError("MAX transport send failed on chunk 2")
            return await orig_send(chat_id, text, **kwargs)
        client.send_message = failing_send

        # Two chunks response
        long_response = ("chunk_one " * 400) + "\n\n" + ("chunk_two " * 400) + "\nEDIT_IMG: [сделай ярче]"

        edit_mock = AsyncMock(return_value=b"fake_edited_bytes")
        with patch.object(common, "async_session_maker") as mock_session_maker, \
             patch("max_messenger_bot.ai.analyze_image", AsyncMock(return_value=long_response)), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "edit_image", edit_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue_with_image(
                client, chat_id=100, user_id=200, image_bytes=b"input_bytes", caption="Сделай ярче"
            )

        edit_mock.assert_awaited_once_with("[сделай ярче]", b"input_bytes")
        # Ensure mid.analyzing.first_chunk was edited with chunk_one
        first_edit = [e for e in client.events if e[0] == "edit_message" and e[2] == "mid.analyzing.first_chunk"]
        assert len(first_edit) == 1
        assert "chunk_one" in first_edit[0][1]

        # Crucial invariant: mid.analyzing.first_chunk MUST NOT be deleted as temporary progress
        delete_events = [e for e in client.events if e[0] == "delete_message"]
        assert not any(e[1] == "mid.analyzing.first_chunk" for e in delete_events)

    async def test_response_buttons_visible_edit_fails_fallback_buttons_delivered(self):
        client = MockMaxClient(thinking_mid="mid.thinking.btn_fail")
        orig_edit = client.edit_message
        async def failing_edit(message_id, text, **kwargs):
            if message_id == "mid.thinking.btn_fail" and "Выберите" in text:
                raise RuntimeError("MAX transport edit failed for visible text with buttons")
            return await orig_edit(message_id, text, **kwargs)
        client.edit_message = failing_edit

        gen_mock = AsyncMock(return_value=b"fake_image_bytes")
        ai_resp = "Выберите вариант\n[Кнопка 1](btn:b1)\nGEN_IMG: [пейзаж]"
        with patch.object(common, "save_user_message", AsyncMock(return_value=None)), \
             patch.object(common, "async_session_maker") as mock_session_maker, \
             patch.object(common, "get_ai_response", AsyncMock(return_value=ai_resp)), \
             patch.object(common, "save_ai_message", AsyncMock()), \
             patch.object(common, "generate_image", gen_mock):
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await common.run_ai_dialogue(client, chat_id=100, user_id=200, prompt_text="Нарисуй")

        gen_mock.assert_awaited_once_with("пейзаж")
        # Check that fallback buttons were sent
        btn_sends = [e for e in client.events if e[0] == "send_message" and "Выберите действие:" in str(e[1])]
        assert len(btn_sends) == 1

