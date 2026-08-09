import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration


class _Response:
    status_code = 200

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": "Готово"}}]}


class _HttpClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, *args, **kwargs):
        return _Response()


class AIRequestContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_context_contains_profile_state_and_current_metadata(self):
        user = SimpleNamespace(
            id=42,
            name="Ясна",
            first_name="Yasna",
            gender="female",
            age="10",
            subscription=None,
        )
        session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace()))
        automation_context = (
            'СЛУЖЕБНЫЕ ДАННЫЕ ТЕКУЩЕГО ДИАЛОГА:\n'
            '{"current_state":{"current_step":"photo"},"metadata":{"guide":"yoda"}}'
        )
        with (
            patch.object(ai_integration, "build_runtime_automation_context", AsyncMock(return_value=automation_context)),
            patch.object(ai_integration, "active_subscription_flag", return_value=""),
        ):
            result = await ai_integration.build_runtime_context(
                session,
                user=user,
                dialogue_id=7,
                topic_id=3,
            )

        self.assertIn("ИМЯ: Ясна", result)
        self.assertIn("ПОЛ: female", result)
        self.assertIn('"current_step":"photo"', result)
        self.assertIn('"guide":"yoda"', result)
        self.assertNotIn("СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ", result)

    async def test_kie_multimodal_keeps_history_as_messages_and_captures_full_payload(self):
        history = [
            SimpleNamespace(role="user", content="guide_3"),
            SimpleNamespace(role="assistant", content="[1](btn:a) | [2](btn:b) | [3](btn:c) | [4](btn:d)"),
        ]
        capture = {}
        with patch.object(ai_integration.httpx, "AsyncClient", return_value=_HttpClient()):
            result = await ai_integration._call_kie_multimodal(
                "secret",
                "https://api.example",
                "gemini-2.5-flash",
                "SYSTEM",
                [{"type": "text", "text": "RUNTIME"}, {"type": "image_url", "image_url": {"url": "https://image"}}],
                temperature=0.1,
                history=history,
                request_capture=capture,
            )

        self.assertEqual(result, "Готово")
        payload = capture["payload"]
        self.assertEqual(payload["model"], "gemini-2.5-flash")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user", "assistant", "user"])
        self.assertIn("[2](btn:b)", payload["messages"][2]["content"])
        self.assertEqual(payload["messages"][-1]["content"][1]["image_url"]["url"], "https://image")
        self.assertNotIn("secret", capture["endpoint"])

    async def test_kie_vision_keeps_runtime_context_out_of_user_message(self):
        multimodal = AsyncMock(return_value="Готово")
        with (
            patch.object(ai_integration, "_upload_file_to_kie", AsyncMock(return_value="https://image")),
            patch.object(ai_integration, "_call_kie_multimodal", multimodal),
        ):
            await ai_integration._call_kie_vision(
                "secret",
                "https://api.example",
                "https://upload.example",
                "gemini-2.5-flash",
                b"image",
                "SYSTEM PROMPT",
                history=[],
                request_context="RUNTIME METADATA",
            )

        args = multimodal.await_args.args
        self.assertIn("RUNTIME METADATA", args[3])
        self.assertNotIn("RUNTIME METADATA", args[4][0]["text"])
