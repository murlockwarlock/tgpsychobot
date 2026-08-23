import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration
from ai_request_context import AIRequestLayout, AIRequestMessage


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


class _CompletionClient:
    calls = []

    class _Completions:
        async def create(self, **payload):
            _CompletionClient.calls.append(payload)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Готово"))]
            )

    class _Chat:
        def __init__(self):
            self.completions = _CompletionClient._Completions()

    def __init__(self, *args, **kwargs):
        self.chat = self._Chat()

    async def close(self):
        return None


class _ClaudeClient:
    calls = []

    class _Messages:
        async def create(self, **payload):
            _ClaudeClient.calls.append(payload)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Готово")])

    def __init__(self, *args, **kwargs):
        self.messages = self._Messages()


class _GeminiResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {"candidates": [{"content": {"parts": [{"text": "Готово"}]}}]}


class _GeminiClient:
    calls = []

    def __init__(self, *args, **kwargs):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, endpoint, **kwargs):
        self.calls.append({"endpoint": endpoint, "kwargs": kwargs})
        _GeminiClient.calls.append(kwargs["json"])
        return _GeminiResponse()


def _canonical_layout(client_name="Ясна", state="photo"):
    return AIRequestLayout(
        stable_system_prompt="STABLE CONFIGURED TOPIC PROMPT",
        shared_instructions=("SHARED TOPIC INSTRUCTIONS",),
        runtime_context=(f"ДАННЫЕ КЛИЕНТА:\nИМЯ: {client_name}",),
        scenario_context=(
            f'СЛУЖЕБНЫЕ ДАННЫЕ ТЕКУЩЕГО ДИАЛОГА: '
            f'{{"current_state":"{state}","metadata":"meta-{client_name}"}}',
        ),
        request_context=("TEST RESULTS", "KB RAG RESULTS", "GLOBAL MEMORY"),
        history=(
            AIRequestMessage("user", "предыдущий вопрос"),
            AIRequestMessage("assistant", "предыдущий ответ"),
        ),
        current_user_content="текущий вопрос",
    )


def _assert_direct_payload_layout(provider, payload, client_name="Ясна"):
    stable = "STABLE CONFIGURED TOPIC PROMPT"
    state = "photo"
    metadata = f"meta-{client_name}"
    if provider in {"OpenAI", "Deepseek"}:
        system_blocks = [
            message["content"]
            for message in payload["messages"]
            if message["role"] == "system"
        ]
        history = payload["messages"][-3:-1]
        current = payload["messages"][-1]
    elif provider == "Claude":
        system_blocks = [block["text"] for block in payload["system"]]
        history = payload["messages"][-3:-1]
        current = payload["messages"][-1]
    else:
        system_blocks = [part["text"] for part in payload["systemInstruction"]["parts"]]
        history = payload["contents"][-3:-1]
        current = payload["contents"][-1]

    assert system_blocks[0] == stable
    assert client_name not in system_blocks[0]
    assert state not in system_blocks[0]
    assert metadata not in system_blocks[0]
    dynamic_index = next(i for i, block in enumerate(system_blocks) if client_name in block)
    assert state in system_blocks[dynamic_index]
    assert metadata in system_blocks[dynamic_index]
    assert dynamic_index > 0
    shared_index = next(i for i, block in enumerate(system_blocks) if "SHARED TOPIC INSTRUCTIONS" in block)
    assert 0 < shared_index < dynamic_index
    request_indices = [
        i for i, block in enumerate(system_blocks)
        if any(value in block for value in ("TEST RESULTS", "KB RAG RESULTS", "GLOBAL MEMORY"))
    ]
    assert request_indices
    assert min(request_indices) > dynamic_index
    assert all(value in "\n".join(system_blocks) for value in ("TEST RESULTS", "KB RAG RESULTS", "GLOBAL MEMORY"))
    assert history[0]["role"] in {"user", "model"}
    assert "предыдущий вопрос" in str(history[0])
    assert "предыдущий ответ" in str(history[1])
    assert current["role"] in {"user", "model"}
    assert "текущий вопрос" in str(current)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("OpenAI", "Deepseek", "Claude", "Gemini"))
async def test_direct_final_payload_preserves_canonical_layout(monkeypatch, provider):
    capture = {}
    layout = _canonical_layout()
    _CompletionClient.calls.clear()
    _ClaudeClient.calls.clear()
    _GeminiClient.calls.clear()

    if provider == "OpenAI":
        monkeypatch.setattr(ai_integration, "AsyncOpenAI", _CompletionClient)
        await ai_integration._call_openai_api(
            "secret", "model", [], "", "STABLE", request_capture=capture,
            request_layout=layout,
        )
    elif provider == "Deepseek":
        monkeypatch.setattr(ai_integration, "AsyncOpenAI", _CompletionClient)
        await ai_integration._call_deepseek_api(
            "secret", "model", [], "", "STABLE", use_proxy=False,
            request_capture=capture, request_layout=layout,
        )
    elif provider == "Claude":
        monkeypatch.setattr(ai_integration.anthropic, "AsyncAnthropic", _ClaudeClient)
        await ai_integration._call_claude_api(
            "secret", "model", [], "", "STABLE", request_capture=capture,
            request_layout=layout,
        )
    else:
        monkeypatch.setattr(ai_integration.httpx, "AsyncClient", _GeminiClient)
        await ai_integration._call_gemini_api(
            "secret", "model", [], "", "STABLE", request_capture=capture,
            request_layout=layout,
        )

    assert capture["payload"]
    _assert_direct_payload_layout(provider, capture["payload"])


@pytest.mark.asyncio
async def test_deepseek_reference_shape_is_stable_then_one_dynamic_block(monkeypatch):
    _CompletionClient.calls.clear()
    monkeypatch.setattr(ai_integration, "AsyncOpenAI", _CompletionClient)
    capture = {}
    layout = AIRequestLayout(
        stable_system_prompt="STABLE",
        runtime_context=("CLIENT Alice",),
        scenario_context=("STATE photo / META one",),
        history=(
            AIRequestMessage("user", "start_test"),
            AIRequestMessage("user", "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]"),
            AIRequestMessage("assistant", "old answer"),
        ),
        current_user_content="story_yes",
    )
    await ai_integration._call_deepseek_api(
        "secret", "deepseek-chat", [], "", "STABLE", use_proxy=False,
        request_capture=capture, request_layout=layout,
    )
    messages = capture["payload"]["messages"]
    assert [message["role"] for message in messages] == [
        "system", "system", "user", "user", "assistant", "user"
    ]
    assert messages[0]["content"] == "STABLE"
    assert "Alice" not in messages[0]["content"]
    assert "photo" not in messages[0]["content"]
    assert "meta" not in messages[0]["content"]
    assert "Alice" in messages[1]["content"]
    assert "photo" in messages[1]["content"]
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" in messages[3]["content"]
    assert messages[-1]["content"] == "story_yes"


@pytest.mark.asyncio
async def test_deepseek_stable_placeholders_are_neutralized(monkeypatch):
    _CompletionClient.calls.clear()
    monkeypatch.setattr(ai_integration, "AsyncOpenAI", _CompletionClient)
    capture = {}
    await ai_integration._call_deepseek_api(
        "secret", "deepseek-chat", [], "", "STABLE {user_name} {user_gender}",
        use_proxy=False,
        request_capture=capture,
        request_layout=AIRequestLayout(
            stable_system_prompt="STABLE {user_name} {user_gender}",
            shared_instructions=("SHARED {test_results}",),
            runtime_context=("CLIENT Alice",),
            current_user_content="current question",
        ),
    )

    messages = capture["payload"]["messages"]
    stable_content = messages[0]["content"]
    shared_content = messages[1]["content"]
    assert "{user_name}" not in stable_content
    assert "{user_gender}" not in stable_content
    assert "Alice" not in stable_content
    assert "{test_results}" not in shared_content
    assert "Alice" in messages[2]["content"]


@pytest.mark.asyncio
async def test_cache_regression_stable_block_is_identical_across_users(monkeypatch):
    _CompletionClient.calls.clear()
    monkeypatch.setattr(ai_integration, "AsyncOpenAI", _CompletionClient)
    captures = []
    for client_name, state in (("Alice", "one"), ("Boris", "two")):
        capture = {}
        await ai_integration._call_deepseek_api(
            "secret", "model", [], "", "STABLE", use_proxy=False,
            request_capture=capture,
            request_layout=AIRequestLayout(
                stable_system_prompt="STABLE",
                runtime_context=(f"CLIENT {client_name}",),
                scenario_context=(f"STATE {state}",),
                current_user_content="same question",
            ),
        )
        captures.append(capture["payload"])

    first, second = (payload["messages"] for payload in captures)
    assert first[0] == second[0] == {"role": "system", "content": "STABLE"}
    assert first[1] != second[1]
    assert "Alice" in first[1]["content"] and "STATE one" in first[1]["content"]
    assert "Boris" in second[1]["content"] and "STATE two" in second[1]["content"]
    assert "Alice" not in first[0]["content"] and "Boris" not in second[0]["content"]


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
        self.assertEqual(args[3], "SYSTEM PROMPT")
        self.assertNotIn("RUNTIME METADATA", args[4][0]["text"])
        layout = multimodal.await_args.kwargs["request_layout"]
        self.assertIn("RUNTIME METADATA", layout.runtime_context)
