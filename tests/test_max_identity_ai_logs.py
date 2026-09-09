import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import database
import handlers
from ai_request_context import AIRequestLayout
from database import AILog, Base, User
from max_messenger_bot import ai as max_ai
from max_messenger_bot import keyboards as max_keyboards
from max_messenger_bot import models as max_models
from max_messenger_bot.services import admin_clients, common


class _MockOpenAIChatChoice:
    def __init__(self, content="Mocked OpenAI response"):
        self.message = SimpleNamespace(content=content)


class _MockOpenAIChatCompletion:
    def __init__(self, content="Mocked OpenAI response"):
        self.choices = [_MockOpenAIChatChoice(content)]


class _MockOpenAIClient:
    calls = []
    response_content = "Mocked OpenAI response"

    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        _MockOpenAIClient.calls.append({"kwargs": kwargs, "api_key": self.api_key, "base_url": self.base_url})
        return _MockOpenAIChatCompletion(self.response_content)


class _MockClaudeContentBlock:
    def __init__(self, text="Mocked Claude response"):
        self.text = text


class _MockClaudeMessage:
    def __init__(self, text="Mocked Claude response"):
        self.content = [_MockClaudeContentBlock(text)]


class _MockAnthropicClient:
    calls = []
    response_content = "Mocked Claude response"

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        _MockAnthropicClient.calls.append({"kwargs": kwargs, "api_key": self.api_key})
        return _MockClaudeMessage(self.response_content)


class _MockHttpxResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP error {self.status_code}")


class _MockHttpxClient:
    calls = []
    response_json = None
    response_text = ""
    status_code = 200

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, json=None, headers=None, **kwargs):
        _MockHttpxClient.calls.append({"url": url, "json": json, "headers": headers, "kwargs": kwargs})
        if self.status_code >= 400:
            resp = _MockHttpxResponse(self.status_code, self.response_json or {}, self.text)
            resp.raise_for_status()
        if self.response_json is not None:
            return _MockHttpxResponse(self.status_code, self.response_json, self.response_text)
        default_gemini = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Mocked Gemini response"}]
                }
            }]
        }
        return _MockHttpxResponse(self.status_code, default_gemini, self.response_text)


class _Result:
    def __init__(self, *, scalar=None, rows=()):
        self._scalar = scalar
        self._rows = list(rows)

    def scalar(self):
        return self._scalar

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_args):
        return False


class _EnsureUserSession:
    def __init__(self, user):
        self.user = user
        self.commits = 0

    async def get(self, _model, _key):
        return self.user

    async def commit(self):
        self.commits += 1


class _MaxAISession:
    def __init__(self, user, config):
        self.user = user
        self.config = config
        self.added = []
        self.commits = 0

    async def scalar(self, _statement):
        return self.user

    async def get(self, model, _key, **_kwargs):
        return self.config if model is max_ai.AIConfig else None

    async def execute(self, _statement):
        return _Result(rows=[])

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1


class _TelegramAISession:
    def __init__(self, user, config):
        self.user = user
        self.config = config
        self.execute_count = 0
        self.added = []
        self.commits = 0

    async def execute(self, _statement):
        self.execute_count += 1
        if self.execute_count == 1:
            return _Result(scalar=self.user)
        return _Result(rows=[])

    async def get(self, model, _key):
        if model is handlers.AIConfig:
            return self.config
        return None

    def add(self, value):
        self.added.append(value)

    async def scalar(self, _statement):
        return None

    async def commit(self):
        self.commits += 1


class _DetailSession:
    def __init__(self, log_entry, user):
        self.log_entry = log_entry
        self.user = user

    async def get(self, model, _key, **_kwargs):
        if model is handlers.AILog:
            return self.log_entry
        return self.user


class _LogListSession:
    def __init__(self, logs):
        self.logs = logs
        self.execute_count = 0

    async def execute(self, _statement):
        self.execute_count += 1
        if self.execute_count == 1:
            return _Result(scalar=len(self.logs))
        return _Result(rows=self.logs)


def _max_user(*, topic_id=None, topic=None):
    return SimpleNamespace(
        id=max_models.MAX_ID_OFFSET + 100018792559,
        username=None,
        first_name="Зоя Александровна",
        name="Мама",
        gender="female",
        age=None,
        response_length="normal",
        current_dialogue_id=4,
        current_topic_id=topic_id,
        current_topic=topic,
        subscription=None,
        ai_debug_enabled=False,
        tg_user_id=None,
        is_admin=False,
        can_view_history=False,
    )


def _max_config():
    return SimpleNamespace(
        provider="Gemini",
        gemini_api_key="max-secret",
        gemini_model="gemini-max",
        system_prompt="SYSTEM",
        shared_prompt_block="",
        service_prompt_block=None,
        context_limit_first=2,
        context_limit_recent=10,
        memory_mode="reset",
        preserve_topic_context=False,
        temperature=0.7,
        allow_fallback=False,
        fallback_provider=None,
        fallback_model=None,
    )


def _telegram_config():
    return SimpleNamespace(
        provider="Gemini",
        gemini_api_key="telegram-secret",
        gemini_model="gemini-telegram",
        system_prompt="SYSTEM",
        prompt_mode="text",
        prompt_filename=None,
        shared_prompt_block="",
        service_prompt_block=None,
        temperature=0.7,
        context_limit_first=2,
        context_limit_recent=10,
        memory_mode="reset",
        preserve_topic_context=False,
        use_proxy=False,
        fallback_timeout=60,
        allow_fallback=False,
        vision_provider="Gemini",
        vision_model="gemini-vision",
    )


@pytest.mark.asyncio
async def test_existing_max_user_refreshes_public_name_without_changing_communication_name(monkeypatch):
    user = SimpleNamespace(
        id=max_models.MAX_ID_OFFSET + 55,
        first_name="Старое имя",
        name="Мама",
        username="mama",
    )
    session = _EnsureUserSession(user)
    monkeypatch.setattr(common, "async_session_maker", lambda: _SessionContext(session))

    result = await common.ensure_user(
        user.id,
        "mama",
        "Зоя Александровна",
        public_name="Зоя Александровна",
    )

    assert result is user
    assert user.first_name == "Зоя Александровна"
    assert user.name == "Мама"
    assert session.commits == 1


def test_max_client_list_label_uses_communication_and_public_names():
    user = SimpleNamespace(
        id=max_models.MAX_ID_OFFSET + 100018792559,
        name="Мама",
        first_name="Зоя Александровна",
        username=None,
    )

    markup = max_keyboards.admin_clients_keyboard(0, 1, [user])
    button = markup[0]["payload"]["buttons"][0][0]

    assert button["text"] == "Мама (Зоя Александровна)"
    assert str(max_models.MAX_ID_OFFSET + 100018792559) not in button["text"]


@pytest.mark.asyncio
async def test_max_client_profile_uses_raw_id_and_separate_identity_fields(monkeypatch):
    user = _max_user()
    user.created_at = datetime(2026, 8, 28, 10, 0)
    session = _DetailSession(None, user)
    client = SimpleNamespace(send_message=AsyncMock())
    monkeypatch.setattr(admin_clients, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(admin_clients, "load_active_subscription", AsyncMock(return_value=None))

    await admin_clients.show_client_profile(client, 9001, user.id)

    text_value = client.send_message.await_args.kwargs["text"]
    assert "<b>ID max:</b> <code>100018792559</code>" in text_value
    assert str(user.id) not in text_value
    assert "Имя для общения:</b> Мама" in text_value
    assert "Имя в max:</b> Зоя Александровна" in text_value
    assert "Username:</b> не указан" in text_value
    assert "t.me" not in text_value


@pytest.mark.asyncio
async def test_telegram_admin_max_profile_uses_raw_id_and_max_identity(monkeypatch):
    user = _max_user()
    user.created_at = datetime(2026, 8, 28, 10, 0)
    session = _DetailSession(None, user)
    callback = SimpleNamespace(
        data=f"view_client_{user.id}",
        from_user=SimpleNamespace(id=9001),
        message=SimpleNamespace(edit_text=AsyncMock()),
    )
    state = SimpleNamespace(update_data=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(handlers, "check_history_permission", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "OWNER_IDS", set())

    await handlers.view_client_profile(callback, state)

    text_value = callback.message.edit_text.await_args.args[0]
    assert "<b>ID max:</b> <code>100018792559</code>" in text_value
    assert str(user.id) not in text_value
    assert "Имя для общения:</b> Мама" in text_value
    assert "Имя в max:</b> Зоя Александровна" in text_value
    assert "Username:</b> не указан" in text_value
    assert "t.me" not in text_value


@pytest.mark.asyncio
async def test_max_chat_response_creates_shared_ai_log(monkeypatch):
    _MockHttpxClient.calls.clear()
    _MockHttpxClient.response_json = None
    _MockHttpxClient.status_code = 200
    monkeypatch.setattr(max_ai.httpx, "AsyncClient", _MockHttpxClient)
    user = _max_user()
    config = _max_config()
    config.gemini_model = "gemini-3.7-flash"
    session = _MaxAISession(user, config)
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))

    result = await max_ai.get_ai_response(user.id, "question")

    assert result == "Mocked Gemini response"
    assert len(session.added) == 1
    log_entry = session.added[0]
    assert isinstance(log_entry, AILog)
    assert log_entry.user_id == user.id
    assert log_entry.platform == "max"
    assert log_entry.context_kind == "main"
    assert log_entry.topic_id is None
    assert log_entry.topic_name_snapshot is None
    assert log_entry.request_payload
    assert "max-secret" not in log_entry.request_payload
    assert "?key=" not in log_entry.request_payload
    assert session.commits in (1, 2)


@pytest.mark.asyncio
async def test_max_primary_chat_log_records_provider_latency(monkeypatch):
    user = _max_user()
    session = _MaxAISession(user, _max_config())
    monotonic_values = iter((100.0, 100.5))
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(return_value="MAX answer"))
    monkeypatch.setattr(max_ai, "time", SimpleNamespace(monotonic=lambda: next(monotonic_values)))

    await max_ai.get_ai_response(user.id, "question")

    assert session.added[0].latency_ms == 500


@pytest.mark.asyncio
async def test_max_fallback_chat_log_records_whole_provider_latency_and_fallback_model(monkeypatch):
    user = _max_user()
    config = _max_config()
    config.openai_api_key = "fallback-secret"
    config.fallback_provider = "OpenAI"
    config.fallback_model = "gpt-fallback"
    config.allow_fallback = True
    session = _MaxAISession(user, config)
    monotonic_values = iter((200.0, 201.25))
    primary_error = max_ai.AIServiceError("primary provider failed")
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(side_effect=primary_error))
    monkeypatch.setattr(max_ai, "_call_openai", AsyncMock(return_value="fallback answer"))
    monkeypatch.setattr(max_ai, "time", SimpleNamespace(monotonic=lambda: next(monotonic_values)))

    result = await max_ai.get_ai_response(user.id, "question")

    assert result == "fallback answer"
    log_entry = session.added[0]
    assert log_entry.provider == "OpenAI"
    assert log_entry.model == "gpt-fallback"
    assert log_entry.latency_ms == 1250


@pytest.mark.asyncio
async def test_max_failed_chat_request_preserves_provider_error_and_does_not_create_log(monkeypatch):
    user = _max_user()
    session = _MaxAISession(user, _max_config())
    primary_error = max_ai.AIServiceError("primary provider failed")
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(side_effect=primary_error))

    with pytest.raises(max_ai.AIServiceError) as raised:
        await max_ai.get_ai_response(user.id, "question")

    assert raised.value is primary_error
    assert session.added == []


@pytest.mark.asyncio
async def test_max_log_is_visible_in_existing_global_and_per_user_log_ui(monkeypatch):
    user_id = max_models.MAX_ID_OFFSET + 55
    log_entry = AILog(
        id=42,
        user_id=user_id,
        provider="Gemini",
        model="gemini-max",
        prompt_summary="question",
        raw_response="answer",
        clean_text="answer",
        created_at=datetime(2026, 8, 28, 10, 0),
        platform="max",
        context_kind="main",
    )

    for filter_user_id in (None, user_id):
        session = _LogListSession([log_entry])
        event = SimpleNamespace(answer=AsyncMock())
        monkeypatch.setattr(handlers, "async_session_maker", lambda session=session: _SessionContext(session))

        await handlers.show_ai_logs_list(event, filter_user_id=filter_user_id)

        text_value = event.answer.await_args.args[0]
        markup = event.answer.await_args.kwargs["reply_markup"]
        buttons = [button for row in markup.inline_keyboard for button in row]
        assert "42" in " ".join(button.callback_data for button in buttons if button.callback_data)
        if filter_user_id:
            assert "ID max: 55" in text_value
            assert str(user_id) not in text_value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("topic_id", "topic_name", "expected_kind", "expected_name"),
    ((None, None, "main", None), (7, "Отношения", "topic", "Отношения")),
)
async def test_telegram_ai_log_records_request_time_context(
    monkeypatch,
    topic_id,
    topic_name,
    expected_kind,
    expected_name,
):
    topic = SimpleNamespace(name=topic_name, system_prompt=None, knowledge_base_files=[])
    user = SimpleNamespace(
        id=777,
        name="Анна",
        first_name="Anna",
        gender="female",
        age=None,
        response_length="normal",
        current_dialogue_id=3,
        current_topic_id=topic_id,
        current_topic=topic if topic_id is not None else None,
        subscription=None,
        ai_debug_enabled=False,
    )
    session = _TelegramAISession(user, _telegram_config())
    request_layout = AIRequestLayout(stable_system_prompt="SYSTEM", current_user_content="question")
    monkeypatch.setattr(ai_integration_module := __import__("ai_integration"), "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(ai_integration_module, "load_available_media", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(ai_integration_module, "build_ai_request_layout", AsyncMock(return_value=request_layout))
    monkeypatch.setattr(ai_integration_module, "_call_gemini_api", AsyncMock(return_value="Telegram answer"))

    result = await ai_integration_module.get_ai_response(
        user.id,
        "question",
        "Анна",
        "female",
        include_test_context=False,
    )

    assert result == "Telegram answer"
    log_entry = session.added[-1]
    assert log_entry.platform == "telegram"
    assert log_entry.context_kind == expected_kind
    assert log_entry.topic_id == topic_id
    assert log_entry.topic_name_snapshot == expected_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("topic_id", "topic_name", "expected_kind", "expected_name"),
    ((None, None, "main", None), (7, "Отношения", "topic", "Отношения")),
)
async def test_max_ai_log_records_platform_and_request_time_context(
    monkeypatch,
    topic_id,
    topic_name,
    expected_kind,
    expected_name,
):
    topic = SimpleNamespace(name=topic_name, system_prompt=None, knowledge_base_files=[])
    user = _max_user(topic_id=topic_id, topic=topic if topic_id is not None else None)
    session = _MaxAISession(user, _max_config())
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(return_value="MAX answer"))

    await max_ai.get_ai_response(user.id, "question")

    log_entry = session.added[-1]
    assert log_entry.platform == "max"
    assert log_entry.context_kind == expected_kind
    assert log_entry.topic_id == topic_id
    assert log_entry.topic_name_snapshot == expected_name


@pytest.mark.asyncio
async def test_historical_log_context_does_not_follow_current_topic(monkeypatch):
    log_entry = AILog(
        id=9,
        user_id=777,
        provider="Gemini",
        model="gemini",
        prompt_summary="question",
        raw_response="answer",
        clean_text="answer",
        created_at=datetime(2026, 8, 28, 10, 0),
        platform="telegram",
        context_kind="topic",
        topic_id=7,
        topic_name_snapshot="Отношения",
    )
    user = SimpleNamespace(
        id=777,
        username="anna",
        first_name="Анна",
        name="Анна",
        tg_user_id=None,
        current_topic_id=8,
        current_topic=SimpleNamespace(name="Другое"),
    )
    session = _DetailSession(log_entry, user)
    event = SimpleNamespace(answer=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _SessionContext(session))

    await handlers.show_ai_log_detail(event, log_entry.id)

    text_value = event.answer.await_args.args[0]
    assert "Тема диалога — «Отношения»" in text_value
    assert "Другое" not in text_value


@pytest.mark.asyncio
async def test_max_log_detail_uses_raw_identity_and_request_context(monkeypatch):
    user = _max_user()
    log_entry = AILog(
        id=12,
        user_id=user.id,
        provider="Gemini",
        model="gemini-max",
        prompt_summary="question",
        raw_response="answer",
        clean_text="answer",
        created_at=datetime(2026, 8, 28, 10, 0),
        platform="max",
        context_kind="main",
    )
    session = _DetailSession(log_entry, user)
    event = SimpleNamespace(answer=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _SessionContext(session))

    await handlers.show_ai_log_detail(event, log_entry.id)

    text_value = event.answer.await_args.args[0]
    assert "Платформа:</b> MAX" in text_value
    assert "ID max:</b> <code>100018792559</code>" in text_value
    assert str(user.id) not in text_value
    assert "Имя для общения:</b> Мама" in text_value
    assert "Имя в max:</b> Зоя Александровна" in text_value
    assert "Username:</b> не указан" in text_value
    assert "Контекст:</b> Основной диалог" in text_value

    file_content = handlers._build_ai_log_file_content(log_entry)
    assert "User ID: 100018792559" in file_content
    assert str(user.id) not in file_content


@pytest.mark.asyncio
async def test_legacy_log_without_context_renders_unrecorded_context(monkeypatch):
    log_entry = AILog(
        id=10,
        user_id=777,
        provider="Gemini",
        model="gemini",
        prompt_summary="question",
        raw_response="answer",
        created_at=datetime(2026, 8, 28, 10, 0),
    )
    user = SimpleNamespace(
        id=777,
        username="anna",
        first_name="Анна",
        name="Анна",
        tg_user_id=None,
    )
    session = _DetailSession(log_entry, user)
    event = SimpleNamespace(answer=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _SessionContext(session))

    await handlers.show_ai_log_detail(event, log_entry.id)

    text_value = event.answer.await_args.args[0]
    assert "Контекст:</b> не зафиксирован" in text_value


@pytest.mark.asyncio
async def test_telegram_identity_detail_remains_username_compatible(monkeypatch):
    log_entry = AILog(
        id=11,
        user_id=777,
        provider="Gemini",
        model="gemini",
        prompt_summary="question",
        raw_response="answer",
        created_at=datetime(2026, 8, 28, 10, 0),
        platform="telegram",
        context_kind="main",
    )
    user = SimpleNamespace(
        id=777,
        username="anna",
        first_name="Анна",
        name="Анна для общения",
        tg_user_id=None,
    )
    session = _DetailSession(log_entry, user)
    event = SimpleNamespace(answer=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _SessionContext(session))

    await handlers.show_ai_log_detail(event, log_entry.id)

    text_value = event.answer.await_args.args[0]
    assert "<b>@anna</b> (ID: 777)" in text_value
    assert "Telegram ID:</b> <code>777</code>" in text_value
    assert "Имя для общения:</b> Анна для общения" in text_value


@pytest.mark.asyncio
async def test_init_db_adds_ai_log_context_columns_to_legacy_schema(tmp_path, monkeypatch):
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ai-log-migration.db'}")
    test_sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "async_session_maker", test_sessions)

    try:
        async with test_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(text("DROP INDEX IF EXISTS ix_ai_logs_platform"))
            for column in ("platform", "context_kind", "topic_id", "topic_name_snapshot"):
                await connection.execute(text(f"ALTER TABLE ai_logs DROP COLUMN {column}"))

        await database.init_db()

        async with test_engine.connect() as connection:
            columns = await connection.run_sync(lambda sync_connection: inspect(sync_connection).get_columns("ai_logs"))
        assert {"platform", "context_kind", "topic_id", "topic_name_snapshot"}.issubset(
            {column["name"] for column in columns}
        )
    finally:
        await test_engine.dispose()


@pytest.mark.asyncio
async def test_max_chat_response_with_topic_kb_retrieves_chunks(monkeypatch):
    user = _max_user()
    topic = SimpleNamespace(
        id=42,
        name="Аурика",
        system_prompt=None,
        instruction=None,
        use_common_instruction=True,
        knowledge_base_files=[SimpleNamespace(id=101), SimpleNamespace(id=102)],
    )
    user.current_topic_id = 42
    user.current_topic = topic
    session = _MaxAISession(user, _max_config())
    mock_search = AsyncMock(return_value=["Чанк из базы знаний 1", "Чанк 2"])

    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "search_relevant_chunks", mock_search)
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(return_value="Ответ с контекстом КБ"))

    result = await max_ai.get_ai_response(user.id, "привет")

    assert result == "Ответ с контекстом КБ"
    mock_search.assert_awaited_once_with("привет", n_results=3, document_ids=[101, 102])


@pytest.mark.asyncio
async def test_max_chat_response_with_general_kb_retrieves_chunks(monkeypatch):
    user = _max_user()
    user.active_topic_id = None
    session = _MaxAISession(user, _max_config())
    mock_search = AsyncMock(return_value=["Общий чанк"])

    async def _execute_general_kb(stmt):
        sql = str(stmt)
        if "knowledge_base" in sql.lower():
            return _Result(rows=[(201, "guide.pdf", "текст документа")])
        return _Result(rows=[])

    session.execute = _execute_general_kb
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "search_relevant_chunks", mock_search)
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(return_value="Ответ общий КБ"))

    result = await max_ai.get_ai_response(user.id, "вопрос")

    assert result == "Ответ общий КБ"
    mock_search.assert_awaited_once_with("вопрос", n_results=3, document_ids=[201])


@pytest.mark.asyncio
async def test_max_chat_response_loads_prompt_from_file(monkeypatch, tmp_path):
    user = _max_user()
    config = _max_config()
    config.prompt_mode = "file"
    config.prompt_filename = "test_aurika_prompt.txt"
    config.system_prompt = "Неправильный stub промпт"

    # Create temporary prompt file in system_prompts
    prompts_dir = tmp_path / "system_prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "test_aurika_prompt.txt"
    prompt_file.write_text("Ты — Аурика, трансформационный практик.", encoding="utf-8")

    session = _MaxAISession(user, config)
    captured_layout = []

    async def _mock_dispatch(cfg, layout, **kwargs):
        captured_layout.append(layout)
        return "Ответ Аурики"

    monkeypatch.setattr(max_ai, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "_dispatch_provider", _mock_dispatch)

    result = await max_ai.get_ai_response(user.id, "привет")

    assert result == "Ответ Аурики"
    assert len(captured_layout) == 1
    assert "Ты — Аурика, трансформационный практик." in captured_layout[0].stable_system_prompt
    assert "Неправильный stub промпт" not in captured_layout[0].stable_system_prompt


@pytest.mark.asyncio
async def test_max_chat_response_with_history_topic_relationship(monkeypatch):
    user = _max_user()
    session = _MaxAISession(user, _max_config())
    topic = SimpleNamespace(id=10, name="Энергетика")
    msg1 = SimpleNamespace(
        id=1,
        role="user",
        content="прошлый вопрос",
        topic_id=10,
        topic=topic,
        timestamp=datetime(2026, 8, 28, 10, 0),
    )
    msg2 = SimpleNamespace(
        id=2,
        role="assistant",
        content="прошлый ответ",
        topic_id=10,
        topic=topic,
        timestamp=datetime(2026, 8, 28, 10, 1),
    )

    async def _execute_history(stmt):
        sql = str(stmt)
        if "messages" in sql.lower():
            return _Result(rows=[msg1, msg2])
        return _Result(rows=[])

    session.execute = _execute_history
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(return_value="Ответ с историей"))

    result = await max_ai.get_ai_response(user.id, "новый вопрос")

    assert result == "Ответ с историей"


@pytest.mark.asyncio
async def test_max_openai_request_payload_captured_and_sanitized(monkeypatch):
    _MockOpenAIClient.calls.clear()
    monkeypatch.setattr(max_ai, "AsyncOpenAI", _MockOpenAIClient)
    user = _max_user()
    config = _max_config()
    config.provider = "OpenAI"
    config.openai_api_key = "sk-live-openai-secret-token-12345"
    config.openai_model = "gpt-5.6-terra"
    session = _MaxAISession(user, config)

    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))

    result = await max_ai.get_ai_response(user.id, "Как дела?")

    assert result == "Mocked OpenAI response"
    assert len(session.added) == 1
    log_entry = session.added[0]
    assert log_entry.provider == "OpenAI"
    assert log_entry.model == "gpt-5.6-terra"

    # Assert payload exists first (strong assertion)
    assert log_entry.request_payload
    parsed = json.loads(log_entry.request_payload)
    assert parsed["provider"] == "OpenAI"
    assert parsed["endpoint"] == "https://api.openai.com/v1/chat/completions"
    assert parsed["payload"]["model"] == "gpt-5.6-terra"
    assert parsed["payload"]["max_completion_tokens"] == 4096
    messages = parsed["payload"]["messages"]
    assert any(m["role"] == "user" and m["content"] == "Как дела?" for m in messages)

    # Sanitization
    assert "sk-live-openai-secret-token-12345" not in log_entry.request_payload
    assert "Bearer" not in log_entry.request_payload
    assert "Authorization" not in log_entry.request_payload


@pytest.mark.asyncio
async def test_max_deepseek_topic_request_payload_complete_and_sanitized(monkeypatch):
    _MockOpenAIClient.calls.clear()
    monkeypatch.setattr(max_ai, "AsyncOpenAI", _MockOpenAIClient)

    topic = SimpleNamespace(
        id=15,
        name="Психосоматика",
        system_prompt="Специальный промпт темы",
        instruction=None,
        use_common_instruction=True,
        knowledge_base_files=[],
    )
    user = _max_user(topic_id=15, topic=topic)
    config = _max_config()
    config.provider = "Deepseek"
    config.deepseek_api_key = "sk-deepseek-super-secret-999"
    config.deepseek_model = "deepseek-chat"
    config.system_prompt = "Базовый системный промпт"
    session = _MaxAISession(user, config)

    history_msg1 = SimpleNamespace(
        id=1, role="user", content="болит голова", topic_id=15, topic=topic, timestamp=datetime(2026, 9, 1, 10, 0)
    )
    history_msg2 = SimpleNamespace(
        id=2, role="assistant", content="расскажите подробнее", topic_id=15, topic=topic, timestamp=datetime(2026, 9, 1, 10, 1)
    )

    async def _execute_mock(stmt):
        sql = str(stmt)
        if "messages" in sql.lower():
            return _Result(rows=[history_msg1, history_msg2])
        return _Result(rows=[])

    session.execute = _execute_mock
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value="СЛУЖЕБНЫЙ_КОНТЕКСТ_ТЕСТ"))

    result = await max_ai.get_ai_response(user.id, "болит висок")

    assert result == "Mocked OpenAI response"
    assert len(session.added) == 1
    log_entry = session.added[0]
    assert log_entry.provider == "Deepseek"
    assert log_entry.model == "deepseek-v4-flash"
    assert log_entry.context_kind == "topic"
    assert log_entry.topic_id == 15

    # Assert payload exists first
    assert log_entry.request_payload
    parsed = json.loads(log_entry.request_payload)
    assert parsed["provider"] == "Deepseek"
    assert parsed["endpoint"] == "https://api.deepseek.com/chat/completions"
    assert parsed["payload"]["model"] == "deepseek-v4-flash"

    messages = parsed["payload"]["messages"]
    # 1. Effective system/topic prompt
    assert any(m["role"] == "system" and "Специальный промпт темы" in m["content"] for m in messages)
    # 2. Topic/runtime context
    assert any(m["role"] == "system" and "СЛУЖЕБНЫЙ_КОНТЕКСТ_ТЕСТ" in m["content"] for m in messages)
    assert any(m["role"] == "system" and "ДАННЫЕ КЛИЕНТА:" in m["content"] for m in messages)
    # 3. Conversation history
    assert any(m["role"] == "user" and m["content"] == "болит голова" for m in messages)
    assert any(m["role"] == "assistant" and m["content"] == "расскажите подробнее" for m in messages)
    # 4. Current user prompt
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "болит висок"

    # Sanitization
    assert "sk-deepseek-super-secret-999" not in log_entry.request_payload


@pytest.mark.asyncio
async def test_max_claude_request_payload_shape_and_sanitized(monkeypatch):
    _MockAnthropicClient.calls.clear()
    monkeypatch.setattr(max_ai.anthropic, "AsyncAnthropic", _MockAnthropicClient)
    user = _max_user()
    config = _max_config()
    config.provider = "Claude"
    config.claude_api_key = "sk-ant-api03-claude-secret-key-xyz"
    config.claude_model = "claude-sonnet-5"
    config.system_prompt = "Ты мудрый терапевт"
    session = _MaxAISession(user, config)

    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))

    result = await max_ai.get_ai_response(user.id, "Вопрос к Клоду")

    assert result == "Mocked Claude response"
    assert len(session.added) == 1
    log_entry = session.added[0]
    assert log_entry.provider == "Claude"
    assert log_entry.model == "claude-sonnet-5"

    # Assert payload exists first
    assert log_entry.request_payload
    parsed = json.loads(log_entry.request_payload)
    assert parsed["provider"] == "Claude"
    assert parsed["endpoint"] == "https://api.anthropic.com/v1/messages"
    assert parsed["payload"]["model"] == "claude-sonnet-5"
    assert parsed["payload"]["max_tokens"] == 4096
    assert isinstance(parsed["payload"]["system"], list)
    assert any("Ты мудрый терапевт" in block.get("text", "") for block in parsed["payload"]["system"])
    assert parsed["payload"]["messages"][-1] == {"role": "user", "content": "Вопрос к Клоду"}

    # Sanitization
    assert "sk-ant-api03-claude-secret-key-xyz" not in log_entry.request_payload


@pytest.mark.asyncio
async def test_max_gemini_request_payload_endpoint_and_sanitized(monkeypatch):
    _MockHttpxClient.calls.clear()
    _MockHttpxClient.response_json = None
    _MockHttpxClient.status_code = 200
    monkeypatch.setattr(max_ai.httpx, "AsyncClient", _MockHttpxClient)
    user = _max_user()
    config = _max_config()
    config.provider = "Gemini"
    config.gemini_api_key = "AIzaSyD-gemini-confidential-key-789"
    config.gemini_model = "gemini-3.7-flash"
    session = _MaxAISession(user, config)

    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))

    result = await max_ai.get_ai_response(user.id, "Вопрос к Gemini")

    assert result == "Mocked Gemini response"
    assert len(session.added) == 1
    log_entry = session.added[0]
    assert log_entry.provider == "Gemini"
    assert log_entry.model == "gemini-3.7-flash"

    # Assert payload exists first
    assert log_entry.request_payload
    parsed = json.loads(log_entry.request_payload)
    assert parsed["provider"] == "Gemini"
    assert parsed["endpoint"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent"
    # Prove no ?key= query parameter in endpoint and whole request_payload
    assert "?key=" not in parsed["endpoint"]
    assert "&key=" not in parsed["endpoint"]
    assert "?key=" not in log_entry.request_payload
    assert "AIzaSyD-gemini-confidential-key-789" not in log_entry.request_payload

    # Prove actual payload structure
    assert "contents" in parsed["payload"]
    assert "systemInstruction" in parsed["payload"]
    assert "generationConfig" in parsed["payload"]
    contents = parsed["payload"]["contents"]
    assert any(part.get("text") == "Вопрос к Gemini" for c in contents for part in c.get("parts", []))


@pytest.mark.asyncio
async def test_max_kie_request_payload_matches_outbound_and_sanitized(monkeypatch):
    _MockHttpxClient.calls.clear()
    _MockHttpxClient.response_json = {
        "code": 200,
        "data": {
            "choices": [{
                "message": {"content": "Mocked KIE response"}
            }]
        }
    }
    _MockHttpxClient.status_code = 200
    monkeypatch.setattr(max_ai.httpx, "AsyncClient", _MockHttpxClient)
    user = _max_user()
    config = _max_config()
    config.provider = "KIE"
    config.kie_api_key = "kie-secret-api-key-55555"
    config.kie_model = "gemini-3-flash"
    session = _MaxAISession(user, config)

    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))

    result = await max_ai.get_ai_response(user.id, "Вопрос к KIE")

    assert result == "Mocked KIE response"
    assert len(session.added) == 1
    log_entry = session.added[0]
    assert log_entry.provider == "KIE"
    assert log_entry.model == "gemini-3-flash"

    # Assert payload exists first
    assert log_entry.request_payload
    parsed = json.loads(log_entry.request_payload)
    assert parsed["provider"] == "KIE"
    assert "kie-secret-api-key-55555" not in log_entry.request_payload
    assert "Bearer" not in log_entry.request_payload
    assert "Authorization" not in log_entry.request_payload

    # Assert matches outbound request passed to HTTP client
    assert len(_MockHttpxClient.calls) == 1
    http_call = _MockHttpxClient.calls[0]
    assert parsed["endpoint"] == http_call["url"]
    assert parsed["payload"] == http_call["json"]


@pytest.mark.asyncio
async def test_max_fallback_orchestration_persists_fallback_payload_only(monkeypatch):
    _MockHttpxClient.calls.clear()
    _MockOpenAIClient.calls.clear()

    # Primary Gemini will fail
    class _FailingGeminiClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            raise Exception("Gemini service unavailable 503")

    monkeypatch.setattr(max_ai.httpx, "AsyncClient", _FailingGeminiClient)
    monkeypatch.setattr(max_ai, "AsyncOpenAI", _MockOpenAIClient)

    user = _max_user()
    config = _max_config()
    config.provider = "Gemini"
    config.gemini_api_key = "primary-gemini-secret-key-111"
    config.gemini_model = "gemini-3.7-flash"
    config.allow_fallback = True
    config.fallback_provider = "OpenAI"
    config.fallback_model = "gpt-5.6-terra"
    config.openai_api_key = "fallback-openai-secret-key-222"
    session = _MaxAISession(user, config)

    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))

    result = await max_ai.get_ai_response(user.id, "Вопрос для фолбэка")

    assert result == "Mocked OpenAI response"
    assert len(session.added) == 1
    log_entry = session.added[0]

    # 1. Final provider and model correspond to fallback
    assert log_entry.provider == "OpenAI"
    assert log_entry.model == "gpt-5.6-terra"

    # 2. Assert payload exists first
    assert log_entry.request_payload
    parsed = json.loads(log_entry.request_payload)

    # 3. Final request_payload is fallback request, not primary request
    assert parsed["provider"] == "OpenAI"
    assert parsed["endpoint"] == "https://api.openai.com/v1/chat/completions"
    assert parsed["payload"]["model"] == "gpt-5.6-terra"
    assert "gemini" not in parsed["endpoint"].lower()
    assert "gemini" not in parsed["provider"].lower()

    # 4. Sanitization
    assert "primary-gemini-secret-key-111" not in log_entry.request_payload
    assert "fallback-openai-secret-key-222" not in log_entry.request_payload


@pytest.mark.asyncio
async def test_max_deepseek_fallback_legacy_alias_persists_normalized_model_and_payload(monkeypatch):
    _MockHttpxClient.calls.clear()
    _MockOpenAIClient.calls.clear()

    # Primary Gemini will fail
    class _FailingGeminiClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            raise Exception("Gemini service unavailable 503")

    monkeypatch.setattr(max_ai.httpx, "AsyncClient", _FailingGeminiClient)
    monkeypatch.setattr(max_ai, "AsyncOpenAI", _MockOpenAIClient)

    user = _max_user()
    config = _max_config()
    config.provider = "Gemini"
    config.gemini_api_key = "primary-gemini-secret-key-111"
    config.gemini_model = "gemini-3.7-flash"
    config.allow_fallback = True
    config.fallback_provider = "Deepseek"
    config.fallback_model = "deepseek-chat"  # legacy alias
    config.deepseek_api_key = "fallback-deepseek-secret-key-333"
    session = _MaxAISession(user, config)

    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))

    result = await max_ai.get_ai_response(user.id, "Вопрос для DeepSeek фолбэка")

    assert result == "Mocked OpenAI response"
    assert len(session.added) == 1
    log_entry = session.added[0]

    # 1. Final provider and model correspond to DeepSeek fallback
    assert log_entry.provider == "Deepseek"
    assert log_entry.model == "deepseek-v4-flash"

    # 2. Assert payload exists first
    assert log_entry.request_payload
    parsed = json.loads(log_entry.request_payload)

    # 3. Final request_payload is DeepSeek fallback request with normalized model
    assert parsed["provider"] == "Deepseek"
    assert parsed["endpoint"] == "https://api.deepseek.com/chat/completions"
    assert parsed["payload"]["model"] == log_entry.model
    assert parsed["payload"]["model"] == "deepseek-v4-flash"
    assert "gemini" not in parsed["endpoint"].lower()
    assert "gemini" not in parsed["provider"].lower()

    # 4. Sanitization
    assert "primary-gemini-secret-key-111" not in log_entry.request_payload
    assert "fallback-deepseek-secret-key-333" not in log_entry.request_payload


@pytest.mark.asyncio
async def test_max_ai_log_detail_and_export_with_populated_request_payload(monkeypatch):
    user = _max_user()
    sample_payload = {
        "provider": "OpenAI",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "payload": {
            "model": "gpt-5.6-terra",
            "messages": [
                {"role": "system", "content": "Системный промпт"},
                {"role": "user", "content": "Привет, бот"},
            ],
        },
    }
    payload_str = json.dumps(sample_payload, ensure_ascii=False, indent=2)
    log_entry = AILog(
        id=25,
        user_id=user.id,
        provider="OpenAI",
        model="gpt-5.6-terra",
        prompt_summary="Привет, бот",
        request_payload=payload_str,
        raw_response="Ответ модели",
        clean_text="Ответ модели",
        created_at=datetime(2026, 9, 6, 12, 0),
        platform="max",
        context_kind="main",
    )
    session = _DetailSession(log_entry, user)
    event = SimpleNamespace(answer=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _SessionContext(session))

    await handlers.show_ai_log_detail(event, log_entry.id)

    text_value = event.answer.await_args.args[0]
    assert "Полный payload запроса (превью):" in text_value
    assert "gpt-5.6-terra" in text_value
    assert "Привет, бот" in text_value

    file_content = handlers._build_ai_log_file_content(log_entry)
    assert "[1] FULL REQUEST PAYLOAD:" in file_content
    assert payload_str in file_content
    assert "User ID: 100018792559" in file_content


def test_ai_log_unpopulated_request_payload_displays_unrecorded():
    log_entry = AILog(
        id=26,
        user_id=123,
        provider="OpenAI",
        model="gpt-5.6-terra",
        prompt_summary="только саммари без payload",
        request_payload=None,
        raw_response="Ответ",
        clean_text="Ответ",
        created_at=datetime(2026, 9, 6, 12, 0),
        platform="max",
        context_kind="main",
    )
    file_content = handlers._build_ai_log_file_content(log_entry)
    assert "[1] FULL REQUEST PAYLOAD:\n----------------------------------------\nне зафиксирован\n" in file_content
    assert "только саммари без payload" not in file_content.split("[2] RAW RESPONSE")[0]

