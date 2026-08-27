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
    user = _max_user()
    session = _MaxAISession(user, _max_config())
    monkeypatch.setattr(max_ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(max_ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    monkeypatch.setattr(max_ai, "_dispatch_provider", AsyncMock(return_value="MAX answer"))

    result = await max_ai.get_ai_response(user.id, "question")

    assert result == "MAX answer"
    assert len(session.added) == 1
    log_entry = session.added[0]
    assert isinstance(log_entry, AILog)
    assert log_entry.user_id == user.id
    assert log_entry.platform == "max"
    assert log_entry.context_kind == "main"
    assert log_entry.topic_id is None
    assert log_entry.topic_name_snapshot is None
    assert "max-secret" not in (log_entry.request_payload or "")
    assert session.commits == 1


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
