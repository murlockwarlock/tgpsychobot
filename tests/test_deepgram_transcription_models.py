import asyncio
from datetime import datetime, timezone
import html
import io
import os
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "123456:TEST_BOT_TOKEN")

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.enums import ChatType
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, File, InlineKeyboardMarkup, Message, Update, User as TgUser, Voice
import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from database import AIConfig, Base, User
import handlers
import keyboards as kb
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.api import MaxApiClient
import max_messenger_bot.keyboards as max_kb
from max_messenger_bot.services.common import run_ai_dialogue_with_voice
from max_messenger_bot.storage import StorageBase
import provider_adapters
from provider_adapters import ProviderAdapterError, call_deepgram
import provider_models
from provider_models import (
    DEEPGRAM_DEFAULT_MODEL,
    DEEPGRAM_TRANSCRIPTION_MODELS,
    PROVIDER_DEEPGRAM,
    ModelUnavailableError,
    build_telegram_model_callback_data,
    get_capability_providers,
    get_default_model,
    get_selectable_models,
    resolve_telegram_model_callback,
    validate_model_selection,
)


# ---------------------------------------------------------------------------
# Validating Session for aiogram (Pydantic validation of outgoing methods)
# ---------------------------------------------------------------------------

class ValidatingTelegramSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        type(method).model_validate(method.model_dump())
        if isinstance(method, SendMessage):
            return Message(
                message_id=len(self.calls),
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id, type=ChatType.PRIVATE),
                text=method.text or "",
            ).as_(bot)
        if isinstance(method, EditMessageText):
            return Message(
                message_id=method.message_id or 10,
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id or 999, type=ChatType.PRIVATE),
                text=method.text or "",
            ).as_(bot)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


def _last_markup(calls):
    return next(c.reply_markup for c in reversed(calls) if hasattr(c, "reply_markup") and c.reply_markup is not None)


_real_async_client = httpx.AsyncClient


def mock_httpx_async_client(transport):
    def _client_factory(*args, **kwargs):
        kw = dict(kwargs)
        kw["transport"] = transport
        return _real_async_client(*args, **kw)
    return _client_factory


# ---------------------------------------------------------------------------
# Test Fixtures & In-Memory Database
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(StorageBase.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def init_ai_config(session_factory):
    async with session_factory() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(
                id=1,
                provider="Deepseek",
                deepseek_model="deepseek-flash",
                transcription_provider=PROVIDER_DEEPGRAM,
                deepgram_model="nova-3",
                deepgram_api_key="test-deepgram-api-key",
                max_voice_duration_sec=180,
            )
            session.add(cfg)
            await session.commit()
    return cfg


# ---------------------------------------------------------------------------
# 1. Model Registry Contract
# ---------------------------------------------------------------------------

def test_deepgram_catalog_and_default():
    """Verify Deepgram transcription catalog contains exact models and default is nova-3."""
    assert DEEPGRAM_TRANSCRIPTION_MODELS == ("nova-3", "nova-2")
    assert DEEPGRAM_DEFAULT_MODEL == "nova-3"
    assert get_selectable_models(PROVIDER_DEEPGRAM, channel="transcription") == ("nova-3", "nova-2")
    assert get_default_model(PROVIDER_DEEPGRAM, channel="transcription") == "nova-3"
    assert PROVIDER_DEEPGRAM in get_capability_providers("transcription")


def test_deepgram_compact_model_callbacks():
    """Verify both nova-3 and nova-2 produce valid <=64 byte compact callbacks that resolve."""
    for model_id in ("nova-3", "nova-2"):
        cb = build_telegram_model_callback_data(PROVIDER_DEEPGRAM, "transcription", model_id)
        assert len(cb.encode("utf-8")) <= 64
        resolved = resolve_telegram_model_callback(cb)
        assert resolved == (PROVIDER_DEEPGRAM, "transcription", model_id)


def test_invalid_models_rejected_before_persistence_and_http():
    """Verify unknown, flux, or specialty models cannot be validated or resolved."""
    invalid_candidates = (
        "flux-general-multi",
        "flux-general-en",
        "nova-2-medical",
        "nova-3-medical",
        "nova-2-general",
        "nova-3-general",
        "random-string",
    )
    for invalid in invalid_candidates:
        with pytest.raises(ModelUnavailableError):
            validate_model_selection(PROVIDER_DEEPGRAM, invalid, channel="transcription")

        with pytest.raises(ProviderAdapterError) as exc_info:
            asyncio.run(call_deepgram("key", b"fake", "audio.ogg", model=invalid))
        assert exc_info.value.classification == "invalid_model"


# ---------------------------------------------------------------------------
# 2. Wire Requests Contract (Nova-3 vs Nova-2)
# ---------------------------------------------------------------------------

def test_call_deepgram_nova_3_wire():
    """Verify nova-3 outbound request has model=nova-3, smart_format=true, language=multi."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["auth"] = request.headers.get("Authorization")
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = request.read()
        return httpx.Response(200, json={"results": {"channels": [{"alternatives": [{"transcript": "Nova 3 output"}]}]}})

    transport = httpx.MockTransport(handler)
    with patch("httpx.AsyncClient", mock_httpx_async_client(transport)):
        req_cap = {}
        res = asyncio.run(call_deepgram("secret-key", b"test-audio-bytes", "voice.ogg", model="nova-3", request_capture=req_cap))

    assert res == "Nova 3 output"
    assert captured["method"] == "POST"
    assert captured["auth"] == "Token secret-key"
    assert captured["content_type"] == "audio/ogg"
    assert captured["body"] == b"test-audio-bytes"

    parsed = urlparse(captured["url"])
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.deepgram.com"
    assert parsed.path == "/v1/listen"
    qs = parse_qs(parsed.query)
    assert qs["model"] == ["nova-3"]
    assert qs["smart_format"] == ["true"]
    assert qs["language"] == ["multi"]
    assert "detect_language" not in qs
    assert req_cap["payload"] == {"model": "nova-3", "language": "multi", "smart_format": True}


def test_call_deepgram_nova_2_wire():
    """Verify nova-2 outbound request has model=nova-2, smart_format=true, detect_language=true (omits language=multi)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["auth"] = request.headers.get("Authorization")
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = request.read()
        return httpx.Response(200, json={"results": {"channels": [{"alternatives": [{"transcript": "Nova 2 output"}]}]}})

    transport = httpx.MockTransport(handler)
    with patch("httpx.AsyncClient", mock_httpx_async_client(transport)):
        req_cap = {}
        res = asyncio.run(call_deepgram("secret-key", b"test-audio-bytes", "voice.ogg", model="nova-2", request_capture=req_cap))

    assert res == "Nova 2 output"
    assert captured["method"] == "POST"
    assert captured["auth"] == "Token secret-key"
    assert captured["content_type"] == "audio/ogg"
    assert captured["body"] == b"test-audio-bytes"

    parsed = urlparse(captured["url"])
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.deepgram.com"
    assert parsed.path == "/v1/listen"
    qs = parse_qs(parsed.query)
    assert qs["model"] == ["nova-2"]
    assert qs["smart_format"] == ["true"]
    assert qs["detect_language"] == ["true"]
    assert "language" not in qs, "Nova-2 must NOT send language=multi"
    assert req_cap["payload"] == {"model": "nova-2", "detect_language": True, "smart_format": True}


def test_call_deepgram_empty_response():
    """Verify empty transcript raises empty_response classification."""
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"results": {"channels": [{"alternatives": [{"transcript": ""}]}]}}))
    with patch("httpx.AsyncClient", mock_httpx_async_client(transport)):
        with pytest.raises(ProviderAdapterError) as exc_info:
            asyncio.run(call_deepgram("secret", b"audio", "voice.ogg", model="nova-2"))
        assert exc_info.value.classification == "empty_response"


def test_call_deepgram_error_classification_and_retries():
    """Verify HTTP errors map to canonical categories and 5xx/429 retries are bounded to 2 attempts."""
    # Auth 401
    t_auth = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "unauthorized"}))
    with patch("httpx.AsyncClient", mock_httpx_async_client(t_auth)):
        with pytest.raises(ProviderAdapterError) as exc:
            asyncio.run(call_deepgram("bad-key", b"audio", "voice.ogg"))
        assert exc.value.classification == "auth"

    # Server error 500 retried
    attempts = 0

    def retry_handler(req: httpx.Request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": "internal error"})

    t_retry = httpx.MockTransport(retry_handler)
    with patch("httpx.AsyncClient", mock_httpx_async_client(t_retry)):
        with pytest.raises(ProviderAdapterError) as exc:
            asyncio.run(call_deepgram("key", b"audio", "voice.ogg"))
        assert exc.value.classification == "server_error"
        assert attempts == 2


# ---------------------------------------------------------------------------
# 3. Telegram Full Admin -> Runtime -> Admin Journey (A -> Я -> A)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_full_journey_admin_to_runtime_to_admin(session_factory, monkeypatch, init_ai_config):
    """Real Telegram E2E journey without direct DB mutation:
    Admin Audio -> visible 'Выбрать модель' -> Dispatcher -> select visible nova-2 ->
    DB persistence -> reopen shows ✅ -> visible Back -> parent Audio screen ->
    ordinary user sends Voice update -> file download boundary -> production transcription runtime ->
    Deepgram /v1/listen MockTransport (nova-2, detect_language=true, no language=multi) ->
    outgoing aiogram Pydantic validation -> user transcript result ->
    reopen Admin picker again -> nova-2 still shows ✅.
    """
    # 1. Pre-seed DB: transcription_provider is Deepgram, model is DEFAULT nova-3 (NOT nova-2)
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.transcription_provider = PROVIDER_DEEPGRAM
        cfg.deepgram_model = "nova-3"
        cfg.deepgram_api_key = "test-live-key"

        # Pre-seed client user
        client_user_db = await s.get(User, 555)
        if not client_user_db:
            client_user_db = User(id=555, first_name="ClientUser", accepted_disclaimer=True)
            s.add(client_user_db)
        else:
            client_user_db.accepted_disclaimer = True
        await s.commit()

    async def _is_admin(uid, *args, **kwargs):
        return uid == 999

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr("ai_integration.async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", _is_admin)

    # 2. Intercept Deepgram HTTP request
    captured_requests = []

    def deepgram_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "content": request.read(),
        })
        return httpx.Response(200, json={
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "Мне сегодня очень тревожно, помогите разобраться."
                    }]
                }]
            }
        })

    mock_transport = httpx.MockTransport(deepgram_transport_handler)
    monkeypatch.setattr(httpx, "AsyncClient", mock_httpx_async_client(mock_transport))

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    # Mock audio file download boundary in bot
    bot.get_file = AsyncMock(return_value=File(file_id="voice_123", file_unique_id="u_123", file_path="voice.ogg"))
    bot.download_file = AsyncMock(return_value=io.BytesIO(b"ogg-audio-binary-data"))

    # Stop after voice transcription delivery before downstream dialogue AI
    monkeypatch.setattr(handlers, "process_user_prompt", AsyncMock())

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    admin_user = TgUser(id=999, is_bot=False, first_name="Admin")
    admin_chat = Chat(id=999, type=ChatType.PRIVATE)

    # Step 1: Admin opens audio settings screen
    update_1 = Update(
        update_id=1,
        callback_query=CallbackQuery(
            id="cb1",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Audio"),
            data="admin_ai_audio",
        ),
    )
    await dp.feed_update(bot, update_1)
    audio_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Step 2: Admin finds visible "Выбрать модель" button and clicks it through Dispatcher
    model_btn = next(b for row in audio_kb.inline_keyboard for b in row if "Выбрать модель" in b.text)
    assert model_btn.callback_data == "admin_audio_model"

    update_2 = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb2",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Pick Model"),
            data=model_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_2)
    picker_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Check visible model buttons: default nova-3 has checkmark, nova-2 does not
    buttons = [b for row in picker_kb.inline_keyboard for b in row]
    nova3_btn = next(b for b in buttons if "nova-3" in b.text)
    nova2_btn = next(b for b in buttons if "nova-2" in b.text)
    assert "✅" in nova3_btn.text
    assert "✅" not in nova2_btn.text

    # Step 3: Admin selects nova-2 via visible button callback through Dispatcher
    update_3 = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb3",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Selecting"),
            data=nova2_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_3)

    # Verify DB persistence directly from handler execution (NOT manually written)
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-2"

    # Step 4: Reopen picker to verify checkmark moved to nova-2
    update_4 = Update(
        update_id=4,
        callback_query=CallbackQuery(
            id="cb4",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Reopen"),
            data="admin_audio_model",
        ),
    )
    await dp.feed_update(bot, update_4)
    reopened_picker: InlineKeyboardMarkup = _last_markup(session.calls)

    re_buttons = [b for row in reopened_picker.inline_keyboard for b in row]
    re_nova3 = next(b for b in re_buttons if "nova-3" in b.text)
    re_nova2 = next(b for b in re_buttons if "nova-2" in b.text)
    assert "✅" in re_nova2.text
    assert "✅" not in re_nova3.text

    # Step 5: Press the actual visible Back button through Dispatcher
    back_btn = next(b for b in re_buttons if "Назад" in b.text)
    assert back_btn.callback_data == "admin_ai_audio"

    update_5 = Update(
        update_id=5,
        callback_query=CallbackQuery(
            id="cb5",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Back"),
            data=back_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_5)

    # Verify actual resulting screen is immediate parent (Audio screen) with visible controls
    parent_call = next(c for c in reversed(session.calls) if isinstance(c, EditMessageText))
    assert "🎙 <b>Аудио</b>" in parent_call.text
    assert "Модель: <code>nova-2</code>" in parent_call.text
    parent_audio_kb: InlineKeyboardMarkup = parent_call.reply_markup
    parent_btn_texts = [b.text for row in parent_audio_kb.inline_keyboard for b in row]
    assert any("Выбрать провайдера" in t for t in parent_btn_texts)
    assert any("Выбрать модель" in t for t in parent_btn_texts)

    # Step 6: Ordinary user sends real voice message through Dispatcher
    client_user = TgUser(id=555, is_bot=False, first_name="ClientUser")
    client_chat = Chat(id=555, type=ChatType.PRIVATE)
    voice_msg = Message(
        message_id=42,
        date=datetime.now(timezone.utc),
        chat=client_chat,
        from_user=client_user,
        voice=Voice(file_id="voice_123", file_unique_id="u_123", duration=10),
    )
    voice_update = Update(update_id=101, message=voice_msg)
    await dp.feed_update(bot, voice_update)

    # Step 7: Verify selected model was sent over Deepgram HTTP wire
    assert len(captured_requests) == 1
    req = captured_requests[0]
    parsed = urlparse(req["url"])
    assert parsed.path == "/v1/listen"
    qs = parse_qs(parsed.query)
    assert qs["model"] == ["nova-2"]
    assert qs["detect_language"] == ["true"]
    assert "language" not in qs
    assert req["headers"]["authorization"] == "Token test-live-key"
    assert req["content"] == b"ogg-audio-binary-data"

    # Step 8: Verify user received the transcript via aiogram validated method
    edit_calls = [c for c in session.calls if isinstance(c, EditMessageText) and c.chat_id == 555]
    assert edit_calls, "Thinking message must be edited with transcript for client"
    assert "<i>Мне сегодня очень тревожно, помогите разобраться.</i>" in edit_calls[0].text

    # Step 9: Reopen Admin picker again -> nova-2 still shows ✅
    update_final = Update(
        update_id=102,
        callback_query=CallbackQuery(
            id="cb_final",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Reopen Again"),
            data="admin_audio_model",
        ),
    )
    await dp.feed_update(bot, update_final)
    final_picker: InlineKeyboardMarkup = _last_markup(session.calls)
    final_buttons = [b for row in final_picker.inline_keyboard for b in row]
    final_nova2 = next(b for b in final_buttons if "nova-2" in b.text)
    final_nova3 = next(b for b in final_buttons if "nova-3" in b.text)
    assert "✅" in final_nova2.text
    assert "✅" not in final_nova3.text

    # Final DB check confirms no settings loss
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-2"


# ---------------------------------------------------------------------------
# 4. MAX Full Admin -> Runtime -> Admin Journey (A -> Я -> A)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_full_journey_admin_to_runtime_to_admin(session_factory, monkeypatch, init_ai_config):
    """Real MAX E2E journey through MaxBotApplication.handle_update():
    Admin opens Deepgram models -> visible nova-2 button -> handle_update(callback) ->
    DB persistence -> reopen shows ✅ -> visible Back button -> handle_update(callback) ->
    parent screen verified -> ordinary user message_created audio update ->
    production media branch -> _handle_voice -> download_attachment boundary ->
    transcribe_audio -> call_deepgram -> MockTransport captures /v1/listen (nova-2, detect_language=true) ->
    transcript delivered to user via MAX API -> reopen Admin picker again -> nova-2 still shows ✅.
    """
    from max_messenger_bot.models import MAX_ID_OFFSET
    import max_messenger_bot.services.admin_ai as max_admin_ai
    import max_messenger_bot.ai as max_ai
    import max_messenger_bot.services.common as max_common
    import followups

    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr(max_admin_ai, "async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(followups, "async_session_maker", session_factory)

    # 1. Pre-seed DB: transcription_provider is Deepgram, model is DEFAULT nova-3 (NOT nova-2)
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.transcription_provider = PROVIDER_DEEPGRAM
        cfg.deepgram_model = "nova-3"
        cfg.deepgram_api_key = "test-live-key"

        client_internal_uid = 777 + MAX_ID_OFFSET
        client_max_user = await s.get(User, client_internal_uid)
        if not client_max_user:
            client_max_user = User(
                id=client_internal_uid,
                name="MaxClient",
                first_name="MaxClient",
                accepted_disclaimer=True,
            )
            s.add(client_max_user)
        else:
            client_max_user.name = "MaxClient"
            client_max_user.accepted_disclaimer = True
        await s.commit()

    async def _is_admin(uid, *args, **kwargs):
        return uid in (888, 888 + MAX_ID_OFFSET)

    monkeypatch.setattr(max_common, "is_admin", _is_admin)
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))
    monkeypatch.setattr(max_common, "maybe_require_disclaimer", AsyncMock(return_value=False))
    # Stub dialogue AI generation after transcript delivery
    monkeypatch.setattr("max_messenger_bot.services.common.run_ai_dialogue", AsyncMock())

    # 2. Intercept MAX API client calls
    sent_messages = []

    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        sent_messages.append({"method": method, "path": path, "params": params or {}, "body": json_data or {}})
        if path == "/messages":
            return {"message": {"body": {"mid": f"mid_{len(sent_messages)}"}}}
        return {}

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = fake_request
    client.download_attachment = AsyncMock(return_value=b"ogg-binary-audio-for-max")
    app = MaxBotApplication(client=client)

    # 3. Intercept Deepgram HTTP request
    captured_max_http = []

    def max_deepgram_handler(req: httpx.Request) -> httpx.Response:
        captured_max_http.append({
            "url": str(req.url),
            "headers": dict(req.headers),
            "content": req.read(),
        })
        return httpx.Response(200, json={
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "Голос в MAX успешно распознан через nova-2"
                    }]
                }]
            }
        })

    t_max = httpx.MockTransport(max_deepgram_handler)
    monkeypatch.setattr(httpx, "AsyncClient", mock_httpx_async_client(t_max))

    up_id = 300

    async def press_admin_cb(payload):
        nonlocal up_id
        up_id += 1
        await app.handle_update({
            "update_type": "message_callback",
            "update_id": up_id,
            "callback": {
                "callback_id": f"cb_{up_id}",
                "payload": payload,
                "sender": {"user_id": 888, "first_name": "Admin"},
            },
            "message": {
                "mid": f"msg_{up_id}",
                "recipient": {"chat_id": 888},
                "body": {"attachments": []},
            },
        })
        messages = [m for m in sent_messages if m["path"] == "/messages"]
        return messages[-1]

    # Step 1: Admin opens Deepgram model choices in MAX
    sent_messages.clear()
    screen1 = await press_admin_cb(f"admin_ai_provider_models_{PROVIDER_DEEPGRAM}")
    buttons1 = [b for row in screen1["body"]["attachments"][0]["payload"]["buttons"] for b in row]
    btn_n3 = next(b for b in buttons1 if "nova-3" in b["text"])
    btn_n2 = next(b for b in buttons1 if "nova-2" in b["text"])
    assert "✅" in btn_n3["text"]
    assert "✅" not in btn_n2["text"]

    # Step 2: Admin selects nova-2 via visible button payload
    sent_messages.clear()
    await press_admin_cb(btn_n2["payload"])

    # Verify DB persistence directly from handler
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-2"

    # Step 3: Reopen model choices in MAX -> verify ✅ on nova-2
    sent_messages.clear()
    screen2 = await press_admin_cb(f"admin_ai_provider_models_{PROVIDER_DEEPGRAM}")
    buttons2 = [b for row in screen2["body"]["attachments"][0]["payload"]["buttons"] for b in row]
    re_n3 = next(b for b in buttons2 if "nova-3" in b["text"])
    re_n2 = next(b for b in buttons2 if "nova-2" in b["text"])
    assert "✅" in re_n2["text"]
    assert "✅" not in re_n3["text"]

    # Step 4: Find actual visible Back button and press it through handle_update
    back_btn = next(b for b in buttons2 if "Назад" in b["text"])
    assert back_btn["payload"] == f"admin_ai_models_{PROVIDER_DEEPGRAM}"

    sent_messages.clear()
    parent_screen = await press_admin_cb(back_btn["payload"])

    # Verify parent screen shows Deepgram provider settings with nova-2
    parent_text = parent_screen["body"].get("text", "")
    assert "<b>Deepgram</b>" in parent_text
    assert "Модель: <code>nova-2</code>" in parent_text
    parent_buttons = [b for row in parent_screen["body"]["attachments"][0]["payload"]["buttons"] for b in row]
    assert any("Выбрать модель" in b["text"] for b in parent_buttons)

    # Step 5: Ordinary MAX user sends realistic message_created audio update
    sent_messages.clear()
    up_id += 1
    user_audio_update = {
        "update_type": "message_created",
        "update_id": up_id,
        "message": {
            "mid": f"msg_voice_{up_id}",
            "sender": {"user_id": 777, "first_name": "MaxClient"},
            "recipient": {"chat_id": 777},
            "body": {
                "text": "",
                "attachments": [
                    {
                        "type": "audio",
                        "payload": {
                            "token": "tok_audio_voice_file",
                            "url": "https://max.test/voice.ogg",
                        },
                    }
                ],
            },
        },
    }
    await app.handle_update(user_audio_update)

    # Await background task spawned for client user
    user_task = app.user_tasks.get(client_internal_uid)
    if user_task:
        await user_task
    elif app.background_tasks:
        await asyncio.gather(*list(app.background_tasks))

    # Step 6: Verify download boundary, Deepgram wire request, and user transcript delivery
    client.download_attachment.assert_awaited_once_with("tok_audio_voice_file", "https://max.test/voice.ogg")

    assert len(captured_max_http) == 1
    req_max = captured_max_http[0]
    parsed_max = urlparse(req_max["url"])
    assert parsed_max.path == "/v1/listen"
    qs_max = parse_qs(parsed_max.query)
    assert qs_max["model"] == ["nova-2"]
    assert qs_max["detect_language"] == ["true"]
    assert "language" not in qs_max
    assert req_max["headers"]["authorization"] == "Token test-live-key"
    assert req_max["content"] == b"ogg-binary-audio-for-max"

    # Verify transcript reached user in MAX via real edit_message/send_message boundary
    transcript_msgs = [
        m for m in sent_messages
        if "🎙 <i>Голос в MAX успешно распознан через nova-2</i>" in m["body"].get("text", "")
    ]
    assert transcript_msgs, "User must receive final transcript in MAX"
    delivered = transcript_msgs[0]
    assert delivered["method"] in {"POST", "PUT"}
    assert delivered["path"] == "/messages"
    assert "🎙 <i>Голос в MAX успешно распознан через nova-2</i>" in delivered["body"]["text"]

    # Step 7: Reopen Admin picker again -> nova-2 still shows ✅
    sent_messages.clear()
    screen_final = await press_admin_cb(f"admin_ai_provider_models_{PROVIDER_DEEPGRAM}")
    buttons_final = [b for row in screen_final["body"]["attachments"][0]["payload"]["buttons"] for b in row]
    final_n2 = next(b for b in buttons_final if "nova-2" in b["text"])
    final_n3 = next(b for b in buttons_final if "nova-3" in b["text"])
    assert "✅" in final_n2["text"]
    assert "✅" not in final_n3["text"]

    # Final DB check confirms no settings loss
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-2"


# ---------------------------------------------------------------------------
# 6. Button Classification & Stale Model Safety
# ---------------------------------------------------------------------------

def test_visible_button_classification_deepgram():
    """All affected visible Deepgram buttons must be classified (navigation or mutation)."""
    # Telegram models picker
    info = {m: {"name": m} for m in DEEPGRAM_TRANSCRIPTION_MODELS}
    tg_kb = kb.model_selection_keyboard(PROVIDER_DEEPGRAM, info, channel="transcription", back_callback="admin_ai_audio", current_model="nova-3")
    all_tg_callbacks = [b.callback_data for row in tg_kb.inline_keyboard for b in row]

    # MAX models picker
    max_k = max_kb.admin_ai_model_selection_keyboard(PROVIDER_DEEPGRAM, "nova-3", list(DEEPGRAM_TRANSCRIPTION_MODELS), back_callback="admin_ai_models_Deepgram")
    all_max_payloads = [b["payload"] for row in max_k[0]["payload"]["buttons"] for b in row]

    classified = 0
    for cb in all_tg_callbacks:
        if cb.startswith("ai_m_"):
            classification = "mutation"
        elif cb == "admin_ai_audio":
            classification = "navigation"
        else:
            pytest.fail(f"Unclassified Telegram callback: {cb}")
        classified += 1

    for pl in all_max_payloads:
        if pl.startswith(f"admin_ai_set_model_{PROVIDER_DEEPGRAM}_"):
            classification = "mutation"
        elif pl == f"admin_ai_models_{PROVIDER_DEEPGRAM}":
            classification = "navigation"
        else:
            pytest.fail(f"Unclassified MAX payload: {pl}")
        classified += 1

    assert classified == len(all_tg_callbacks) + len(all_max_payloads)


@pytest.mark.asyncio
async def test_telegram_and_max_stale_invalid_model_rejection(session_factory, monkeypatch, init_ai_config):
    """Verify that stale/invalid model callbacks (e.g. Flux or unknown) cannot silently persist in Telegram or MAX."""
    # 1. Telegram Dispatcher rejection
    async def _is_admin(uid, *args, **kwargs):
        return True

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", _is_admin)

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)
    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    admin_user = TgUser(id=999, is_bot=False, first_name="Admin")
    admin_chat = Chat(id=999, type=ChatType.PRIVATE)

    # An old/crafted callback that cannot resolve to any valid model
    invalid_cb_data = "ai_m_tr_0000000000000000"
    update_invalid = Update(
        update_id=99,
        callback_query=CallbackQuery(
            id="cb_inv",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Pick"),
            data=invalid_cb_data,
        ),
    )
    await dp.feed_update(bot, update_invalid)

    # DB remains nova-3
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-3"

    # 2. MAX Application rejection
    import max_messenger_bot.services.admin_ai as max_admin_ai
    import max_messenger_bot.services.common as max_common
    monkeypatch.setattr(max_admin_ai, "async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_common, "is_admin", _is_admin)

    sent_messages = []
    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        sent_messages.append({"method": method, "path": path, "params": params or {}, "body": json_data or {}})
        return {"ok": True, "result": {"message_id": 1}}

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = fake_request
    app = MaxBotApplication(client=client)

    await app.handle_update({
        "update_type": "message_callback",
        "update_id": 501,
        "callback": {
            "callback_id": "cb_501",
            "payload": f"admin_ai_set_model_{PROVIDER_DEEPGRAM}_flux-general-multi",
            "sender": {"user_id": 888, "first_name": "Admin"},
        },
        "message": {
            "mid": "msg_501",
            "recipient": {"chat_id": 888},
            "body": {"attachments": []},
        },
    })

    # DB still remains nova-3
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-3"

    # Check rejection message was sent
    assert any("Недопустимая модель" in (m["body"].get("text") or "") for m in sent_messages)
