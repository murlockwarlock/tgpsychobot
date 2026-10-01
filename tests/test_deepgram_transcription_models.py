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

from database import AIConfig, Base
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
# 3. Telegram Admin UI Journey (Audio -> Deepgram -> Select -> Persist -> Reopen -> Back)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_admin_journey_audio_model_picker(session_factory, monkeypatch, init_ai_config):
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

    # 1. Open audio settings
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

    # Click "🤖 Выбрать модель"
    model_btn = next(b for row in audio_kb.inline_keyboard for b in row if "Выбрать модель" in b.text)
    assert model_btn.callback_data == "admin_audio_model"

    # 2. Feed admin_audio_model into Dispatcher
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

    # Check visible model buttons
    buttons = [b for row in picker_kb.inline_keyboard for b in row]
    btn_texts = [b.text for b in buttons]
    assert any("nova-3" in t for t in btn_texts)
    assert any("nova-2" in t for t in btn_texts)

    # Currently nova-3 is default, so nova-3 has checkmark
    nova3_btn = next(b for b in buttons if "nova-3" in b.text)
    nova2_btn = next(b for b in buttons if "nova-2" in b.text)
    assert "✅" in nova3_btn.text
    assert "✅" not in nova2_btn.text

    # 3. Select nova-2 via Dispatcher
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

    # Verify DB persistence
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-2"

    # 4. Reopen model picker to verify ✅ is now on nova-2
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

    # 5. Click Back button -> returns to admin_ai_audio
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
    last_audio_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert any("Выбрать провайдера" in b.text for row in last_audio_kb.inline_keyboard for b in row)

    # 6. Switch back to nova-3
    update_6 = Update(
        update_id=6,
        callback_query=CallbackQuery(
            id="cb6",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Pick Model"),
            data="admin_audio_model",
        ),
    )
    await dp.feed_update(bot, update_6)
    picker_switch: InlineKeyboardMarkup = _last_markup(session.calls)
    sw_nova3 = next(b for row in picker_switch.inline_keyboard for b in row if "nova-3" in b.text)

    update_7 = Update(
        update_id=7,
        callback_query=CallbackQuery(
            id="cb7",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Select nova-3"),
            data=sw_nova3.callback_data,
        ),
    )
    await dp.feed_update(bot, update_7)

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-3"


# ---------------------------------------------------------------------------
# 4. Telegram Real User Voice Runtime E2E
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_real_voice_runtime_e2e(session_factory, monkeypatch, init_ai_config):
    """Admin configures nova-2 -> user sends voice -> real voice handler downloads -> Deepgram /v1/listen called with nova-2 -> transcript delivered."""
    # 1. Set DB config to Deepgram nova-2
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.transcription_provider = PROVIDER_DEEPGRAM
        cfg.deepgram_model = "nova-2"
        cfg.deepgram_api_key = "test-live-key"
        await s.commit()

    async def _not_admin(uid, *args, **kwargs):
        return False

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr("ai_integration.async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", _not_admin)

    # Intercept Deepgram HTTP request
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

    # Mock audio file download in bot
    bot.get_file = AsyncMock(return_value=File(file_id="voice_123", file_unique_id="u_123", file_path="voice.ogg"))
    bot.download_file = AsyncMock(return_value=io.BytesIO(b"ogg-audio-binary-data"))

    # Mock dialogue AI generation so the voice handler can complete smoothly
    monkeypatch.setattr(handlers, "process_user_prompt", AsyncMock())

    dp = Dispatcher(storage=MemoryStorage())
    handlers.router._parent_router = None
    dp.include_router(handlers.router)

    user = TgUser(id=555, is_bot=False, first_name="ClientUser")
    chat = Chat(id=555, type=ChatType.PRIVATE)

    voice_msg = Message(
        message_id=42,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        voice=Voice(file_id="voice_123", file_unique_id="u_123", duration=10),
    )
    voice_update = Update(update_id=101, message=voice_msg)

    # 2. Feed voice update through real Dispatcher
    await dp.feed_update(bot, voice_update)

    # 3. Verify outbound Deepgram HTTP request
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

    # 4. Verify outgoing aiogram methods passed Pydantic validation and thinking msg was updated with transcript
    edit_calls = [c for c in session.calls if isinstance(c, EditMessageText)]
    assert edit_calls, "Thinking message must be edited with transcript"
    assert "<i>Мне сегодня очень тревожно, помогите разобраться.</i>" in edit_calls[0].text


# ---------------------------------------------------------------------------
# 5. MAX Admin UI Journey & Voice Runtime E2E
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_admin_journey_and_runtime_voice_e2e(session_factory, monkeypatch, init_ai_config):
    """MAX Admin selects nova-2 -> DB persists -> reopen shows ✅ -> normal user voice transcribes via nova-2."""
    import max_messenger_bot.services.admin_ai as max_admin_ai
    import max_messenger_bot.ai as max_ai
    import max_messenger_bot.services.common as max_common
    monkeypatch.setattr(max_admin_ai, "async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)

    async def _is_admin(uid, *args, **kwargs):
        return True

    monkeypatch.setattr(max_common, "is_admin", _is_admin)

    sent_messages = []

    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        sent_messages.append({"method": method, "path": path, "params": params or {}, "body": json_data or {}})
        if path == "/messages":
            return {"message": {"body": {"mid": str(len(sent_messages))}}}
        return {}

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = fake_request
    app = MaxBotApplication(client=client)

    chat_id = 888
    user_id = 888
    up_id = 200

    async def press_cb(payload):
        nonlocal up_id
        up_id += 1
        await app.handle_update({
            "update_type": "message_callback",
            "update_id": up_id,
            "callback": {
                "callback_id": f"cb_{up_id}",
                "payload": payload,
                "sender": {"user_id": user_id, "first_name": "Admin"},
            },
            "message": {
                "mid": f"msg_{up_id}",
                "recipient": {"chat_id": chat_id},
                "body": {"attachments": []},
            },
        })
        messages = [m for m in sent_messages if m["path"] == "/messages"]
        return messages[-1]["body"]

    # 1. Open Deepgram model choices in MAX
    sent_messages.clear()
    models_screen = await press_cb(f"admin_ai_provider_models_{PROVIDER_DEEPGRAM}")
    attachment = models_screen["attachments"][0]
    buttons = [b for row in attachment["payload"]["buttons"] for b in row]

    # Verify both models visible
    assert any("nova-3" in b["text"] for b in buttons)
    assert any("nova-2" in b["text"] for b in buttons)

    # Initial selection shows ✅ on nova-3
    btn_n3 = next(b for b in buttons if "nova-3" in b["text"])
    btn_n2 = next(b for b in buttons if "nova-2" in b["text"])
    assert "✅" in btn_n3["text"]
    assert "✅" not in btn_n2["text"]
    assert btn_n2["payload"] == f"admin_ai_set_model_{PROVIDER_DEEPGRAM}_nova-2"

    # 2. Select nova-2
    sent_messages.clear()
    await press_cb(btn_n2["payload"])

    # Verify DB persistence
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.deepgram_model == "nova-2"

    # 3. Reopen model choices in MAX -> verify ✅ on nova-2
    sent_messages.clear()
    reopen_screen = await press_cb(f"admin_ai_provider_models_{PROVIDER_DEEPGRAM}")
    re_attachment = reopen_screen["attachments"][0]
    re_buttons = [b for row in re_attachment["payload"]["buttons"] for b in row]
    re_n3 = next(b for b in re_buttons if "nova-3" in b["text"])
    re_n2 = next(b for b in re_buttons if "nova-2" in b["text"])
    assert "✅" in re_n2["text"]
    assert "✅" not in re_n3["text"]

    # 4. Verify Back button leads to Deepgram provider settings
    back_btn = next(b for b in re_buttons if "Назад" in b["text"])
    assert back_btn["payload"] == f"admin_ai_models_{PROVIDER_DEEPGRAM}"

    # 5. MAX Runtime Voice E2E
    captured_max_http = []

    def max_deepgram_handler(req: httpx.Request) -> httpx.Response:
        captured_max_http.append(str(req.url))
        return httpx.Response(200, json={
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "Голос в MAX успешно распознан"
                    }]
                }]
            }
        })

    t_max = httpx.MockTransport(max_deepgram_handler)
    monkeypatch.setattr(httpx, "AsyncClient", mock_httpx_async_client(t_max))
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))
    monkeypatch.setattr("max_messenger_bot.services.common.run_ai_dialogue", AsyncMock())

    # Run MAX voice dialogue
    sent_messages.clear()
    await run_ai_dialogue_with_voice(client, chat_id=111, user_id=111, audio_bytes=b"max_audio_bytes", filename="audio.ogg")

    assert len(captured_max_http) == 1
    assert "model=nova-2" in captured_max_http[0]
    assert "detect_language=true" in captured_max_http[0]
    assert "language=multi" not in captured_max_http[0]


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
