import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import AIConfig, Base
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot import app as max_app_module
from max_messenger_bot.services import admin_ai as max_admin_ai
from max_messenger_bot.storage import StorageBase
from provider_models import get_capability_providers


@pytest_asyncio.fixture
async def max_settings_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)
    async with sessions() as session:
        session.add(
            AIConfig(
                id=1,
                provider="Deepseek",
                deepseek_model="deepseek-flash",
                transcription_provider="OpenAI",
                vision_provider="Gemini",
                vision_model="gemini-3.7-flash",
                image_generation_provider="OpenAI",
                image_generation_model="gpt-image-2",
                image_edit_provider="KIE",
                image_edit_model="seedream/4.5-edit",
                allow_image_generation=False,
                allow_image_edit=False,
                allow_vision_fallback=False,
            )
        )
        await session.commit()
    yield sessions
    await engine.dispose()


def _max_buttons(attachments):
    return [
        button
        for attachment in (attachments or [])
        if attachment.get("type") == "inline_keyboard"
        for row in attachment.get("payload", {}).get("buttons", [])
        for button in row
        if isinstance(button, dict) and button.get("type") == "callback"
    ]


def _validate_max_payload(path, json_data):
    if path in {"/messages", "/answers"} and json_data is not None:
        assert isinstance(json_data, dict)
        body = json_data.get("text")
        if body is not None:
            assert isinstance(body, str)
        nested = json_data.get("message")
        if isinstance(nested, dict) and nested.get("text") is not None:
            assert isinstance(nested["text"], str)
        for attachment in json_data.get("attachments", []):
            assert attachment.get("type") == "inline_keyboard"
            buttons = attachment.get("payload", {}).get("buttons")
            assert isinstance(buttons, list)
            for row in buttons:
                assert isinstance(row, list)
                for button in row:
                    assert isinstance(button.get("text"), str)
                    assert isinstance(button.get("payload"), str)


@pytest.mark.asyncio
async def test_max_ai_settings_use_real_app_callback_journeys(max_settings_db, monkeypatch):
    monkeypatch.setattr(max_admin_ai, "async_session_maker", max_settings_db)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", max_settings_db)
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())

    captured = []

    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        _validate_max_payload(path, json_data)
        captured.append({"method": method, "path": path, "params": params or {}, "body": json_data or {}})
        if path == "/messages":
            return {"message": {"body": {"mid": str(len(captured))}}}
        return {}

    client = MaxApiClient(token="test", base_url="http://max.test")
    client._request = fake_request
    app = MaxBotApplication(client=client)
    update_id = 1

    async def send_admin_command(command):
        nonlocal update_id
        update_id += 1
        await app.handle_update({
            "update_type": "message_created",
            "update_id": update_id,
            "message": {
                "body": {"text": command},
                "recipient": {"chat_id": 1001},
                "sender": {"user_id": 1001, "name": "Admin"},
            },
        })
        return captured[-1]["body"]

    async def press(payload, attachments):
        nonlocal update_id
        update_id += 1
        await app.handle_update({
            "update_type": "message_callback",
            "update_id": update_id,
            "callback": {
                "callback_id": f"cb-{update_id}",
                "payload": payload,
                "sender": {"user_id": 1001, "name": "Admin"},
            },
            "message": {
                "mid": "admin-message",
                "recipient": {"chat_id": 1001},
                "body": {"attachments": attachments},
            },
        })
        messages = [item for item in captured if item["path"] == "/messages"]
        assert messages, payload
        return messages[-1]["body"]

    root = await send_admin_command("/admin")
    assert "Добро пожаловать в админ-панель MAX" in root["text"]
    ai = await press("admin_ai_settings", root.get("attachments", []))
    assert "Настройки ИИ" in ai["text"]
    keys = await press("admin_ai_keys", ai.get("attachments", []))
    assert "Провайдеры и модели" in keys["text"]

    key_buttons = _max_buttons(keys.get("attachments"))
    key_buttons_only = [button for button in key_buttons if button["payload"].startswith("admin_ai_key_")]
    model_buttons = [button for button in key_buttons if button["payload"].startswith("admin_ai_models_")]
    assert len(key_buttons_only) == 8
    assert len(model_buttons) == 8
    rows = keys["attachments"][0]["payload"]["buttons"]
    assert all(len(row) == 2 for row in rows[:4])
    assert all(len(row) == 2 for row in rows[5:9])

    for button in key_buttons_only:
        prompt = await press(button["payload"], keys.get("attachments", []))
        assert "API key" in prompt["text"]
        keys = await send_admin_command("/admin")
        keys = await press("admin_ai_keys", keys.get("attachments", []))

    for button in model_buttons:
        model_screen = await press(button["payload"], keys.get("attachments", []))
        choices = _max_buttons(model_screen.get("attachments"))
        assert choices, button["payload"]
        selected = await press(choices[0]["payload"], model_screen.get("attachments", []))
        assert isinstance(selected["text"], str)
        keys = await press("admin_ai_keys", selected.get("attachments", []))

    picker_callbacks = {
        "admin_ai_select_transcription_provider": "transcription",
        "admin_ai_select_vision_provider": "vision",
        "admin_ai_select_image_generation_provider": "image_gen",
        "admin_ai_select_image_edit_provider": "image_edit",
    }
    for picker_payload, channel in picker_callbacks.items():
        picker = await press(picker_payload, keys.get("attachments", []))
        providers = [
            button
            for button in _max_buttons(picker.get("attachments"))
            if button["payload"].startswith("admin_ai_choose_capability_")
        ]
        expected = set(get_capability_providers(channel))
        assert expected <= {button["payload"].rsplit("_", 1)[-1] for button in providers}
        for provider_button in providers:
            chosen = await press(provider_button["payload"], picker.get("attachments", []))
            if provider_button["payload"].endswith("_None"):
                keys = chosen
                assert "Провайдеры и модели" in keys["text"]
                continue
            model_choice = _max_buttons(chosen.get("attachments"))[0]
            keys = await press(model_choice["payload"], chosen.get("attachments", []))
            assert "Провайдеры и модели" in keys["text"], (channel, provider_button["payload"], model_choice["payload"])
            picker = await press(picker_payload, keys.get("attachments", []))

    await press("admin_ai_toggle_vision_fallback", keys.get("attachments", []))
    fallback_picker = await press("admin_ai_vision_fallback_models", keys.get("attachments", []))
    fallback_provider = _max_buttons(fallback_picker.get("attachments"))[0]
    fallback_models = await press(fallback_provider["payload"], fallback_picker.get("attachments", []))
    fallback_model = _max_buttons(fallback_models.get("attachments"))[0]
    keys = await press(fallback_model["payload"], fallback_models.get("attachments", []))
    assert "Vision" in keys["text"]

    model_settings = await press("admin_ai_model_settings", keys.get("attachments", []))
    assert "Max tokens" in model_settings["text"]
    reasoning = await press("admin_ai_model_reasoning", model_settings.get("attachments", []))
    max_button = next(button for button in _max_buttons(reasoning.get("attachments")) if button["payload"].endswith("max"))
    model_settings = await press(max_button["payload"], reasoning.get("attachments", []))
    assert "Reasoning: <b>Max" in model_settings["text"]
    await press("admin_ai_model_max_tokens", model_settings.get("attachments", []))
    await send_admin_command("Авто")
    model_settings = await press("admin_ai_model_settings", keys.get("attachments", []))
    await press("admin_ai_model_reasoning", model_settings.get("attachments", []))
    none_button = next(button for button in _max_buttons(captured[-1]["body"].get("attachments")) if button["payload"].endswith("none"))
    model_settings = await press(none_button["payload"], captured[-1]["body"].get("attachments", []))
    assert "🌡 Temperature" in model_settings["text"]
    keys = await press("admin_ai_keys", model_settings.get("attachments", []))
    ai = await press("admin_ai_settings", keys.get("attachments", []))
    root = await press("admin_panel", ai.get("attachments", []))
    assert "Добро пожаловать" in root["text"]


async def _true_async():
    return True
