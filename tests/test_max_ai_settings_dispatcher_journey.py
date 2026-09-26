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


def _classify_max_callback(payload):
    navigation = (
        "admin_panel",
        "admin_ai_settings",
        "admin_ai_keys",
        "admin_ai_main_chat",
        "admin_ai_main_chat_provider",
        "admin_ai_main_chat_model",
        "admin_ai_model_settings",
        "admin_ai_model_reasoning",
        "admin_ai_model_max_tokens",
        "admin_ai_model_temperature",
        "admin_ai_model_key",
        "admin_ai_model_choices",
        "admin_ai_models_",
        "admin_ai_provider_models_",
        "admin_ai_select_",
        "admin_ai_models_",
        "admin_ai_vision_fallback_models",
        "admin_ai_choose_capability_",
        "admin_ai_set_channel_model_",
        "admin_ai_set_model_",
        "admin_ai_reasoning_",
        "admin_ai_cancel_",
        "admin_ai_text_fallback",
        "admin_ai_fallback_provider",
        "admin_ai_fallback_model",
        "admin_ai_vision_fallback",
        "admin_ai_vision_fallback_provider",
        "admin_ai_vision_fallback_model",
        "admin_ai_audio",
        "admin_ai_vision",
        "admin_ai_image_generation",
        "admin_ai_image_edit",
        "admin_ai_common",
        "admin_ai_provider_models_",
    )
    mutation = (
        "admin_ai_key_",
        "admin_ai_main_chat_set_model_",
        "admin_ai_model_max_tokens_",
        "admin_ai_model_reasoning_",
        "admin_ai_model_temperature_",
        "admin_ai_model_key_",
        "admin_ai_set_kie_",
        "admin_ai_toggle_",
        "admin_ai_set_context_",
        "admin_ai_set_audio_limit",
        "admin_ai_cycle_memory_scope",
        "admin_ai_image_generation_models",
        "admin_ai_image_edit_models",
        "admin_ai_vision_models",
        "admin_ai_fallback_models",
        "admin_ai_set_fallback_provider_",
        "admin_ai_save_fallback_",
        "admin_ai_set_vision_model_",
        "admin_ai_set_image_gen_model_",
        "admin_ai_set_image_edit_model_",
        "admin_ai_set_vision_fallback_provider_",
        "admin_ai_save_vision_fallback_",
        "admin_ai_vision_fallback_models",
        "admin_ai_fallback_toggle",
        "admin_ai_vision_fallback_toggle",
        "admin_ai_image_generation_toggle",
            "admin_ai_image_edit_toggle",
            "admin_ai_deepseek_proxy",
        )
    if payload.startswith(navigation) or payload.startswith(mutation):
        return "mutation" if payload.startswith(mutation) else "navigation"
    raise AssertionError(f"unclassified changed MAX AI callback: {payload}")


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
    classified_callbacks = set()
    visible_callback_count = 0
    classified_callback_count = 0
    model_picker_back_passes = 0
    journey_counts = {
        "api_keys": 0,
        "text_fallback": 0,
        "vision_fallback": 0,
        "text_fallback_model_back": 0,
        "vision_fallback_model_back": 0,
        "api_key_back": 0,
        "deepseek_proxy": 0,
        "deepseek_reasoning": 0,
        "deepseek_reasoning_back": 0,
    }

    def classify_visible(attachments):
        nonlocal visible_callback_count, classified_callback_count
        for button in _max_buttons(attachments):
            visible_callback_count += 1
            _classify_max_callback(button["payload"])
            classified_callback_count += 1
            classified_callbacks.add(button["payload"])

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
    classify_visible(keys.get("attachments"))

    key_buttons = _max_buttons(keys.get("attachments"))
    key_buttons_only = [button for button in key_buttons if button["payload"].startswith("admin_ai_key_")]
    model_buttons = [button for button in key_buttons if button["payload"].startswith("admin_ai_models_")]
    assert len(key_buttons_only) == 8
    assert len(model_buttons) == 8
    rows = keys["attachments"][0]["payload"]["buttons"]
    assert all(len(row) == 2 for row in rows[:8])
    assert all(len(row) == 1 for row in rows[8:])

    key_fields = {
        "Deepseek": "deepseek_api_key",
        "Claude": "claude_api_key",
        "Gemini": "gemini_api_key",
        "OpenAI": "openai_api_key",
        "KIE": "kie_api_key",
        "OpenRouter": "openrouter_api_key",
        "Perplexity": "perplexity_api_key",
        "Deepgram": "deepgram_api_key",
    }
    for button in key_buttons_only:
        provider = button["payload"].replace("admin_ai_key_", "", 1)
        secret = f"max-isolated-{provider.lower()}-key"
        prompt = await press(button["payload"], keys.get("attachments", []))
        assert "API key" in prompt["text"]
        keys = await send_admin_command(secret)
        assert "Провайдеры и модели" in keys["text"]
        async with max_settings_db() as session:
            config = await session.get(AIConfig, 1)
            assert getattr(config, key_fields[provider]) == secret
        detail = await press(f"admin_ai_models_{provider}", keys.get("attachments", []))
        assert provider in detail["text"]
        back = next(item for item in _max_buttons(detail.get("attachments", [])) if item["payload"] == "admin_ai_keys")
        keys = await press(back["payload"], detail.get("attachments", []))
        assert "Провайдеры и модели" in keys["text"]
        journey_counts["api_keys"] += 1
        journey_counts["api_key_back"] += 1
        classify_visible(keys.get("attachments"))
    assert journey_counts["api_keys"] == 8

    chat_providers = {payload.replace("admin_ai_models_", "", 1) for payload in (button["payload"] for button in model_buttons)}
    assert "Deepgram" in chat_providers
    chat_providers.discard("Deepgram")
    assert len(chat_providers) == 7
    for button in model_buttons:
        provider = button["payload"].replace("admin_ai_models_", "", 1)
        detail = await press(button["payload"], keys.get("attachments", []))
        assert provider in detail["text"]
        detail_buttons = _max_buttons(detail.get("attachments"))
        classify_visible(detail.get("attachments"))
        if provider != "Deepgram":
            assert any(item["payload"] == f"admin_ai_model_max_tokens_{provider}" for item in detail_buttons)
            prompt = await press(f"admin_ai_model_max_tokens_{provider}", detail.get("attachments", []))
            assert "Авто" in prompt["text"]
            classify_visible(prompt.get("attachments"))
            saved = await send_admin_command("Авто")
            assert provider in saved["text"] and "Max tokens" in saved["text"]
            classify_visible(saved.get("attachments"))
        else:
            saved = detail
        model_screen = await press(f"admin_ai_provider_models_{provider}", saved.get("attachments", []))
        classify_visible(model_screen.get("attachments"))
        choices = _max_buttons(model_screen.get("attachments"))
        assert choices, provider
        model_back = next(item for item in choices if item["payload"] == f"admin_ai_models_{provider}")
        detail = await press(model_back["payload"], model_screen.get("attachments", []))
        assert provider in detail["text"]
        model_picker_back_passes += 1
        model_screen = await press(f"admin_ai_provider_models_{provider}", detail.get("attachments", []))
        classify_visible(model_screen.get("attachments"))
        choices = _max_buttons(model_screen.get("attachments"))
        selected = await press(choices[0]["payload"], model_screen.get("attachments", []))
        assert provider in selected["text"]
        classify_visible(selected.get("attachments"))
        keys = await press("admin_ai_keys", selected.get("attachments", []))

    deepseek_detail = await press("admin_ai_models_Deepseek", keys.get("attachments", []))
    classify_visible(deepseek_detail.get("attachments"))
    proxy_button = next(item for item in _max_buttons(deepseek_detail.get("attachments", [])) if item["payload"] == "admin_ai_deepseek_proxy")
    async with max_settings_db() as session:
        config = await session.get(AIConfig, 1)
        proxy_before = bool(config.use_proxy)
    deepseek_detail = await press(proxy_button["payload"], deepseek_detail.get("attachments", []))
    classify_visible(deepseek_detail.get("attachments"))
    async with max_settings_db() as session:
        config = await session.get(AIConfig, 1)
        assert bool(config.use_proxy) is not proxy_before
    journey_counts["deepseek_proxy"] += 1
    keys = await press("admin_ai_keys", deepseek_detail.get("attachments", []))
    deepseek_detail = await press("admin_ai_models_Deepseek", keys.get("attachments", []))
    deepseek_detail = await press("admin_ai_deepseek_proxy", deepseek_detail.get("attachments", []))
    classify_visible(deepseek_detail.get("attachments"))
    async with max_settings_db() as session:
        config = await session.get(AIConfig, 1)
        assert bool(config.use_proxy) is proxy_before

    reasoning_picker = await press("admin_ai_model_reasoning_Deepseek", deepseek_detail.get("attachments", []))
    classify_visible(reasoning_picker.get("attachments"))
    reasoning_choices = [
        item
        for item in _max_buttons(reasoning_picker.get("attachments"))
        if item["payload"].startswith("admin_ai_reasoning_Deepseek_")
    ]
    assert {item["payload"].rsplit("_", 1)[-1] for item in reasoning_choices} == {"auto", "none", "low", "high", "max"}
    reasoning_back = next(item for item in _max_buttons(reasoning_picker.get("attachments")) if item["payload"] == "admin_ai_models_Deepseek")
    deepseek_detail = await press(reasoning_back["payload"], reasoning_picker.get("attachments", []))
    journey_counts["deepseek_reasoning_back"] += 1
    reasoning_picker = await press("admin_ai_model_reasoning_Deepseek", deepseek_detail.get("attachments", []))
    max_reasoning = next(item for item in _max_buttons(reasoning_picker.get("attachments")) if item["payload"] == "admin_ai_reasoning_Deepseek_max")
    deepseek_detail = await press(max_reasoning["payload"], reasoning_picker.get("attachments", []))
    classify_visible(deepseek_detail.get("attachments"))
    assert "Reasoning: <b>Max</b>" in deepseek_detail["text"]
    journey_counts["deepseek_reasoning"] += 1
    keys = await press("admin_ai_keys", deepseek_detail.get("attachments", []))
    deepseek_detail = await press("admin_ai_models_Deepseek", keys.get("attachments", []))
    assert "Reasoning: <b>Max</b>" in deepseek_detail["text"]
    keys = await press("admin_ai_keys", deepseek_detail.get("attachments", []))
    assert journey_counts["deepseek_proxy"] == 1
    assert journey_counts["deepseek_reasoning"] == 1
    assert journey_counts["deepseek_reasoning_back"] == 1

    picker_callbacks = {
        "admin_ai_select_transcription_provider": "transcription",
        "admin_ai_select_vision_provider": "vision",
        "admin_ai_select_image_generation_provider": "image_gen",
        "admin_ai_select_image_edit_provider": "image_edit",
    }
    for picker_payload, channel in picker_callbacks.items():
        picker = await press(picker_payload, keys.get("attachments", []))
        classify_visible(picker.get("attachments"))
        providers = [
            button
            for button in _max_buttons(picker.get("attachments"))
            if button["payload"].startswith("admin_ai_choose_capability_")
        ]
        expected = set(get_capability_providers(channel))
        assert expected <= {button["payload"].rsplit("_", 1)[-1] for button in providers}
        for provider_button in providers:
            chosen = await press(provider_button["payload"], picker.get("attachments", []))
            classify_visible(chosen.get("attachments"))
            if provider_button["payload"].endswith("_None"):
                keys = chosen
                assert "Провайдеры и модели" in keys["text"]
                continue
            model_choice = _max_buttons(chosen.get("attachments"))[0]
            keys = await press(model_choice["payload"], chosen.get("attachments", []))
            classify_visible(keys.get("attachments"))
            heading_title = {
                "transcription": "Аудио",
                "vision": "Vision",
                "image_gen": "Генерация",
                "image_edit": "Редактирование",
            }[channel]
            assert heading_title in keys["text"], (channel, provider_button["payload"], model_choice["payload"])
            picker = await press(picker_payload, keys.get("attachments", []))

    fallback_providers = get_capability_providers("fallback")
    for expected_provider in fallback_providers:
        fallback = await press("admin_ai_text_fallback", keys.get("attachments", []))
        classify_visible(fallback.get("attachments"))
        provider_picker = await press("admin_ai_fallback_provider", fallback.get("attachments", []))
        classify_visible(provider_picker.get("attachments"))
        provider_buttons = [
            item
            for item in _max_buttons(provider_picker.get("attachments"))
            if item["payload"].startswith("admin_ai_set_fallback_provider_")
        ]
        assert {item["payload"].replace("admin_ai_set_fallback_provider_", "", 1) for item in provider_buttons} == set(fallback_providers)
        provider_button = next(
            item
            for item in provider_buttons
            if item["payload"] == f"admin_ai_set_fallback_provider_{expected_provider}"
        )
        model_picker = await press(provider_button["payload"], provider_picker.get("attachments", []))
        classify_visible(model_picker.get("attachments"))
        model_buttons = [
            item
            for item in _max_buttons(model_picker.get("attachments"))
            if item["payload"].startswith(f"admin_ai_save_fallback_{expected_provider}_")
        ]
        assert model_buttons
        model_back = next(item for item in _max_buttons(model_picker.get("attachments")) if item["payload"] == "admin_ai_text_fallback")
        fallback = await press(model_back["payload"], model_picker.get("attachments", []))
        assert "Резерв текста" in fallback["text"]
        journey_counts["text_fallback_model_back"] += 1
        model_picker = await press("admin_ai_fallback_model", fallback.get("attachments", []))
        classify_visible(model_picker.get("attachments"))
        model_buttons = [
            item
            for item in _max_buttons(model_picker.get("attachments"))
            if item["payload"].startswith(f"admin_ai_save_fallback_{expected_provider}_")
        ]
        selected_model_button = model_buttons[-1]
        selected_fallback_model = selected_model_button["payload"].replace(f"admin_ai_save_fallback_{expected_provider}_", "", 1)
        fallback = await press(selected_model_button["payload"], model_picker.get("attachments", []))
        classify_visible(fallback.get("attachments"))
        if "Включён" in fallback["text"]:
            fallback = await press("admin_ai_fallback_toggle", fallback.get("attachments", []))
            classify_visible(fallback.get("attachments"))
        assert "Выключен" in fallback["text"]
        fallback = await press("admin_ai_fallback_toggle", fallback.get("attachments", []))
        classify_visible(fallback.get("attachments"))
        assert "Включён" in fallback["text"]
        fallback = await press("admin_ai_fallback_toggle", fallback.get("attachments", []))
        classify_visible(fallback.get("attachments"))
        assert "Выключен" in fallback["text"]
        keys = await press("admin_ai_keys", fallback.get("attachments", []))
        fallback = await press("admin_ai_text_fallback", keys.get("attachments", []))
        classify_visible(fallback.get("attachments"))
        assert "Выключен" in fallback["text"]
        assert expected_provider in fallback["text"] and selected_fallback_model in fallback["text"]
        fallback = await press("admin_ai_fallback_toggle", fallback.get("attachments", []))
        classify_visible(fallback.get("attachments"))
        assert "Включён" in fallback["text"]
        assert expected_provider in fallback["text"] and selected_fallback_model in fallback["text"]
        keys = await press("admin_ai_keys", fallback.get("attachments", []))
        journey_counts["text_fallback"] += 1
    assert journey_counts["text_fallback"] == len(fallback_providers)

    for picker_payload, heading, channel in (
        ("admin_ai_audio", "Аудио", "transcription"),
        ("admin_ai_vision", "Vision", "vision"),
        ("admin_ai_image_generation", "Генерация", "image_gen"),
        ("admin_ai_image_edit", "Редактирование", "image_edit"),
    ):
        screen = await press(picker_payload, keys.get("attachments", []))
        classify_visible(screen.get("attachments"))
        assert heading in screen["text"]
        provider_picker = await press(
            {
                "admin_ai_audio": "admin_ai_select_transcription_provider",
                "admin_ai_vision": "admin_ai_select_vision_provider",
                "admin_ai_image_generation": "admin_ai_select_image_generation_provider",
                "admin_ai_image_edit": "admin_ai_select_image_edit_provider",
            }[picker_payload],
            screen.get("attachments", []),
        )
        classify_visible(provider_picker.get("attachments"))
        provider_buttons = [
            item for item in _max_buttons(provider_picker.get("attachments"))
            if item["payload"].startswith(f"admin_ai_choose_capability_{channel}_")
        ]
        assert provider_buttons, channel
        chosen = await press(provider_buttons[0]["payload"], provider_picker.get("attachments", []))
        classify_visible(chosen.get("attachments"))
        model_buttons_for_channel = _max_buttons(chosen.get("attachments"))
        if model_buttons_for_channel and not model_buttons_for_channel[0]["payload"].endswith("_None"):
            selected = next(item for item in model_buttons_for_channel if not item["payload"].endswith("_None"))
            chosen = await press(selected["payload"], chosen.get("attachments", []))
            classify_visible(chosen.get("attachments"))
        assert heading in chosen["text"]
        keys = await press("admin_ai_keys", chosen.get("attachments", []))

    vision_fallback_providers = get_capability_providers("vision_fallback")
    for expected_provider in vision_fallback_providers:
        vision_fallback = await press("admin_ai_vision_fallback", keys.get("attachments", []))
        classify_visible(vision_fallback.get("attachments"))
        provider_picker = await press("admin_ai_vision_fallback_provider", vision_fallback.get("attachments", []))
        classify_visible(provider_picker.get("attachments"))
        provider_buttons = [
            item
            for item in _max_buttons(provider_picker.get("attachments"))
            if item["payload"].startswith("admin_ai_set_vision_fallback_provider_")
        ]
        assert {item["payload"].replace("admin_ai_set_vision_fallback_provider_", "", 1) for item in provider_buttons} == set(vision_fallback_providers)
        provider_button = next(
            item
            for item in provider_buttons
            if item["payload"] == f"admin_ai_set_vision_fallback_provider_{expected_provider}"
        )
        model_picker = await press(provider_button["payload"], provider_picker.get("attachments", []))
        classify_visible(model_picker.get("attachments"))
        model_buttons = [
            item
            for item in _max_buttons(model_picker.get("attachments"))
            if item["payload"].startswith(f"admin_ai_save_vision_fallback_{expected_provider}_")
        ]
        assert model_buttons
        model_back = next(item for item in _max_buttons(model_picker.get("attachments")) if item["payload"] == "admin_ai_vision_fallback")
        vision_fallback = await press(model_back["payload"], model_picker.get("attachments", []))
        assert "Vision резерв" in vision_fallback["text"]
        journey_counts["vision_fallback_model_back"] += 1
        model_picker = await press("admin_ai_vision_fallback_model", vision_fallback.get("attachments", []))
        classify_visible(model_picker.get("attachments"))
        model_buttons = [
            item
            for item in _max_buttons(model_picker.get("attachments"))
            if item["payload"].startswith(f"admin_ai_save_vision_fallback_{expected_provider}_")
        ]
        selected_model_button = model_buttons[-1]
        selected_vision_model = selected_model_button["payload"].replace(f"admin_ai_save_vision_fallback_{expected_provider}_", "", 1)
        vision_fallback = await press(selected_model_button["payload"], model_picker.get("attachments", []))
        classify_visible(vision_fallback.get("attachments"))
        assert "Vision резерв" in vision_fallback["text"] and expected_provider in vision_fallback["text"]
        if "Включён" in vision_fallback["text"]:
            vision_fallback = await press("admin_ai_vision_fallback_toggle", vision_fallback.get("attachments", []))
            classify_visible(vision_fallback.get("attachments"))
        vision_fallback = await press("admin_ai_vision_fallback_toggle", vision_fallback.get("attachments", []))
        classify_visible(vision_fallback.get("attachments"))
        assert "Включён" in vision_fallback["text"]
        vision_fallback = await press("admin_ai_vision_fallback_toggle", vision_fallback.get("attachments", []))
        classify_visible(vision_fallback.get("attachments"))
        assert "Выключен" in vision_fallback["text"]
        keys = await press("admin_ai_keys", vision_fallback.get("attachments", []))
        vision_fallback = await press("admin_ai_vision_fallback", keys.get("attachments", []))
        classify_visible(vision_fallback.get("attachments"))
        assert "Выключен" in vision_fallback["text"]
        assert expected_provider in vision_fallback["text"] and selected_vision_model in vision_fallback["text"]
        vision_fallback = await press("admin_ai_vision_fallback_toggle", vision_fallback.get("attachments", []))
        classify_visible(vision_fallback.get("attachments"))
        assert "Включён" in vision_fallback["text"]
        assert expected_provider in vision_fallback["text"] and selected_vision_model in vision_fallback["text"]
        keys = await press("admin_ai_keys", vision_fallback.get("attachments", []))
        journey_counts["vision_fallback"] += 1
    assert journey_counts["vision_fallback"] == len(vision_fallback_providers)
    ai = await press("admin_ai_settings", keys.get("attachments", []))
    root = await press("admin_panel", ai.get("attachments", []))
    assert "Добро пожаловать" in root["text"]
    assert classified_callbacks
    assert model_picker_back_passes == 8
    assert classified_callback_count == visible_callback_count
    assert journey_counts["api_keys"] == 8
    assert journey_counts["text_fallback"] == len(get_capability_providers("fallback"))
    assert journey_counts["vision_fallback"] == len(get_capability_providers("vision_fallback"))
    assert journey_counts["deepseek_proxy"] == 1
    assert journey_counts["deepseek_reasoning"] == 1
    assert journey_counts["deepseek_reasoning_back"] == 1
    assert journey_counts["api_key_back"] == 8
    assert journey_counts["text_fallback_model_back"] == len(get_capability_providers("fallback"))
    assert journey_counts["vision_fallback_model_back"] == len(get_capability_providers("vision_fallback"))
    assert (
        model_picker_back_passes
        + journey_counts["api_key_back"]
        + journey_counts["text_fallback_model_back"]
        + journey_counts["vision_fallback_model_back"]
        + journey_counts["deepseek_reasoning_back"]
        == 30
    )


async def _true_async():
    return True
