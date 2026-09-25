from __future__ import annotations

import html
from pathlib import Path

from ..api import MaxApiClient
from ..keyboards import admin_ai_model_selection_keyboard, admin_ai_settings_keyboard, admin_ai_vision_models_keyboard, callback_button, inline_keyboard
from ..legacy import AIConfig, AIModelSettings, async_session_maker
from ..storage import StateStore
from memory_mode import MEMORY_MODE_RESET, memory_mode_label, next_memory_mode, normalize_memory_mode
from provider_models import (
    PROVIDER_CLAUDE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    PROVIDER_PERPLEXITY,
    PROVIDER_DEEPGRAM,
    ModelUnavailableError,
    canonical_provider_name,
    get_default_model,
    get_capability_providers,
    get_selectable_models,
    validate_model_selection,
)
from ai_model_settings import (
    get_generation_capabilities,
    get_or_create_model_settings,
    resolve_model_settings,
    validate_model_setting,
    REASONING_AUTO,
    REASONING_NONE,
    REASONING_LOW,
    REASONING_HIGH,
    REASONING_MAX,
)

FALLBACK_PROVIDERS = ["OpenAI", "Gemini", "Claude", "Deepseek", "KIE"]

KEY_FIELDS = {
    "Deepseek": "deepseek_api_key",
    "Claude": "claude_api_key",
    "Gemini": "gemini_api_key",
    "OpenAI": "openai_api_key",
    "KIE": "kie_api_key",
    "OpenRouter": "openrouter_api_key",
    "Perplexity": "perplexity_api_key",
    "Deepgram": "deepgram_api_key",
}

MODEL_FIELDS = {
    "Deepseek": "deepseek_model",
    "Claude": "claude_model",
    "Gemini": "gemini_model",
    "OpenAI": "openai_model",
    "KIE": "kie_model",
    "OpenRouter": "openrouter_model",
    "Perplexity": "perplexity_model",
    "Deepgram": "deepgram_model",
}

KIE_EXTRA_FIELDS = {
    "kie_base_url": "KIE Base URL",
    "kie_upload_base_url": "KIE Upload URL",
    "kie_transcription_model": "KIE Transcription Model",
}


def _mask(value: str | None) -> str:
    if not value:
        return "Не задан"
    if len(value) <= 8:
        return "••••"
    return f"{value[:4]}...{value[-4:]}"


def _provider_model(provider: str | None, model: str | None) -> str:
    provider_label = provider or "нет"
    model_label = model or "нет"
    return f"{provider_label}/{model_label}"


def _split_provider_model_payload(payload: str) -> tuple[str | None, str]:
    """Parse provider-bound callbacks while retaining validated legacy support."""
    provider, separator, model = (payload or "").partition("_")
    if separator and provider in MODEL_FIELDS:
        return provider, model
    return None, payload or ""


async def _reject_model_selection(client: MaxApiClient, chat_id: int) -> None:
    await client.send_message(chat_id=chat_id, text="Недопустимая модель. Настройки не изменены.")


def _fallback_model_for_provider(config: AIConfig, provider: str | None) -> str:
    selectable = get_selectable_models(provider or "", channel="fallback")
    current_fb_provider = getattr(config, "fallback_provider", None)
    current_fb_model = getattr(config, "fallback_model", None)
    if current_fb_provider == provider and current_fb_model in selectable:
        return current_fb_model
    return get_default_model(provider or "", channel="fallback")


def _prompt_input_keyboard(cancel_payload: str) -> list[dict]:
    return inline_keyboard([[callback_button("⬅️ Отмена", cancel_payload)]])


def _status_model_button(prefix: str, enabled: bool, provider: str | None, model: str | None) -> str:
    status = "✅" if enabled else "❌"
    return f"{prefix} {status} {_provider_model(provider, model)}"


async def _get_config() -> AIConfig:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        await session.commit()
        await session.refresh(config)
        return config


async def _ensure_session_config(session) -> AIConfig:
    config = await session.get(AIConfig, 1)
    if not config:
        config = AIConfig(id=1)
        session.add(config)
        await session.flush()
    return config


async def show_settings(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    model_name = getattr(config, MODEL_FIELDS.get(config.provider or "", ""), None) or "не выбрана"
    text = (
        "🤖 <b>Настройки ИИ</b>\n\n"
        f"▫️ Текущий провайдер: <b>{html.escape(config.provider or 'Не задан')}</b>\n"
        f"▫️ Активная модель: <code>{html.escape(model_name)}</code>\n\n"
        f"🎙 <b>Аудио:</b> {html.escape(config.transcription_provider or 'OpenAI')}\n"
        f"🖼 <b>Vision:</b> {html.escape(config.vision_provider)} / <code>{html.escape(config.vision_model)}</code>\n"
        f"⏱️ <b>Лимит аудио:</b> {config.max_voice_duration_sec} сек."
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_ai_settings_keyboard(config.provider or "Gemini"))


async def set_provider(client: MaxApiClient, chat_id: int, provider: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.provider = provider
        await session.commit()
    await show_settings(client, chat_id)


VISION_FALLBACK_PROVIDERS = list(get_capability_providers("vision"))


def _vision_fallback_model_for_provider(config: AIConfig, provider: str | None) -> str:
    if not provider:
        return "—"
    selectable = get_selectable_models(provider, channel="vision_fallback")
    current_fb_provider = getattr(config, "vision_fallback_provider", None)
    current_fb_model = getattr(config, "vision_fallback_model", None)
    if current_fb_provider == provider and current_fb_model in selectable:
        return current_fb_model
    try:
        return get_default_model(provider, channel="vision_fallback")
    except Exception:
        return "—"


def _build_keys_keyboard(config) -> list:
    img_gen_enabled = getattr(config, 'allow_image_generation', False)
    img_edit_enabled = getattr(config, 'allow_image_edit', False)
    fallback_enabled = getattr(config, 'allow_fallback', False)
    vision_fallback_enabled = getattr(config, 'allow_vision_fallback', False)
    vision_fb_provider = getattr(config, 'vision_fallback_provider', None)
    vision_fb_model = getattr(config, 'vision_fallback_model', None) or _vision_fallback_model_for_provider(config, vision_fb_provider)
    memory = normalize_memory_mode(config)
    transcription_provider = getattr(config, "transcription_provider", None) or "None"
    transcription_label = transcription_provider
    if transcription_provider == PROVIDER_DEEPGRAM:
        transcription_label = f"{transcription_provider} / {getattr(config, 'deepgram_model', None) or get_default_model(PROVIDER_DEEPGRAM, channel='transcription')}"
    elif transcription_provider == "None":
        transcription_label = "выкл"
    rows = [
        [callback_button(f"Deepseek: {_mask(config.deepseek_api_key)}", "admin_ai_key_Deepseek"),
         callback_button(f"Claude: {_mask(config.claude_api_key)}", "admin_ai_key_Claude")],
        [callback_button(f"Gemini: {_mask(config.gemini_api_key)}", "admin_ai_key_Gemini"),
         callback_button(f"OpenAI: {_mask(config.openai_api_key)}", "admin_ai_key_OpenAI")],
        [callback_button(f"KIE: {_mask(config.kie_api_key)}", "admin_ai_key_KIE"),
         callback_button(f"OpenRouter: {_mask(getattr(config, 'openrouter_api_key', None))}", "admin_ai_key_OpenRouter")],
        [callback_button(f"Perplexity: {_mask(getattr(config, 'perplexity_api_key', None))}", "admin_ai_key_Perplexity"),
         callback_button(f"Deepgram: {_mask(getattr(config, 'deepgram_api_key', None))}", "admin_ai_key_Deepgram")],
        [callback_button(f"📊 Порог KIE: {config.kie_credit_alert_threshold}", "admin_ai_set_kie_threshold")],
        [callback_button("🔤 Deepseek модель", "admin_ai_models_Deepseek"),
         callback_button("🔤 Claude модель", "admin_ai_models_Claude")],
        [callback_button("🔤 Gemini модель", "admin_ai_models_Gemini"),
         callback_button("🔤 OpenAI модель", "admin_ai_models_OpenAI")],
        [callback_button("🔤 KIE модель", "admin_ai_models_KIE"),
         callback_button("🔤 OpenRouter модель", "admin_ai_models_OpenRouter")],
        [callback_button("🔤 Perplexity режим", "admin_ai_models_Perplexity"),
         callback_button(f"🗣️ Deepgram · {getattr(config, 'deepgram_model', None) or get_default_model(PROVIDER_DEEPGRAM, channel='transcription')}", "admin_ai_models_Deepgram")],
        [callback_button(f"🎙 Транскрипция: {transcription_label}", "admin_ai_select_transcription_provider"),
         callback_button(f"⏱ Лимит аудио: {config.max_voice_duration_sec}с", "admin_ai_set_audio_limit")],
        [callback_button(f"👁 Vision: {config.vision_provider}/{config.vision_model}", "admin_ai_select_vision_provider"),
         callback_button("🔤 Vision модель", "admin_ai_vision_models")],
        [callback_button(f"🔄👁 {'✅' if vision_fallback_enabled else '❌'} {_provider_model(vision_fb_provider, vision_fb_model)}", "admin_ai_toggle_vision_fallback")],
        [callback_button("🔤 Фолбэк Vision провайдер/модель", "admin_ai_vision_fallback_models")],
        [callback_button(_status_model_button("🎨 Генерация", img_gen_enabled, config.image_generation_provider, config.image_generation_model), "admin_ai_select_image_generation_provider")],
        [callback_button("🔤 Модель генерации", "admin_ai_image_generation_models")],
        [callback_button(_status_model_button("✏️ Редактирование", img_edit_enabled, config.image_edit_provider, config.image_edit_model), "admin_ai_select_image_edit_provider")],
        [callback_button("🔤 Модель редактирования", "admin_ai_image_edit_models")],
        [callback_button(f"🔄 {'✅' if fallback_enabled else '❌'} {_provider_model(config.fallback_provider, config.fallback_model or _fallback_model_for_provider(config, config.fallback_provider))}", "admin_ai_toggle_fallback")],
        [callback_button("🔤 Фолбэк провайдер/модель", "admin_ai_fallback_models")],
        [callback_button(f"📐 Контекст: первые {config.context_limit_first}", "admin_ai_set_context_first"),
         callback_button(f"📐 Последние {config.context_limit_recent}", "admin_ai_set_context_recent")],
        [callback_button("⚙️ Параметры активной модели", "admin_ai_model_settings")],
        [callback_button(f"🧠 Режим памяти: {memory_mode_label(memory)}", "admin_ai_cycle_memory_scope")],
        [callback_button("◀️ Назад", "admin_ai_settings")],
    ]
    return inline_keyboard(rows)


async def show_keys(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    current_memory_mode = normalize_memory_mode(config)
    img_gen_enabled = getattr(config, 'allow_image_generation', False)
    img_edit_enabled = getattr(config, 'allow_image_edit', False)
    fallback_enabled = getattr(config, 'allow_fallback', False)
    vision_fallback_enabled = getattr(config, 'allow_vision_fallback', False)
    img_gen = _provider_model(config.image_generation_provider, config.image_generation_model)
    img_edit = _provider_model(config.image_edit_provider, config.image_edit_model)
    fallback_info = _provider_model(config.fallback_provider, config.fallback_model or _fallback_model_for_provider(config, config.fallback_provider))
    vision_fb_provider = getattr(config, 'vision_fallback_provider', None)
    vision_fb_model = getattr(config, 'vision_fallback_model', None) or _vision_fallback_model_for_provider(config, vision_fb_provider)
    vision_fallback_info = _provider_model(vision_fb_provider, vision_fb_model)
    transcription_provider = getattr(config, "transcription_provider", None) or "None"
    if transcription_provider == PROVIDER_DEEPGRAM:
        transcription_info = f"{transcription_provider} · {getattr(config, 'deepgram_model', None) or get_default_model(PROVIDER_DEEPGRAM, channel='transcription')}"
    elif transcription_provider == "None":
        transcription_info = "выключено"
    else:
        transcription_info = transcription_provider
    kie_key = config.kie_api_key
    kie_threshold = config.kie_credit_alert_threshold
    text = (
        "<b>Провайдеры и модели</b>\n\n"
        f"<b>Deepseek:</b> <code>{_mask(config.deepseek_api_key)}</code>\n"
        f"<b>Claude:</b> <code>{_mask(config.claude_api_key)}</code>\n"
        f"<b>Gemini:</b> <code>{_mask(config.gemini_api_key)}</code>\n"
        f"<b>OpenAI:</b> <code>{_mask(config.openai_api_key)}</code>\n"
        f"🤖 <b>KIE:</b> <code>{_mask(kie_key)}</code> / порог: {kie_threshold}\n"
        f"<b>OpenRouter:</b> <code>{_mask(getattr(config, 'openrouter_api_key', None))}</code>\n"
        f"<b>Perplexity:</b> <code>{_mask(getattr(config, 'perplexity_api_key', None))}</code>\n"
        f"<b>Deepgram:</b> <code>{_mask(getattr(config, 'deepgram_api_key', None))}</code>\n"
        f"<b>Режим памяти:</b> {html.escape(memory_mode_label(current_memory_mode))}\n\n"
        f"🗣 <b>Распознавание:</b> {html.escape(transcription_info)}\n"
        f"🎨 <b>Генерация изображений:</b> {'✅' if img_gen_enabled else '❌'} / {html.escape(img_gen)}\n"
        f"✏️ <b>Редактирование изображений:</b> {'✅' if img_edit_enabled else '❌'} / {html.escape(img_edit)}\n"
        f"🔄 <b>Фолбэк:</b> {'✅' if fallback_enabled else '❌'} / {html.escape(fallback_info)}\n"
        f"🖼 <b>Фото (Vision) резерв:</b> {html.escape(vision_fallback_info) if vision_fallback_enabled else 'выключен'}\n\n"
        "Ниже доступны смена моделей, лимитов контекста и vision/audio-параметров."
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=_build_keys_keyboard(config),
    )


async def show_capability_providers(client: MaxApiClient, chat_id: int, channel: str) -> None:
    labels = {
        "transcription": "Выберите провайдера распознавания голосовых:",
        "vision": "Выберите провайдера анализа фото:",
        "image_gen": "Выберите провайдера генерации изображений:",
        "image_edit": "Выберите провайдера редактирования изображений:",
    }
    rows = [[callback_button(provider, f"admin_ai_choose_capability_{channel}_{provider}")] for provider in get_capability_providers(channel)]
    if channel in {"transcription", "image_gen", "image_edit"}:
        rows.append([callback_button("Выкл", f"admin_ai_choose_capability_{channel}_None")])
    rows.append([callback_button("⬅️ Назад", "admin_ai_keys")])
    await client.send_message(chat_id=chat_id, text=labels.get(channel, "Выберите провайдера:"), attachments=inline_keyboard(rows))


async def choose_capability_provider(client: MaxApiClient, chat_id: int, channel: str, provider_value: str) -> None:
    if provider_value == "None":
        async with async_session_maker() as session:
            config = await _ensure_session_config(session)
            if channel == "transcription":
                config.transcription_provider = "None"
            elif channel == "image_gen":
                config.image_generation_provider = "None"
                config.image_generation_model = ""
            elif channel == "image_edit":
                config.image_edit_provider = "None"
                config.image_edit_model = ""
            await session.commit()
        await show_keys(client, chat_id)
        return
    provider = canonical_provider_name(provider_value)
    if provider not in get_capability_providers(channel):
        await client.send_message(chat_id=chat_id, text="Провайдер не поддерживает этот канал.")
        return
    model = get_default_model(provider, channel=channel)
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        if channel == "transcription":
            config.transcription_provider = provider
            if provider == PROVIDER_DEEPGRAM:
                config.deepgram_model = model
            elif provider == PROVIDER_KIE:
                config.kie_transcription_model = model
        elif channel == "vision":
            config.vision_provider = provider
            config.vision_model = model
        elif channel == "image_gen":
            config.image_generation_provider = provider
            config.image_generation_model = model
        elif channel == "image_edit":
            config.image_edit_provider = provider
            config.image_edit_model = model
        await session.commit()
    await show_models(client, chat_id, provider) if channel == "chat" else await show_channel_models(client, chat_id, provider, channel)


async def show_channel_models(client: MaxApiClient, chat_id: int, provider: str, channel: str) -> None:
    models = list(get_selectable_models(provider, channel=channel))
    current = ""
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        if channel == "transcription":
            current = getattr(config, "deepgram_model", None) if provider == PROVIDER_DEEPGRAM else getattr(config, "kie_transcription_model", None)
        elif channel == "vision":
            current = config.vision_model
        elif channel == "image_gen":
            current = config.image_generation_model
        elif channel == "image_edit":
            current = config.image_edit_model
    rows = [[callback_button(f"✅ {m}" if m == current else m, f"admin_ai_set_channel_model_{channel}_{provider}_{m}")] for m in models]
    rows.append([callback_button("⬅️ Назад", "admin_ai_keys")])
    await client.send_message(chat_id=chat_id, text=f"Выберите модель для {provider}:", attachments=inline_keyboard(rows))


async def set_channel_model(client: MaxApiClient, chat_id: int, channel: str, provider: str, model: str) -> None:
    try:
        normalized = validate_model_selection(provider, model, channel=channel)
    except ModelUnavailableError:
        await _reject_model_selection(client, chat_id)
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        if channel == "transcription":
            config.transcription_provider = provider
            if provider == PROVIDER_DEEPGRAM:
                config.deepgram_model = normalized
            elif provider == PROVIDER_KIE:
                config.kie_transcription_model = normalized
        elif channel == "vision":
            config.vision_provider = provider
            config.vision_model = normalized
        elif channel == "image_gen":
            config.image_generation_provider = provider
            config.image_generation_model = normalized
        elif channel == "image_edit":
            config.image_edit_provider = provider
            config.image_edit_model = normalized
        await session.commit()
    await show_keys(client, chat_id)


def _active_chat_scope(config: AIConfig) -> tuple[str, str]:
    provider = canonical_provider_name(config.provider)
    field = MODEL_FIELDS.get(provider) or f"{provider.lower()}_model"
    model = getattr(config, field, None)
    if not model:
        model = get_default_model(provider, channel="chat")
    return provider, model


def _reasoning_label(value: str) -> str:
    return {
        REASONING_AUTO: "Авто",
        REASONING_NONE: "Выкл",
        REASONING_LOW: "Low",
        REASONING_HIGH: "High",
        REASONING_MAX: "Max",
    }.get(value, "Авто")


async def show_model_settings(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        provider, model = _active_chat_scope(config)
        settings = await resolve_model_settings(session, provider, model, "chat", config=config)
    caps = get_generation_capabilities(provider, model, "chat")
    max_label = "Авто" if settings.max_output_tokens is None else str(settings.max_output_tokens)
    text = (
        "<b>Параметры модели</b>\n\n"
        f"Провайдер: <b>{html.escape(provider)}</b>\n"
        f"Модель: <code>{html.escape(model)}</code>\n\n"
        f"📏 Max tokens: <b>{max_label}</b>\n"
    )
    if caps.reasoning_effort:
        text += f"🧠 Reasoning: <b>{_reasoning_label(settings.reasoning_effort)}</b>\n"
    if caps.temperature or provider == PROVIDER_DEEPSEEK:
        temp = "не применяется" if provider == PROVIDER_DEEPSEEK and settings.reasoning_effort != REASONING_NONE else ("Авто" if settings.temperature is None else str(settings.temperature))
        text += f"🌡 Temperature: <b>{temp}</b>\n"
    rows = [
        [callback_button("📏 Max tokens", "admin_ai_model_max_tokens")],
    ]
    if caps.reasoning_effort:
        rows.append([callback_button("🧠 Reasoning", "admin_ai_model_reasoning")])
    if caps.temperature and (provider != PROVIDER_DEEPSEEK or settings.reasoning_effort == REASONING_NONE):
        rows.append([callback_button("🌡 Temperature", "admin_ai_model_temperature")])
    rows.extend([
        [callback_button("🤖 Выбрать модель", "admin_ai_model_choices")],
        [callback_button("🔑 API-ключ", "admin_ai_model_key")],
        [callback_button("⬅️ Назад", "admin_ai_keys")],
    ])
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(rows))


async def show_model_reasoning(client: MaxApiClient, chat_id: int) -> None:
    rows = [[callback_button(label, f"admin_ai_reasoning_{value}")] for value, label in (("auto", "Авто"), ("none", "Выкл"), ("low", "Low"), ("high", "High"), ("max", "Max"))]
    rows.append([callback_button("⬅️ Назад", "admin_ai_model_settings")])
    await client.send_message(chat_id=chat_id, text="<b>Reasoning DeepSeek</b>", attachments=inline_keyboard(rows))


async def save_model_reasoning(client: MaxApiClient, chat_id: int, value: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        provider, model = _active_chat_scope(config)
        try:
            value = validate_model_setting(provider, model, "reasoning_effort", value)
        except ValueError as exc:
            await client.send_message(chat_id=chat_id, text=str(exc))
            return
        row = await get_or_create_model_settings(session, provider, model, "chat", config=config)
        row.reasoning_effort = value
        await session.commit()
    await show_model_settings(client, chat_id)


async def start_model_max_tokens(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    config = await _get_config()
    provider, model = _active_chat_scope(config)
    await states.set(user_id, chat_id, "admin_ai_model_max_tokens", {"provider": provider, "model": model})
    await client.send_message(chat_id=chat_id, text="Введите число от 1 до лимита модели или «Авто».")


async def save_model_max_tokens(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    state = await states.get(user_id)
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        provider, model = _active_chat_scope(config)
        if state:
            provider = state.data.get("provider") or provider
            model = state.data.get("model") or model
        settings = await resolve_model_settings(session, provider, model, "chat", config=config)
        try:
            value = validate_model_setting(provider, model, "max_output_tokens", text.strip(), reasoning_effort=settings.reasoning_effort)
        except ValueError as exc:
            await client.send_message(chat_id=chat_id, text=str(exc))
            return
        row = await get_or_create_model_settings(session, provider, model, "chat", config=config)
        row.max_output_tokens = value
        await session.commit()
    await states.clear(user_id)
    await show_model_settings(client, chat_id)


async def start_model_temperature(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    config = await _get_config()
    provider, model = _active_chat_scope(config)
    async with async_session_maker() as session:
        settings = await resolve_model_settings(session, provider, model, "chat", config=config)
    if provider == PROVIDER_DEEPSEEK and settings.reasoning_effort != REASONING_NONE:
        await client.send_message(chat_id=chat_id, text="Температура не применяется в режиме Reasoning.")
        return
    await states.set(user_id, chat_id, "admin_ai_model_temperature", {"provider": provider, "model": model})
    await client.send_message(chat_id=chat_id, text="Введите число от 0.0 до 2.0 или «Авто».")


async def save_model_temperature(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    state = await states.get(user_id)
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        provider, model = _active_chat_scope(config)
        if state:
            provider = state.data.get("provider") or provider
            model = state.data.get("model") or model
        try:
            value = validate_model_setting(provider, model, "temperature", text.strip().replace(",", "."))
        except ValueError as exc:
            await client.send_message(chat_id=chat_id, text=str(exc))
            return
        row = await get_or_create_model_settings(session, provider, model, "chat", config=config)
        row.temperature = value
        await session.commit()
    await states.clear(user_id)
    await show_model_settings(client, chat_id)


async def start_set_key(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    provider: str,
    *,
    return_to_model_settings: bool = False,
) -> None:
    field = KEY_FIELDS.get(provider)
    if not field:
        await client.send_message(chat_id=chat_id, text="Неизвестный провайдер.")
        return
    data = {"field": field, "provider": provider}
    if return_to_model_settings:
        config = await _get_config()
        active_provider, active_model = _active_chat_scope(config)
        data.update({"model_settings": True, "provider": active_provider, "model": active_model})
    await states.set(user_id, chat_id, "admin_ai_set_key", data)
    await client.send_message(chat_id=chat_id, text=f"Введите новый API key для {provider}.")


async def save_key(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    field = snapshot.data.get("field") if snapshot else None
    if not field:
        await client.send_message(chat_id=chat_id, text="Состояние ключа потеряно.")
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        setattr(config, field, text.strip())
        await session.commit()
    model_settings = bool(snapshot.data.get("model_settings")) if snapshot else False
    await states.clear(user_id)
    if model_settings:
        await show_model_settings(client, chat_id)
    else:
        await show_keys(client, chat_id)


async def show_models(client: MaxApiClient, chat_id: int, provider: str) -> None:
    config = await _get_config()
    field = MODEL_FIELDS.get(provider)
    channel = "transcription" if canonical_provider_name(provider) == PROVIDER_DEEPGRAM else "chat"
    models = list(get_selectable_models(provider, channel=channel))
    current_model = getattr(config, field) if field else ""
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите модель для {provider}.",
        attachments=admin_ai_model_selection_keyboard(provider, current_model or "", models),
    )


async def set_model(client: MaxApiClient, chat_id: int, provider: str, model_name: str) -> None:
    provider = canonical_provider_name(provider)
    field = MODEL_FIELDS.get(provider)
    if not field:
        await _reject_model_selection(client, chat_id)
        return
    channel = "transcription" if provider == PROVIDER_DEEPGRAM else "chat"
    try:
        normalized_model = validate_model_selection(provider, model_name, channel=channel)
    except ModelUnavailableError:
        await _reject_model_selection(client, chat_id)
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        setattr(config, field, normalized_model)
        if provider == PROVIDER_DEEPGRAM:
            config.transcription_provider = PROVIDER_DEEPGRAM
        await session.commit()
    await show_model_settings(client, chat_id) if channel == "chat" else await show_keys(client, chat_id)


async def toggle_transcription(client: MaxApiClient, chat_id: int) -> None:
    await show_capability_providers(client, chat_id, "transcription")


async def toggle_vision(client: MaxApiClient, chat_id: int) -> None:
    await show_capability_providers(client, chat_id, "vision")


async def show_vision_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    provider = canonical_provider_name(config.vision_provider)
    models = list(get_selectable_models(provider, channel="vision"))
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите vision-модель для {provider}.",
        attachments=admin_ai_vision_models_keyboard(provider, config.vision_model, models),
    )


async def set_vision_model(
    client: MaxApiClient,
    chat_id: int,
    model_name: str,
    provider: str | None = None,
) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        intended_provider = canonical_provider_name(provider or config.vision_provider)
        try:
            normalized_model = validate_model_selection(
                intended_provider,
                model_name,
                channel="vision",
            )
        except ModelUnavailableError:
            await _reject_model_selection(client, chat_id)
            return
        config.vision_provider = intended_provider
        config.vision_model = normalized_model
        await session.commit()
    await show_keys(client, chat_id)


async def start_set_int(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, state_name: str, field: str, prompt: str) -> None:
    await states.set(user_id, chat_id, state_name, {"field": field})
    await client.send_message(chat_id=chat_id, text=prompt)


async def save_int(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str, *, minimum: int = 0) -> None:
    try:
        value = int(text.strip())
        if value < minimum:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text=f"Введите целое число не меньше {minimum}.")
        return
    snapshot = await states.get(user_id)
    field = snapshot.data.get("field") if snapshot else None
    if not field:
        await client.send_message(chat_id=chat_id, text="Состояние настройки потеряно.")
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        setattr(config, field, value)
        await session.commit()
    await states.clear(user_id)
    await show_keys(client, chat_id)


async def start_set_temperature(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await start_model_temperature(client, states, chat_id, user_id)


async def save_temperature(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    await save_model_temperature(client, states, chat_id, user_id, text)


async def cycle_memory_scope(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        new_mode = next_memory_mode(normalize_memory_mode(config))
        config.memory_mode = new_mode
        config.preserve_topic_context = new_mode != MEMORY_MODE_RESET
        await session.commit()
    await show_keys(client, chat_id)


async def start_edit_system_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    config = await _get_config()
    preview = (config.system_prompt or "Не задан.")[:3000]
    await states.set(user_id, chat_id, "admin_ai_set_system_prompt", {})
    from ..time_utils import format_msk
    time_str = ""
    if config.system_prompt and config.system_prompt_updated_at:
        time_str = f"<b>Загружен:</b> {format_msk(config.system_prompt_updated_at, '%d.%m.%y %H:%M')}\n\n"
    elif config.system_prompt:
        time_str = "<b>Загружен:</b> —\n\n"
    await client.send_message(
        chat_id=chat_id,
        text=f"<b>Текущий системный промпт</b>\n<pre><code>{html.escape(preview)}</code></pre>\n{time_str}Отправьте новый текст промпта сообщением или загрузите <b>.txt/.md</b> файл.",
        attachments=_prompt_input_keyboard("admin_ai_cancel_system_prompt"),
    )


async def start_edit_global_prompt_appendix(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    config = await _get_config()
    preview = (config.shared_prompt_block or "Не задан.")[:3000]
    await states.set(user_id, chat_id, "admin_ai_set_global_prompt_appendix", {})
    await client.send_message(
        chat_id=chat_id,
        text=f"<b>Общий блок для всех промптов</b>\n<pre><code>{html.escape(preview)}</code></pre>\nОтправьте новый текст сообщением или загрузите <b>.txt/.md</b> файл. Для очистки отправьте <code>-</code>.",
        attachments=_prompt_input_keyboard("admin_ai_cancel_global_prompt_appendix"),
    )


async def _send_prompt_text_file(client: MaxApiClient, chat_id: int, filename: str, content: str) -> None:
    safe_name = Path(filename).name or "system_prompt.txt"
    try:
        await client.send_text_file(chat_id=chat_id, filename=safe_name, content=content, caption=f"📥 {safe_name}")
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"Не удалось отправить файл: {html.escape(str(exc))}")


async def download_system_prompt(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    content = config.system_prompt or ""
    filename = config.prompt_filename or "system_prompt.txt"
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    await _send_prompt_text_file(client, chat_id, filename, content)


async def download_global_prompt_appendix(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    await _send_prompt_text_file(client, chat_id, "shared_prompt_block.txt", config.shared_prompt_block or "")


async def save_system_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        from datetime import datetime
        config.system_prompt = text
        config.prompt_mode = "text"
        config.prompt_filename = None
        config.system_prompt_updated_at = datetime.utcnow()
        await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def save_global_prompt_appendix(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    value = None if text.strip() == "-" else text
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.shared_prompt_block = value
        await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def toggle_image_generation(client: MaxApiClient, chat_id: int) -> None:
    await show_capability_providers(client, chat_id, "image_gen")


async def toggle_image_edit(client: MaxApiClient, chat_id: int) -> None:
    await show_capability_providers(client, chat_id, "image_edit")


async def show_image_generation_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    current_model = config.image_generation_model or ""
    provider = canonical_provider_name(config.image_generation_provider or "OpenAI")
    models = list(get_selectable_models(provider, channel="image_gen"))
    rows = [[callback_button(f"{'✅ ' if m == current_model else ''}{m}", f"admin_ai_set_image_gen_model_{provider}_{m}")] for m in models]
    rows.append([callback_button("◀️ Назад", "admin_ai_keys")])
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите модель генерации изображений для {provider}.",
        attachments=inline_keyboard(rows),
    )


async def set_image_generation_model(
    client: MaxApiClient,
    chat_id: int,
    model_name: str,
    provider: str | None = None,
) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        intended_provider = canonical_provider_name(
            provider or getattr(config, "image_generation_provider", None) or "OpenAI"
        )
        try:
            normalized_model = validate_model_selection(
                intended_provider,
                model_name,
                channel="image_gen",
            )
        except ModelUnavailableError:
            await _reject_model_selection(client, chat_id)
            return
        config.image_generation_provider = intended_provider
        config.image_generation_model = normalized_model
        await session.commit()
    await show_keys(client, chat_id)


async def show_image_edit_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    current_model = config.image_edit_model or ""
    provider = canonical_provider_name(config.image_edit_provider or "KIE")
    models = list(get_selectable_models(provider, channel="image_edit"))
    rows = [[callback_button(f"{'✅ ' if m == current_model else ''}{m}", f"admin_ai_set_image_edit_model_{provider}_{m}")] for m in models]
    rows.append([callback_button("◀️ Назад", "admin_ai_keys")])
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите модель редактирования изображений для {provider}.",
        attachments=inline_keyboard(rows),
    )


async def set_image_edit_model(
    client: MaxApiClient,
    chat_id: int,
    model_name: str,
    provider: str | None = None,
) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        intended_provider = canonical_provider_name(
            provider or getattr(config, "image_edit_provider", None) or "KIE"
        )
        try:
            normalized_model = validate_model_selection(
                intended_provider,
                model_name,
                channel="image_edit",
            )
        except ModelUnavailableError:
            await _reject_model_selection(client, chat_id)
            return
        config.image_edit_provider = intended_provider
        config.image_edit_model = normalized_model
        await session.commit()
    await show_keys(client, chat_id)


async def toggle_fallback(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.allow_fallback = not bool(config.allow_fallback)
        await session.commit()
    await show_keys(client, chat_id)


async def show_fallback_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    current_provider = config.fallback_provider or ""
    rows = [
        [callback_button(f"{'✅ ' if p == current_provider else ''}{p}", f"admin_ai_set_fallback_provider_{p}")]
        for p in FALLBACK_PROVIDERS
    ]
    rows.append([callback_button("◀️ Назад", "admin_ai_keys")])
    await client.send_message(
        chat_id=chat_id,
        text="Выберите провайдер фолбэка.",
        attachments=inline_keyboard(rows),
    )


async def set_fallback_provider(client: MaxApiClient, chat_id: int, provider: str) -> None:
    provider = canonical_provider_name(provider)
    try:
        default_model = get_default_model(provider, channel="fallback")
        normalized_model = validate_model_selection(provider, default_model, channel="fallback")
    except ModelUnavailableError:
        await _reject_model_selection(client, chat_id)
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.fallback_provider = provider
        config.fallback_model = normalized_model
        await session.commit()
        current_model = config.fallback_model
    models = list(get_selectable_models(provider, channel="fallback"))
    rows = [[callback_button(f"{'✅ ' if m == current_model else ''}{m}", f"admin_ai_save_fallback_{provider}_{m}")] for m in models]
    rows.append([callback_button("◀️ Назад", "admin_ai_fallback_models")])
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите модель фолбэка для {provider}.",
        attachments=inline_keyboard(rows),
    )


async def save_fallback_model(client: MaxApiClient, chat_id: int, provider: str, model_name: str) -> None:
    provider = canonical_provider_name(provider)
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        try:
            normalized_model = validate_model_selection(provider, model_name, channel="fallback")
        except ModelUnavailableError:
            await _reject_model_selection(client, chat_id)
            return
        config.fallback_provider = provider
        config.fallback_model = normalized_model
        await session.commit()
    await show_keys(client, chat_id)


async def toggle_vision_fallback(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.allow_vision_fallback = not bool(config.allow_vision_fallback)
        await session.commit()
    await show_keys(client, chat_id)


async def show_vision_fallback_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    current_provider = config.vision_fallback_provider or ""
    rows = [
        [callback_button(f"{'✅ ' if p == current_provider else ''}{p}", f"admin_ai_set_vision_fallback_provider_{p}")]
        for p in VISION_FALLBACK_PROVIDERS
    ]
    rows.append([callback_button("◀️ Назад", "admin_ai_keys")])
    await client.send_message(
        chat_id=chat_id,
        text="Выберите провайдер фолбэка для Vision.",
        attachments=inline_keyboard(rows),
    )


async def set_vision_fallback_provider(client: MaxApiClient, chat_id: int, provider: str) -> None:
    provider = canonical_provider_name(provider)
    try:
        default_model = get_default_model(provider, channel="vision_fallback")
        normalized_model = validate_model_selection(provider, default_model, channel="vision_fallback")
    except ModelUnavailableError:
        await _reject_model_selection(client, chat_id)
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.vision_fallback_provider = provider
        config.vision_fallback_model = normalized_model
        await session.commit()
        current_model = config.vision_fallback_model
    models = list(get_selectable_models(provider, channel="vision_fallback"))
    rows = [[callback_button(f"{'✅ ' if m == current_model else ''}{m}", f"admin_ai_save_vision_fallback_{provider}_{m}")] for m in models]
    rows.append([callback_button("◀️ Назад", "admin_ai_vision_fallback_models")])
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите модель фолбэка Vision для {provider}.",
        attachments=inline_keyboard(rows),
    )


async def save_vision_fallback_model(client: MaxApiClient, chat_id: int, provider: str, model_name: str) -> None:
    provider = canonical_provider_name(provider)
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        try:
            normalized_model = validate_model_selection(provider, model_name, channel="vision_fallback")
        except ModelUnavailableError:
            await _reject_model_selection(client, chat_id)
            return
        config.vision_fallback_provider = provider
        config.vision_fallback_model = normalized_model
        await session.commit()
    await show_keys(client, chat_id)


async def cancel_prompt_input(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def start_set_kie_threshold(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_ai_set_kie_threshold", {})
    await client.send_message(
        chat_id=chat_id,
        text="Введите порог остатка кредитов KIE для оповещения (0 = выключено, например 100.0):",
    )


async def save_kie_threshold(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        value = float(text.strip().replace(",", "."))
        if value < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите число не меньше 0.")
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.kie_credit_alert_threshold = value
        await session.commit()
    await states.clear(user_id)
    await show_keys(client, chat_id)


async def start_set_kie_field(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, field: str) -> None:
    await states.set(user_id, chat_id, "admin_ai_set_key", {"field": field, "provider": "KIE_extra"})
    label = KIE_EXTRA_FIELDS.get(field, field)
    await client.send_message(chat_id=chat_id, text=f"Введите новое значение для {label}:")
