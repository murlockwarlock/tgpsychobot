from __future__ import annotations

import html

from ..api import MaxApiClient
from ..formatting import split_text
from ..keyboards import admin_ai_model_selection_keyboard, admin_ai_settings_keyboard, admin_ai_vision_models_keyboard, callback_button, inline_keyboard
from ..legacy import AIConfig, async_session_maker
from ..storage import StateStore
from memory_mode import memory_mode_label, next_memory_mode, normalize_memory_mode


PROVIDER_MODELS = {
    "Deepseek": ["deepseek-chat", "deepseek-coder"],
    "Claude": ["claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805", "claude-haiku-4-5-20251001", "claude-3-haiku-20240307"],
    "Gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "KIE": ["gemini-3-flash", "flux-dev", "flux-schnell"],
}

VISION_MODELS = {
    "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-3-flash-preview"],
    "OpenAI": ["gpt-4o", "gpt-4o-mini"],
}

IMAGE_GEN_MODELS = {
    "Gemini": ["imagen-4.0-generate-001", "gemini-2.0-flash-preview-image-generation"],
    "OpenAI": ["dall-e-3", "dall-e-2"],
    "KIE": ["gemini-3-flash", "flux-dev", "flux-schnell"],
}

IMAGE_EDIT_MODELS = {
    "Gemini": ["gemini-3-pro-image-preview", "gemini-2.0-flash"],
    "KIE": ["gemini-3-flash", "flux-dev"],
}

FALLBACK_PROVIDERS = ["OpenAI", "Gemini", "Claude", "Deepseek"]

KEY_FIELDS = {
    "Deepseek": "deepseek_api_key",
    "Claude": "claude_api_key",
    "Gemini": "gemini_api_key",
    "OpenAI": "openai_api_key",
    "KIE": "kie_api_key",
}

MODEL_FIELDS = {
    "Deepseek": "deepseek_model",
    "Claude": "claude_model",
    "Gemini": "gemini_model",
    "OpenAI": "openai_model",
    "KIE": "kie_model",
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
        return value
    return f"{value[:4]}...{value[-4:]}"


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


def _build_keys_keyboard(config) -> list:
    img_gen_enabled = getattr(config, 'allow_image_generation', False)
    img_edit_enabled = getattr(config, 'allow_image_edit', False)
    fallback_enabled = getattr(config, 'allow_fallback', False)
    memory = normalize_memory_mode(config)
    rows = [
        [callback_button(f"Deepseek: {_mask(config.deepseek_api_key)}", "admin_ai_key_Deepseek"),
         callback_button(f"Claude: {_mask(config.claude_api_key)}", "admin_ai_key_Claude")],
        [callback_button(f"Gemini: {_mask(config.gemini_api_key)}", "admin_ai_key_Gemini"),
         callback_button(f"OpenAI: {_mask(config.openai_api_key)}", "admin_ai_key_OpenAI")],
        [callback_button(f"KIE: {_mask(config.kie_api_key)}", "admin_ai_key_KIE")],
        [callback_button("🔤 Deepseek модель", "admin_ai_models_Deepseek"),
         callback_button("🔤 Claude модель", "admin_ai_models_Claude")],
        [callback_button("🔤 Gemini модель", "admin_ai_models_Gemini"),
         callback_button("🔤 OpenAI модель", "admin_ai_models_OpenAI")],
        [callback_button("🔤 KIE модель", "admin_ai_models_KIE")],
        [callback_button(f"🎙 Транскрипция: {config.transcription_provider or 'OpenAI'}", "admin_ai_toggle_transcription"),
         callback_button(f"⏱ Лимит аудио: {config.max_voice_duration_sec}с", "admin_ai_set_audio_limit")],
        [callback_button(f"👁 Vision: {config.vision_provider}/{config.vision_model}", "admin_ai_toggle_vision"),
         callback_button("🔤 Vision модель", "admin_ai_vision_models")],
        [callback_button(f"🎨 Ген. изображений: {'✅' if img_gen_enabled else '❌'}", "admin_ai_toggle_image_generation"),
         callback_button("🔤 Модель ген.", "admin_ai_image_generation_models")],
        [callback_button(f"✏️ Ред. изображений: {'✅' if img_edit_enabled else '❌'}", "admin_ai_toggle_image_edit"),
         callback_button("🔤 Модель ред.", "admin_ai_image_edit_models")],
        [callback_button(f"🔄 Фолбэк: {'✅' if fallback_enabled else '❌'}", "admin_ai_toggle_fallback"),
         callback_button("🔤 Фолбэк провайдер/модель", "admin_ai_fallback_models")],
        [callback_button(f"📊 Порог KIE: {config.kie_credit_alert_threshold}", "admin_ai_set_kie_threshold")],
        [callback_button(f"📐 Контекст: первые {config.context_limit_first}", "admin_ai_set_context_first"),
         callback_button(f"📐 Последние {config.context_limit_recent}", "admin_ai_set_context_recent")],
        [callback_button(f"🌡 Температура: {config.temperature}", "admin_ai_set_temperature")],
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
    img_gen = config.image_generation_model or "не выбрана"
    img_edit = f"{config.image_edit_provider}/{config.image_edit_model}"
    fallback_info = f"{config.fallback_provider or 'нет'}/{config.fallback_model or 'нет'}"
    kie_key = config.kie_api_key
    kie_threshold = config.kie_credit_alert_threshold
    text = (
        "<b>Ключи и модели ИИ</b>\n\n"
        f"<b>Deepseek:</b> <code>{_mask(config.deepseek_api_key)}</code>\n"
        f"<b>Claude:</b> <code>{_mask(config.claude_api_key)}</code>\n"
        f"<b>Gemini:</b> <code>{_mask(config.gemini_api_key)}</code>\n"
        f"<b>OpenAI:</b> <code>{_mask(config.openai_api_key)}</code>\n"
        f"<b>Режим памяти:</b> {html.escape(memory_mode_label(current_memory_mode))}\n\n"
        f"🎨 <b>Генерация изображений:</b> {'✅' if img_gen_enabled else '❌'} / {html.escape(img_gen)}\n"
        f"✏️ <b>Редактирование изображений:</b> {'✅' if img_edit_enabled else '❌'} / {html.escape(img_edit)}\n"
        f"🔄 <b>Фолбэк:</b> {'✅' if fallback_enabled else '❌'} / {html.escape(fallback_info)}\n"
        f"🤖 <b>KIE:</b> <code>{_mask(kie_key)}</code> / порог: {kie_threshold}\n\n"
        "Ниже доступны смена моделей, лимитов контекста и vision/audio-параметров."
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=_build_keys_keyboard(config),
    )


async def start_set_key(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, provider: str) -> None:
    field = KEY_FIELDS.get(provider)
    if not field:
        await client.send_message(chat_id=chat_id, text="Неизвестный провайдер.")
        return
    await states.set(user_id, chat_id, "admin_ai_set_key", {"field": field, "provider": provider})
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
    await states.clear(user_id)
    await show_keys(client, chat_id)


async def show_models(client: MaxApiClient, chat_id: int, provider: str) -> None:
    config = await _get_config()
    field = MODEL_FIELDS.get(provider)
    models = PROVIDER_MODELS.get(provider, [])
    current_model = getattr(config, field) if field else ""
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите модель для {provider}.",
        attachments=admin_ai_model_selection_keyboard(provider, current_model or "", models),
    )


async def set_model(client: MaxApiClient, chat_id: int, provider: str, model_name: str) -> None:
    field = MODEL_FIELDS.get(provider)
    if not field:
        await client.send_message(chat_id=chat_id, text="Неизвестный провайдер.")
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        setattr(config, field, model_name)
        await session.commit()
    await show_keys(client, chat_id)


async def toggle_transcription(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        current = config.transcription_provider
        if current == "OpenAI":
            config.transcription_provider = "Gemini"
        elif current == "Gemini":
            config.transcription_provider = "None"
        else:
            config.transcription_provider = "OpenAI"
        await session.commit()
    await show_keys(client, chat_id)


async def toggle_vision(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        if config.vision_provider == "OpenAI":
            config.vision_provider = "Gemini"
            config.vision_model = "gemini-3-flash-preview"
        else:
            config.vision_provider = "OpenAI"
            config.vision_model = "gpt-4o"
        await session.commit()
    await show_keys(client, chat_id)


async def show_vision_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    models = VISION_MODELS.get(config.vision_provider, [])
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите vision-модель для {config.vision_provider}.",
        attachments=admin_ai_vision_models_keyboard(config.vision_model, models),
    )


async def set_vision_model(client: MaxApiClient, chat_id: int, model_name: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.vision_model = model_name
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
    await states.set(user_id, chat_id, "admin_ai_set_temperature", {})
    await client.send_message(chat_id=chat_id, text="Введите температуру от 0.0 до 2.0.")


async def save_temperature(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        value = float(text.strip().replace(",", "."))
        if not 0.0 <= value <= 2.0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите число от 0.0 до 2.0.")
        return
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.temperature = value
        await session.commit()
    await states.clear(user_id)
    await show_keys(client, chat_id)


async def cycle_memory_scope(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        new_mode = next_memory_mode(normalize_memory_mode(config))
        config.memory_scope = new_mode
        config.preserve_topic_context = new_mode != "reset_on_switch"
        await session.commit()
    await show_keys(client, chat_id)


async def start_edit_system_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    config = await _get_config()
    preview = (config.system_prompt or "Не задан.")[:3000]
    await states.set(user_id, chat_id, "admin_ai_set_system_prompt", {})
    await client.send_message(
        chat_id=chat_id,
        text=f"<b>Текущий системный промпт</b>\n<pre><code>{html.escape(preview)}</code></pre>\nОтправьте новый текст промпта сообщением или загрузите <b>.txt/.md</b> файл.",
    )


async def start_edit_global_prompt_appendix(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    config = await _get_config()
    preview = (config.global_prompt_appendix or "Не задан.")[:3000]
    await states.set(user_id, chat_id, "admin_ai_set_global_prompt_appendix", {})
    await client.send_message(
        chat_id=chat_id,
        text=f"<b>Общий блок для всех промптов</b>\n<pre><code>{html.escape(preview)}</code></pre>\nОтправьте новый текст сообщением или загрузите <b>.txt/.md</b> файл. Для очистки отправьте <code>-</code>.",
    )


async def _send_prompt_text_file(client: MaxApiClient, chat_id: int, filename: str, content: str) -> None:
    try:
        await client.send_text_file(chat_id=chat_id, filename=filename, content=content, caption=f"📥 {filename}")
    except Exception:
        escaped = html.escape(content or "Промпт пуст.")
        for chunk in split_text(f"<pre><code>{escaped}</code></pre>", 3900)[:10]:
            await client.send_message(chat_id=chat_id, text=chunk)


async def download_system_prompt(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    content = config.system_prompt or ""
    filename = config.prompt_filename or "system_prompt.txt"
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    await _send_prompt_text_file(client, chat_id, filename, content)


async def download_global_prompt_appendix(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    await _send_prompt_text_file(client, chat_id, "global_prompt_appendix.txt", config.global_prompt_appendix or "")


async def save_system_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.system_prompt = text
        config.prompt_mode = "text"
        config.prompt_filename = None
        await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def save_global_prompt_appendix(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    value = None if text.strip() == "-" else text
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.global_prompt_appendix = value
        await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def toggle_image_generation(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        current = getattr(config, 'allow_image_generation', False)
        setattr(config, 'allow_image_generation', not current)
        await session.commit()
    await show_keys(client, chat_id)


async def toggle_image_edit(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        current = getattr(config, 'allow_image_edit', False)
        setattr(config, 'allow_image_edit', not current)
        await session.commit()
    await show_keys(client, chat_id)


async def show_image_generation_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    current_model = config.image_generation_model or ""
    all_models = [m for models in IMAGE_GEN_MODELS.values() for m in models]
    rows = [[callback_button(f"{'✅ ' if m == current_model else ''}{m}", f"admin_ai_set_image_gen_model_{m}")] for m in all_models]
    rows.append([callback_button("◀️ Назад", "admin_ai_keys")])
    await client.send_message(
        chat_id=chat_id,
        text="Выберите модель генерации изображений.",
        attachments=inline_keyboard(rows),
    )


async def set_image_generation_model(client: MaxApiClient, chat_id: int, model_name: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.image_generation_model = model_name
        await session.commit()
    await show_keys(client, chat_id)


async def show_image_edit_models(client: MaxApiClient, chat_id: int) -> None:
    config = await _get_config()
    current_model = config.image_edit_model or ""
    all_models = [m for models in IMAGE_EDIT_MODELS.values() for m in models]
    rows = [[callback_button(f"{'✅ ' if m == current_model else ''}{m}", f"admin_ai_set_image_edit_model_{m}")] for m in all_models]
    rows.append([callback_button("◀️ Назад", "admin_ai_keys")])
    await client.send_message(
        chat_id=chat_id,
        text="Выберите модель редактирования изображений.",
        attachments=inline_keyboard(rows),
    )


async def set_image_edit_model(client: MaxApiClient, chat_id: int, model_name: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.image_edit_model = model_name
        await session.commit()
    await show_keys(client, chat_id)


async def toggle_fallback(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        current = getattr(config, 'allow_fallback', False)
        setattr(config, 'allow_fallback', not current)
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
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.fallback_provider = provider
        await session.commit()
    models = PROVIDER_MODELS.get(provider, [])
    rows = [[callback_button(m, f"admin_ai_save_fallback_{provider}_{m}")] for m in models]
    rows.append([callback_button("◀️ Назад", "admin_ai_fallback_models")])
    await client.send_message(
        chat_id=chat_id,
        text=f"Выберите модель фолбэка для {provider}.",
        attachments=inline_keyboard(rows),
    )


async def save_fallback_model(client: MaxApiClient, chat_id: int, provider: str, model_name: str) -> None:
    async with async_session_maker() as session:
        config = await _ensure_session_config(session)
        config.fallback_provider = provider
        config.fallback_model = model_name
        await session.commit()
    await show_keys(client, chat_id)


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
