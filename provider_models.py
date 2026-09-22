from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import re
from typing import Any


# ==========================================
# 1. PROVIDER CANONICAL IDENTIFIERS & DEFAULTS
# ==========================================

PROVIDER_GEMINI = "Gemini"
PROVIDER_CLAUDE = "Claude"
PROVIDER_OPENAI = "OpenAI"
PROVIDER_DEEPSEEK = "Deepseek"
PROVIDER_KIE = "KIE"
PROVIDER_OPENROUTER = "OpenRouter"
PROVIDER_PERPLEXITY = "Perplexity"
PROVIDER_DEEPGRAM = "Deepgram"

ALL_PROVIDERS = (
    PROVIDER_GEMINI,
    PROVIDER_CLAUDE,
    PROVIDER_OPENAI,
    PROVIDER_DEEPSEEK,
    PROVIDER_KIE,
    PROVIDER_OPENROUTER,
    PROVIDER_PERPLEXITY,
)

ALL_TRANSCRIPTION_PROVIDERS = (
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_DEEPGRAM,
)

DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)
DEEPSEEK_LEGACY_MODELS = (
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-coder",
)
DEEPSEEK_CHAT_MAX_TOKENS = 65536
OPENAI_CHAT_MAX_TOKENS = 16384
CLAUDE_CHAT_MAX_TOKENS = 16384
GEMINI_CHAT_MAX_TOKENS = 16384
KIE_CHAT_MAX_TOKENS = 4096
PERPLEXITY_CHAT_MAX_TOKENS = 128000
DEEPGRAM_DEFAULT_MODEL = "nova-3"
PERPLEXITY_MODE_OUTPUT_LIMITS = {
    "fast": 8192,
    "low": 32768,
    "medium": 128000,
}

KIE_DEFAULT_CHAT_MODEL = "gemini-3-flash"

PROVIDER_DEFAULT_MODELS = {
    PROVIDER_GEMINI: "gemini-3.7-flash",
    PROVIDER_CLAUDE: "claude-sonnet-5",
    PROVIDER_OPENAI: "gpt-5.6-terra",
    PROVIDER_DEEPSEEK: DEEPSEEK_DEFAULT_MODEL,
    PROVIDER_KIE: KIE_DEFAULT_CHAT_MODEL,
    PROVIDER_OPENROUTER: "openai/gpt-5.6-terra",
    PROVIDER_PERPLEXITY: "low",
    PROVIDER_DEEPGRAM: DEEPGRAM_DEFAULT_MODEL,
}


@dataclass(frozen=True)
class OpenRouterModelSpec:
    model_id: str
    friendly_name: str
    text: bool
    vision: bool
    audio_input: bool
    context_limit: int
    output_limit: int
    status: str = "active"


OPENROUTER_MODEL_SPECS: dict[str, OpenRouterModelSpec] = {
    "openai/gpt-5.6-terra": OpenRouterModelSpec("openai/gpt-5.6-terra", "OpenAI GPT-5.6 Terra", True, True, False, 1050000, 128000),
    "openai/gpt-5.6-sol": OpenRouterModelSpec("openai/gpt-5.6-sol", "OpenAI GPT-5.6 Sol", True, True, False, 1050000, 128000),
    "openai/gpt-5.6-luna-pro": OpenRouterModelSpec("openai/gpt-5.6-luna-pro", "OpenAI GPT-5.6 Luna Pro", True, True, False, 1050000, 128000),
    "google/gemini-3.7-flash": OpenRouterModelSpec("google/gemini-3.7-flash", "Google Gemini 3.7 Flash", True, True, True, 1048576, 65536),
    "google/gemini-3.8-flash": OpenRouterModelSpec("google/gemini-3.8-flash", "Google Gemini 3.8 Flash", True, True, True, 1048576, 65536),
    "google/gemini-3.1-pro-preview": OpenRouterModelSpec("google/gemini-3.1-pro-preview", "Google Gemini 3.1 Pro Preview", True, True, True, 1048576, 65536),
    "anthropic/claude-sonnet-4.6": OpenRouterModelSpec("anthropic/claude-sonnet-4.6", "Anthropic Claude Sonnet 4.6", True, True, False, 1000000, 128000),
    "anthropic/claude-opus-4.6": OpenRouterModelSpec("anthropic/claude-opus-4.6", "Anthropic Claude Opus 4.6", True, True, False, 1000000, 128000),
    "anthropic/claude-haiku-4.5": OpenRouterModelSpec("anthropic/claude-haiku-4.5", "Anthropic Claude Haiku 4.5", True, True, False, 200000, 64000),
    "x-ai/grok-4.7": OpenRouterModelSpec("x-ai/grok-4.7", "xAI Grok 4.7", True, True, False, 500000, 450000),
    "x-ai/grok-4.6": OpenRouterModelSpec("x-ai/grok-4.6", "xAI Grok 4.6", True, True, False, 500000, 450000),
    "deepseek/deepseek-v3.2": OpenRouterModelSpec("deepseek/deepseek-v3.2", "DeepSeek V3.2", True, False, False, 163840, 65536),
    "qwen/qwen3-vl-235b-a22b-instruct": OpenRouterModelSpec("qwen/qwen3-vl-235b-a22b-instruct", "Qwen3 VL 235B Instruct", True, True, False, 262144, 32768),
    "moonshotai/kimi-k2.6": OpenRouterModelSpec("moonshotai/kimi-k2.6", "Kimi K2.6", True, True, False, 262144, 235929),
    "mistralai/mistral-medium-3-5": OpenRouterModelSpec("mistralai/mistral-medium-3-5", "Mistral Medium 3.5", True, True, False, 262144, 209715),
}

OPENROUTER_MODELS = tuple(OPENROUTER_MODEL_SPECS)
OPENROUTER_VISION_MODELS = tuple(spec.model_id for spec in OPENROUTER_MODEL_SPECS.values() if spec.vision)
PERPLEXITY_MODES = ("fast", "low", "medium")

DEFAULT_VISION_MODEL = "gemini-3.7-flash"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
GEMINI_IMAGE_MODELS = (
    DEFAULT_GEMINI_IMAGE_MODEL,
    "gemini-3-pro-image",
)
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_KIE_TRANSCRIPTION_MODEL = "elevenlabs/speech-to-text"


# ==========================================
# 2. MODEL STATUS TAXONOMY
# ==========================================

# Models permanently shut down by providers. MUST NEVER be transmitted over HTTP.
RETIRED_UPSTREAM_MODELS = frozenset({
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06",
    "claude-opus-4-1-20250805",
    "claude-3-haiku-20240307",
    "imagen-4.0-generate-001",
    "gemini-3-pro-image-preview",
})

# Models decommissioned from active UI selection by product policy and migrated in DB.
APP_DISABLED_OR_MIGRATED_MODELS = frozenset({
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-4.1",
    "claude-sonnet-4-5-20250929",
    "gpt-image-1.5",
})


# ==========================================
# 3. SELECTABLE CATALOGS (SINGLE SOURCE OF TRUTH)
# ==========================================

# Active primary chat models offered in Telegram and MAX admin settings
SELECTABLE_CHAT_MODELS: dict[str, tuple[str, ...]] = {
    PROVIDER_GEMINI: (
        "gemini-3.7-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ),
    PROVIDER_CLAUDE: (
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5-20251001",
    ),
    PROVIDER_OPENAI: (
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
    ),
    PROVIDER_DEEPSEEK: DEEPSEEK_MODELS,
    PROVIDER_KIE: (
        "gemini-3-flash",
        "gemini-2.5-flash",
        "claude-haiku-4-5",
        "grok-4-3",
        "gemini-3-7-flash",
        "gpt-5-6-luna",
    ),
    PROVIDER_OPENROUTER: OPENROUTER_MODELS,
    PROVIDER_PERPLEXITY: PERPLEXITY_MODES,
}

# Active fallback models offered in Telegram and MAX admin settings
SELECTABLE_FALLBACK_MODELS: dict[str, tuple[str, ...]] = {
    PROVIDER_GEMINI: (
        "gemini-3.7-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ),
    PROVIDER_CLAUDE: (
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5-20251001",
    ),
    PROVIDER_OPENAI: (
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
    ),
    PROVIDER_DEEPSEEK: DEEPSEEK_MODELS,
    PROVIDER_KIE: (
        "gemini-3-flash",
        "gemini-2.5-flash",
        "claude-haiku-4-5",
        "grok-4-3",
        "gemini-3-7-flash",
        "gpt-5-6-luna",
    ),
    PROVIDER_OPENROUTER: OPENROUTER_MODELS,
    PROVIDER_PERPLEXITY: PERPLEXITY_MODES,
}

# Active vision models offered in Telegram and MAX admin settings
SELECTABLE_VISION_MODELS: dict[str, tuple[str, ...]] = {
    PROVIDER_GEMINI: (
        "gemini-3.7-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ),
    PROVIDER_CLAUDE: (
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5-20251001",
    ),
    PROVIDER_OPENAI: (
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
    ),
    PROVIDER_KIE: (
        "gemini-2.5-flash",
        "gemini-3-flash",
    ),
    PROVIDER_OPENROUTER: OPENROUTER_VISION_MODELS,
}

# Active direct Gemini and KIE text-to-image models.
SELECTABLE_IMAGE_GEN_MODELS: dict[str, tuple[str, ...]] = {
    PROVIDER_GEMINI: GEMINI_IMAGE_MODELS,
    PROVIDER_OPENAI: (
        DEFAULT_OPENAI_IMAGE_MODEL,
    ),
    PROVIDER_KIE: (
        "seedream/4.5-text-to-image",
        "bytedance/seedream-v4-text-to-image",
        "google/imagen4-fast",
        "google/imagen4-ultra",
    ),
}

# Active direct Gemini and KIE image edit models.
SELECTABLE_IMAGE_EDIT_MODELS: dict[str, tuple[str, ...]] = {
    PROVIDER_GEMINI: GEMINI_IMAGE_MODELS,
    PROVIDER_KIE: (
        "seedream/4.5-edit",
        "bytedance/seedream-v4-edit",
        "google/nano-banana-edit",
    ),
}

# Active audio transcription models
SELECTABLE_TRANSCRIPTION_MODELS: dict[str, tuple[str, ...]] = {
    PROVIDER_OPENAI: (
        DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    ),
    PROVIDER_GEMINI: (
        "gemini-3.7-flash",
    ),
    PROVIDER_KIE: (
        DEFAULT_KIE_TRANSCRIPTION_MODEL,
    ),
    PROVIDER_DEEPGRAM: (DEEPGRAM_DEFAULT_MODEL,),
}


# ==========================================
# 4. KIE PROTOCOL SPECS & HELPERS
# ==========================================

@dataclass(frozen=True)
class KIEChatModelSpec:
    model_id: str
    protocol: str
    endpoint_path: str
    stream: bool = False


KIE_CHAT_PROTOCOL_OPENAI = "openai_chat"
KIE_CHAT_PROTOCOL_ANTHROPIC = "anthropic_messages"
KIE_CHAT_PROTOCOL_RESPONSES = "responses"
KIE_CHAT_PROTOCOL_GEMINI = "gemini_native"

KIE_CHAT_MODELS = SELECTABLE_CHAT_MODELS[PROVIDER_KIE]

KIE_CHAT_MODEL_SPECS = {
    "gemini-3-flash": KIEChatModelSpec(
        "gemini-3-flash",
        KIE_CHAT_PROTOCOL_OPENAI,
        "/{model}/v1/chat/completions",
    ),
    "gemini-2.5-flash": KIEChatModelSpec(
        "gemini-2.5-flash",
        KIE_CHAT_PROTOCOL_OPENAI,
        "/{model}/v1/chat/completions",
    ),
    "claude-haiku-4-5": KIEChatModelSpec(
        "claude-haiku-4-5",
        KIE_CHAT_PROTOCOL_ANTHROPIC,
        "/claude/v1/messages",
    ),
    "grok-4-3": KIEChatModelSpec(
        "grok-4-3",
        KIE_CHAT_PROTOCOL_RESPONSES,
        "/grok/v1/responses",
        stream=True,
    ),
    "gemini-3-7-flash": KIEChatModelSpec(
        "gemini-3-7-flash",
        KIE_CHAT_PROTOCOL_GEMINI,
        "/gemini/v1/models/{model}:streamGenerateContent",
        stream=True,
    ),
    "gpt-5-6-luna": KIEChatModelSpec(
        "gpt-5-6-luna",
        KIE_CHAT_PROTOCOL_RESPONSES,
        "/codex/v1/responses",
    ),
}


def get_kie_chat_model_spec(model: str | None) -> KIEChatModelSpec:
    """Return the documented KIE protocol, retaining the legacy route for custom IDs."""
    normalized = (model or "").strip()
    spec = KIE_CHAT_MODEL_SPECS.get(normalized)
    if spec:
        return spec
    return KIEChatModelSpec(
        normalized,
        KIE_CHAT_PROTOCOL_OPENAI,
        "/{model}/v1/chat/completions",
    )


# ==========================================
# 5. SHARED AVAILABILITY & NORMALIZATION LOGIC
# ==========================================

def _canonical_provider_name(provider: str | None) -> str:
    p = (provider or "").strip()
    for canonical in (*ALL_PROVIDERS, *ALL_TRANSCRIPTION_PROVIDERS):
        if p.lower() == canonical.lower():
            return canonical
    return p


def is_retired_model(model: str | None) -> bool:
    """Return True if the model is known to be permanently shut down upstream."""
    if not model:
        return False
    normalized = model.strip()
    return normalized in RETIRED_UPSTREAM_MODELS


def canonical_provider_name(provider: str | None) -> str:
    """Return the canonical provider identifier used by the catalogs."""
    return _canonical_provider_name(provider)


class ModelUnavailableError(Exception):
    """Raised when a model is retired or unsupported for its capability."""
    pass


_SELECTABLE_MODEL_CATALOGS = {
    "chat": SELECTABLE_CHAT_MODELS,
    "fallback": SELECTABLE_FALLBACK_MODELS,
    "vision": SELECTABLE_VISION_MODELS,
    "vision_fallback": SELECTABLE_VISION_MODELS,
    "image_gen": SELECTABLE_IMAGE_GEN_MODELS,
    "image_generation": SELECTABLE_IMAGE_GEN_MODELS,
    "image_edit": SELECTABLE_IMAGE_EDIT_MODELS,
    "transcription": SELECTABLE_TRANSCRIPTION_MODELS,
}

_CAPABILITY_CHANNELS = frozenset({
    "vision",
    "vision_fallback",
    "image_gen",
    "image_generation",
    "image_edit",
    "transcription",
})

TELEGRAM_MODEL_CALLBACK_PREFIX = "ai_m_"
TELEGRAM_CALLBACK_DATA_LIMIT = 64

_TELEGRAM_MODEL_CALLBACK_CHANNEL_CODES = {
    "chat": "c",
    "fallback": "f",
    "vision": "v",
    "vision_fallback": "w",
    "image_gen": "g",
    "image_edit": "e",
    "transcription": "t",
}
_TELEGRAM_MODEL_CALLBACK_CHANNELS = {
    code: channel for channel, code in _TELEGRAM_MODEL_CALLBACK_CHANNEL_CODES.items()
}
_TELEGRAM_MODEL_CALLBACK_TOKEN_RE = re.compile(r"^ai_m_[a-z]_([0-9a-f]{32})$")


def _normalize_model_channel(channel: str | None) -> str:
    channel_name = (channel or "chat").strip().lower()
    if channel_name == "image_generation":
        channel_name = "image_gen"
    return channel_name


def _telegram_model_callback_digest(channel: str, provider: str, model: str) -> str:
    value = f"{channel}\0{provider}\0{model}".encode("utf-8")
    return sha256(value).hexdigest()[:32]


def build_telegram_model_callback_data(provider: str, channel: str, model: str) -> str:
    """Build a compact callback resolved against the current model catalog."""
    channel_name = _normalize_model_channel(channel)
    channel_code = _TELEGRAM_MODEL_CALLBACK_CHANNEL_CODES.get(channel_name)
    if channel_code is None:
        raise ModelUnavailableError(f"Канал '{channel}' не поддерживается для callback модели.")

    normalized_model = validate_model_selection(provider, model, channel=channel_name)
    canonical_provider = canonical_provider_name(provider)
    callback_data = (
        f"{TELEGRAM_MODEL_CALLBACK_PREFIX}{channel_code}_"
        f"{_telegram_model_callback_digest(channel_name, canonical_provider, normalized_model)}"
    )
    if len(callback_data.encode("utf-8")) > TELEGRAM_CALLBACK_DATA_LIMIT:
        raise ModelUnavailableError("Сформирован слишком длинный callback модели.")
    return callback_data


def resolve_telegram_model_callback(callback_data: str | None) -> tuple[str, str, str] | None:
    """Resolve a compact callback only if its tuple is active in the current catalog."""
    match = _TELEGRAM_MODEL_CALLBACK_TOKEN_RE.fullmatch(callback_data or "")
    if not match:
        return None

    channel_code = (callback_data or "").split("_")[2]
    channel = _TELEGRAM_MODEL_CALLBACK_CHANNELS.get(channel_code)
    if channel is None:
        return None
    digest = match.group(1)

    for provider in ALL_PROVIDERS:
        for model in get_selectable_models(provider, channel=channel):
            expected = _telegram_model_callback_digest(channel, provider, model)
            if compare_digest(digest, expected):
                return provider, channel, model
    return None


def ensure_model_available(provider: str | None, model: str | None, channel: str = "chat") -> None:
    """Validate a provider/model/channel tuple before persistence or HTTP."""
    p_name = canonical_provider_name(provider)
    channel_name = (channel or "chat").strip().lower()
    normalized = normalize_model_for_provider(p_name, model)

    if p_name not in (*ALL_PROVIDERS, *ALL_TRANSCRIPTION_PROVIDERS):
        raise ModelUnavailableError(
            f"Провайдер '{provider or 'AI'}' не поддерживается. "
            "Выберите провайдера из доступных настроек."
        )

    if channel_name not in _SELECTABLE_MODEL_CATALOGS:
        raise ModelUnavailableError(
            f"Канал '{channel_name}' не поддерживается для выбора модели."
        )

    if not normalized:
        raise ModelUnavailableError(
            f"Для провайдера '{p_name}' не задана модель для канала '{channel_name}'. "
            "Выберите актуальную модель в настройках."
        )

    if is_retired_model(normalized):
        raise ModelUnavailableError(
            f"Модель '{normalized}' ({p_name}) отключена провайдером. "
            f"Пожалуйста, выберите актуальную модель в настройках."
        )

    if normalized in APP_DISABLED_OR_MIGRATED_MODELS:
        raise ModelUnavailableError(
            f"Модель '{normalized}' ({p_name}) отключена в приложении. "
            "Пожалуйста, выберите актуальную модель в настройках."
        )

    selectable = get_selectable_models(p_name, channel=channel_name)
    if not selectable:
        raise ModelUnavailableError(
            f"Для провайдера '{p_name}' нет доступной модели для канала '{channel_name}'. "
            "Выберите провайдера с поддержкой этой возможности."
        )
    if normalized not in selectable:
        raise ModelUnavailableError(
            f"Модель '{normalized}' не поддерживается провайдером '{p_name}' "
            f"для канала '{channel_name}'. Выберите актуальную модель в настройках."
        )


# Models that omit sampling parameters (temperature, top_p, top_k)
CLAUDE_OMIT_SAMPLING_MODELS = frozenset({
    "claude-sonnet-5",
    "claude-opus-5",
})


def should_omit_claude_sampling(model: str | None) -> bool:
    """Return True if the Claude model does not accept non-default temperature/sampling parameters."""
    normalized = (model or "").strip()
    return normalized in CLAUDE_OMIT_SAMPLING_MODELS


def normalize_deepseek_model(model: str | None) -> str:
    """Normalize local legacy DeepSeek aliases to the default active model."""
    normalized = (model or "").strip()
    if not normalized or normalized in DEEPSEEK_LEGACY_MODELS:
        return DEEPSEEK_DEFAULT_MODEL
    return normalized


def normalize_model_for_provider(provider: str | None, model: str | None) -> str:
    """Normalize local compatibility aliases before catalog validation."""
    normalized = (model or "").strip()
    if canonical_provider_name(provider) == PROVIDER_DEEPSEEK:
        return normalize_deepseek_model(normalized)
    return normalized


def validate_model_selection(provider: str | None, model: str | None, channel: str = "chat") -> str:
    """Validate and return the canonical model for an admin/runtime selection."""
    normalized = normalize_model_for_provider(provider, model)
    ensure_model_available(provider, normalized, channel=channel)
    return normalized


def get_selectable_models(provider: str | None, channel: str = "chat") -> tuple[str, ...]:
    """Return the tuple of active selectable models for a provider and channel."""
    p_name = _canonical_provider_name(provider)
    channel_name = (channel or "chat").strip().lower()
    catalog = _SELECTABLE_MODEL_CATALOGS.get(channel_name, SELECTABLE_CHAT_MODELS)
    return catalog.get(p_name, ())


def get_default_model(provider: str | None, channel: str = "chat") -> str:
    """Return the active default model for a provider and channel."""
    p_name = _canonical_provider_name(provider)
    channel_name = (channel or "chat").strip().lower()
    if channel_name in _CAPABILITY_CHANNELS:
        selectable = get_selectable_models(p_name, channel=channel_name)
        if not selectable:
            raise ModelUnavailableError(
                f"Для провайдера '{p_name or provider or 'AI'}' нет доступной модели "
                f"для канала '{channel_name}'. Выберите провайдера с поддержкой этой возможности."
            )
        return selectable[0]
    return PROVIDER_DEFAULT_MODELS.get(p_name, "gemini-3.7-flash")


def get_chat_output_token_limit(provider: str | None, model: str | None) -> int:
    p_name = canonical_provider_name(provider)
    if p_name == PROVIDER_DEEPSEEK:
        return DEEPSEEK_CHAT_MAX_TOKENS
    if p_name == PROVIDER_OPENAI:
        return OPENAI_CHAT_MAX_TOKENS
    if p_name == PROVIDER_CLAUDE:
        return CLAUDE_CHAT_MAX_TOKENS
    if p_name == PROVIDER_GEMINI:
        return GEMINI_CHAT_MAX_TOKENS
    if p_name == PROVIDER_KIE:
        return KIE_CHAT_MAX_TOKENS
    if p_name == PROVIDER_OPENROUTER:
        spec = OPENROUTER_MODEL_SPECS.get((model or "").strip())
        return spec.output_limit if spec else max(spec.output_limit for spec in OPENROUTER_MODEL_SPECS.values())
    if p_name == PROVIDER_PERPLEXITY:
        return PERPLEXITY_MODE_OUTPUT_LIMITS.get((model or "").strip(), PERPLEXITY_CHAT_MAX_TOKENS)
    return 4096


def effective_chat_output_tokens(
    provider: str | None,
    model: str | None,
    configured_value: int | None,
) -> int:
    limit = get_chat_output_token_limit(provider, model)
    try:
        value = int(configured_value) if configured_value is not None else None
    except (TypeError, ValueError):
        value = None
    if value is None or value < 1 or value > limit:
        return limit
    return value


def validate_chat_output_tokens(
    provider: str | None,
    model: str | None,
    value: int | str | None,
) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Введите целое число токенов.") from exc
    if parsed < 1:
        raise ValueError("Минимальное значение — 1 токен.")
    limit = get_chat_output_token_limit(provider, model)
    if parsed > limit:
        raise ValueError(f"Для выбранной модели максимальное значение — {limit} токенов.")
    return parsed


# ==========================================
# 7. DEEPSEEK RESPONSE INSPECTION & DIAGNOSTICS
# ==========================================

@dataclass(frozen=True)
class DeepSeekDiagnostics:
    provider: str
    model: str
    platform: str
    finish_reason: str | None
    visible_content_present: bool
    visible_content_length: int
    reasoning_content_present: bool
    reasoning_content_length: int
    output_budget_exhausted: bool


def inspect_deepseek_response(
    response: Any,
    *,
    model: str,
    platform: str,
) -> tuple[str | None, DeepSeekDiagnostics]:
    """Safely inspect DeepSeek chat completion response and extract metadata.

    The raw reasoning string is only inspected transiently to derive presence
    and length. It is NEVER retained in diagnostics or returned to callers.
    """
    effective_model = str(model or DEEPSEEK_DEFAULT_MODEL)

    if response is None:
        return None, DeepSeekDiagnostics(
            provider="Deepseek",
            model=effective_model,
            platform=platform,
            finish_reason=None,
            visible_content_present=False,
            visible_content_length=0,
            reasoning_content_present=False,
            reasoning_content_length=0,
            output_budget_exhausted=False,
        )

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")

    if not choices:
        return None, DeepSeekDiagnostics(
            provider="Deepseek",
            model=effective_model,
            platform=platform,
            finish_reason=None,
            visible_content_present=False,
            visible_content_length=0,
            reasoning_content_present=False,
            reasoning_content_length=0,
            output_budget_exhausted=False,
        )

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason is None and isinstance(choice, dict):
        finish_reason = choice.get("finish_reason")
    finish_reason_str = str(finish_reason).strip() if finish_reason is not None else None

    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    content: Any = None
    reasoning_content: Any = None

    if message is not None:
        if isinstance(message, dict):
            content = message.get("content")
            reasoning_content = message.get("reasoning_content")
        else:
            content = getattr(message, "content", None)
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content is None:
                model_extra = getattr(message, "model_extra", None)
                if isinstance(model_extra, dict):
                    reasoning_content = model_extra.get("reasoning_content")
            if reasoning_content is None:
                pydantic_extra = getattr(message, "__pydantic_extra__", None)
                if isinstance(pydantic_extra, dict):
                    reasoning_content = pydantic_extra.get("reasoning_content")

    visible_str = content if isinstance(content, str) else ""
    visible_present = bool(visible_str and visible_str.strip())
    visible_len = len(visible_str)

    reasoning_str = reasoning_content if isinstance(reasoning_content, str) else ""
    reasoning_present = bool(reasoning_str and reasoning_str.strip())
    reasoning_len = len(reasoning_str)

    # Security invariant: reasoning string is never preserved
    del reasoning_content
    del reasoning_str

    output_exhausted = (finish_reason_str.lower() == "length") if finish_reason_str else False

    diagnostics = DeepSeekDiagnostics(
        provider="Deepseek",
        model=effective_model,
        platform=platform,
        finish_reason=finish_reason_str,
        visible_content_present=visible_present,
        visible_content_length=visible_len,
        reasoning_content_present=reasoning_present,
        reasoning_content_length=reasoning_len,
        output_budget_exhausted=output_exhausted,
    )

    clean_content = visible_str if visible_present else None
    return clean_content, diagnostics


# ==========================================
# 8. VISION TOKEN BUDGETS (SINGLE SOURCE OF TRUTH)
# ==========================================

KIE_VISION_INITIAL_MAX_TOKENS: int = 4096
DIRECT_VISION_MAX_TOKENS: int = 16384


def get_provider_vision_max_tokens(provider: str | None, model: str | None = None) -> int:
    """Return max token budget for vision requests by provider."""
    p_name = canonical_provider_name(provider)
    if p_name == PROVIDER_KIE:
        return KIE_VISION_INITIAL_MAX_TOKENS
    if p_name in (PROVIDER_OPENAI, PROVIDER_CLAUDE, PROVIDER_GEMINI):
        return DIRECT_VISION_MAX_TOKENS
    if p_name == PROVIDER_OPENROUTER:
        spec = OPENROUTER_MODEL_SPECS.get((model or "").strip())
        if not spec or not spec.vision:
            raise ValueError(f"Модель OpenRouter '{model}' не поддерживает vision")
        return spec.output_limit
    raise ValueError(f"Unknown or unsupported vision provider: {provider}")
