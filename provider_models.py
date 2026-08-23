from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import re


# ==========================================
# 1. PROVIDER CANONICAL IDENTIFIERS & DEFAULTS
# ==========================================

PROVIDER_GEMINI = "Gemini"
PROVIDER_CLAUDE = "Claude"
PROVIDER_OPENAI = "OpenAI"
PROVIDER_DEEPSEEK = "Deepseek"
PROVIDER_KIE = "KIE"

ALL_PROVIDERS = (
    PROVIDER_GEMINI,
    PROVIDER_CLAUDE,
    PROVIDER_OPENAI,
    PROVIDER_DEEPSEEK,
    PROVIDER_KIE,
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

KIE_DEFAULT_CHAT_MODEL = "gemini-3-flash"

PROVIDER_DEFAULT_MODELS = {
    PROVIDER_GEMINI: "gemini-3.7-flash",
    PROVIDER_CLAUDE: "claude-sonnet-5",
    PROVIDER_OPENAI: "gpt-5.6-terra",
    PROVIDER_DEEPSEEK: DEEPSEEK_DEFAULT_MODEL,
    PROVIDER_KIE: KIE_DEFAULT_CHAT_MODEL,
}

DEFAULT_VISION_MODEL = "gemini-3.7-flash"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2"
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
}

# Active text-to-image models (Google Imagen retired, excluded until protocol migration)
SELECTABLE_IMAGE_GEN_MODELS: dict[str, tuple[str, ...]] = {
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

# Active image edit models (Google Image Edit retired, excluded until protocol migration)
SELECTABLE_IMAGE_EDIT_MODELS: dict[str, tuple[str, ...]] = {
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
    for canonical in ALL_PROVIDERS:
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
    "image_gen": SELECTABLE_IMAGE_GEN_MODELS,
    "image_generation": SELECTABLE_IMAGE_GEN_MODELS,
    "image_edit": SELECTABLE_IMAGE_EDIT_MODELS,
    "transcription": SELECTABLE_TRANSCRIPTION_MODELS,
}

_CAPABILITY_CHANNELS = frozenset({
    "vision",
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

    if p_name not in ALL_PROVIDERS:
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
