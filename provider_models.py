from dataclasses import dataclass


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
        "gemini-3-flash-preview",
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


class ModelUnavailableError(Exception):
    """Raised when an attempt is made to execute a retired model."""
    pass


def ensure_model_available(provider: str | None, model: str | None, channel: str = "chat") -> None:
    """Validate that the model is not retired before initiating network requests."""
    normalized = (model or "").strip()
    if not normalized:
        return
    if is_retired_model(normalized):
        p_name = _canonical_provider_name(provider) or provider or "AI"
        raise ModelUnavailableError(
            f"Модель '{normalized}' ({p_name}) отключена провайдером. "
            f"Пожалуйста, выберите актуальную модель в настройках."
        )


def normalize_deepseek_model(model: str | None) -> str:
    """Normalize local legacy DeepSeek aliases to the default active model."""
    normalized = (model or "").strip()
    if not normalized or normalized in DEEPSEEK_LEGACY_MODELS:
        return DEEPSEEK_DEFAULT_MODEL
    return normalized


def get_selectable_models(provider: str | None, channel: str = "chat") -> tuple[str, ...]:
    """Return the tuple of active selectable models for a provider and channel."""
    p_name = _canonical_provider_name(provider)
    catalogs = {
        "chat": SELECTABLE_CHAT_MODELS,
        "fallback": SELECTABLE_FALLBACK_MODELS,
        "vision": SELECTABLE_VISION_MODELS,
        "image_gen": SELECTABLE_IMAGE_GEN_MODELS,
        "image_generation": SELECTABLE_IMAGE_GEN_MODELS,
        "image_edit": SELECTABLE_IMAGE_EDIT_MODELS,
        "transcription": SELECTABLE_TRANSCRIPTION_MODELS,
    }
    catalog = catalogs.get(channel, SELECTABLE_CHAT_MODELS)
    return catalog.get(p_name, ())


def get_default_model(provider: str | None, channel: str = "chat") -> str:
    """Return the active default model for a provider and channel."""
    p_name = _canonical_provider_name(provider)
    if channel == "vision":
        vision_models = SELECTABLE_VISION_MODELS.get(p_name)
        if vision_models:
            return vision_models[0]
        return DEFAULT_VISION_MODEL
    if channel in ("image_gen", "image_generation"):
        if p_name == PROVIDER_OPENAI:
            return DEFAULT_OPENAI_IMAGE_MODEL
        if p_name == PROVIDER_KIE:
            return "seedream/4.5-text-to-image"
    if channel == "image_edit":
        if p_name == PROVIDER_KIE:
            return "seedream/4.5-edit"
    if channel == "transcription":
        if p_name == PROVIDER_OPENAI:
            return DEFAULT_OPENAI_TRANSCRIPTION_MODEL
        if p_name == PROVIDER_GEMINI:
            return "gemini-3.7-flash"
        if p_name == PROVIDER_KIE:
            return DEFAULT_KIE_TRANSCRIPTION_MODEL
    return PROVIDER_DEFAULT_MODELS.get(p_name, "gemini-3.7-flash")
