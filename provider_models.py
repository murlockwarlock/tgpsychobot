from dataclasses import dataclass


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


KIE_CHAT_MODELS = (
    KIE_DEFAULT_CHAT_MODEL,
    "gemini-2.5-flash",
    "claude-haiku-4-5",
    "grok-4-3",
    "gemini-3-7-flash",
    "gpt-5-6-luna",
)


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


def normalize_deepseek_model(model: str | None) -> str:
    normalized = (model or "").strip()
    if not normalized or normalized in DEEPSEEK_LEGACY_MODELS:
        return DEEPSEEK_DEFAULT_MODEL
    return normalized
