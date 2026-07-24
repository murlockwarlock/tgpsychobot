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


def normalize_deepseek_model(model: str | None) -> str:
    normalized = (model or "").strip()
    if not normalized or normalized in DEEPSEEK_LEGACY_MODELS:
        return DEEPSEEK_DEFAULT_MODEL
    return normalized
