from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from database import AIConfig, AIModelSettings
from provider_models import (
    PROVIDER_CLAUDE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    PROVIDER_PERPLEXITY,
    canonical_provider_name,
    get_chat_output_token_limit,
    get_default_model,
    normalize_deepseek_model,
    should_omit_claude_sampling,
)


REASONING_AUTO = "auto"
REASONING_NONE = "none"
REASONING_LOW = "low"
REASONING_HIGH = "high"
REASONING_MAX = "max"
REASONING_VALUES = (REASONING_AUTO, REASONING_NONE, REASONING_LOW, REASONING_HIGH, REASONING_MAX)


@dataclass(frozen=True)
class GenerationCapabilities:
    max_output_tokens: bool
    max_output_tokens_required: bool
    output_limit: int | None
    temperature: bool
    reasoning_effort: tuple[str, ...]
    temperature_ignored_for_reasoning: bool = False


@dataclass(frozen=True)
class ResolvedModelSettings:
    provider: str
    model: str
    channel: str
    max_output_tokens: int | None
    temperature: float | None
    reasoning_effort: str
    source: str


def normalize_model_scope(provider: str | None, model: str | None, channel: str = "chat") -> tuple[str, str, str]:
    normalized_provider = canonical_provider_name(provider) or str(provider or "").strip()
    normalized_model = str(model or "").strip()
    if normalized_provider == PROVIDER_DEEPSEEK:
        normalized_model = normalize_deepseek_model(normalized_model)
    if not normalized_model:
        try:
            normalized_model = get_default_model(normalized_provider, channel=channel)
        except Exception:
            normalized_model = ""
    return normalized_provider, normalized_model, (channel or "chat").strip().lower()


def get_generation_capabilities(
    provider: str | None,
    model: str | None,
    channel: str = "chat",
) -> GenerationCapabilities:
    normalized_provider, normalized_model, normalized_channel = normalize_model_scope(provider, model, channel)
    if normalized_channel != "chat":
        return GenerationCapabilities(False, False, None, False, ())
    if normalized_provider == PROVIDER_DEEPSEEK:
        return GenerationCapabilities(
            max_output_tokens=True,
            max_output_tokens_required=False,
            output_limit=get_chat_output_token_limit(normalized_provider, normalized_model, REASONING_MAX),
            temperature=True,
            reasoning_effort=REASONING_VALUES,
            temperature_ignored_for_reasoning=True,
        )
    if normalized_provider == PROVIDER_OPENAI:
        return GenerationCapabilities(
            max_output_tokens=True,
            max_output_tokens_required=False,
            output_limit=get_chat_output_token_limit(normalized_provider, normalized_model),
            temperature=not normalized_model.startswith("gpt-5.6"),
            reasoning_effort=(),
        )
    if normalized_provider == PROVIDER_CLAUDE:
        return GenerationCapabilities(
            max_output_tokens=True,
            max_output_tokens_required=True,
            output_limit=get_chat_output_token_limit(normalized_provider, normalized_model),
            temperature=not should_omit_claude_sampling(normalized_model),
            reasoning_effort=(),
        )
    if normalized_provider == PROVIDER_GEMINI:
        return GenerationCapabilities(
            max_output_tokens=True,
            max_output_tokens_required=False,
            output_limit=get_chat_output_token_limit(normalized_provider, normalized_model),
            temperature=not normalized_model.startswith(("gemini-3.7", "gemini-3.6")),
            reasoning_effort=(),
        )
    if normalized_provider == PROVIDER_KIE:
        return GenerationCapabilities(
            max_output_tokens=True,
            max_output_tokens_required=True,
            output_limit=get_chat_output_token_limit(normalized_provider, normalized_model),
            temperature=True,
            reasoning_effort=(),
        )
    if normalized_provider == PROVIDER_OPENROUTER:
        return GenerationCapabilities(
            max_output_tokens=True,
            max_output_tokens_required=False,
            output_limit=get_chat_output_token_limit(normalized_provider, normalized_model),
            temperature=not normalized_model.startswith(("openai/gpt-5.6", "google/gemini-3.7", "google/gemini-3.8")),
            reasoning_effort=(),
        )
    if normalized_provider == PROVIDER_PERPLEXITY:
        return GenerationCapabilities(
            max_output_tokens=True,
            max_output_tokens_required=False,
            output_limit=get_chat_output_token_limit(normalized_provider, normalized_model),
            temperature=False,
            reasoning_effort=(),
        )
    return GenerationCapabilities(True, False, get_chat_output_token_limit(normalized_provider, normalized_model), True, ())


def normalize_reasoning_effort(provider: str | None, value: Any) -> str:
    if canonical_provider_name(provider) != PROVIDER_DEEPSEEK:
        return REASONING_AUTO
    normalized = str(value or REASONING_AUTO).strip().lower()
    aliases = {"default": REASONING_AUTO, "off": REASONING_NONE, "disabled": REASONING_NONE}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in REASONING_VALUES else REASONING_AUTO


def reasoning_from_legacy(value: bool | None) -> str:
    if value is True:
        return REASONING_HIGH
    if value is False:
        return REASONING_NONE
    return REASONING_AUTO


def validate_model_setting(
    provider: str | None,
    model: str | None,
    name: str,
    value: Any,
    *,
    reasoning_effort: str | None = None,
) -> Any:
    normalized_provider, normalized_model, _ = normalize_model_scope(provider, model)
    capabilities = get_generation_capabilities(normalized_provider, normalized_model)
    if name == "max_output_tokens":
        if value is None or str(value).strip().casefold() in {"auto", "авто", "default", "по умолчанию", "по умолчанию."}:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Введите целое число токенов или «Авто».") from exc
        if parsed < 1:
            raise ValueError("Минимальное значение — 1 токен.")
        limit = capabilities.output_limit
        if normalized_provider == PROVIDER_DEEPSEEK:
            limit = get_chat_output_token_limit(
                normalized_provider,
                normalized_model,
                normalize_reasoning_effort(normalized_provider, reasoning_effort),
            )
        if limit is not None and parsed > limit:
            raise ValueError(f"Для выбранной модели максимальное значение — {limit} токенов.")
        return parsed
    if name == "temperature":
        if not capabilities.temperature:
            raise ValueError("Температура для этой модели не поддерживается.")
        if value is None or str(value).strip().casefold() in {"auto", "авто", "default", "по умолчанию", "по умолчанию."}:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Введите число от 0.0 до 2.0.") from exc
        if not 0.0 <= parsed <= 2.0:
            raise ValueError("Введите число от 0.0 до 2.0.")
        return parsed
    if name == "reasoning_effort":
        parsed = normalize_reasoning_effort(normalized_provider, value)
        if parsed not in capabilities.reasoning_effort:
            raise ValueError("Этот режим рассуждений для модели не поддерживается.")
        return parsed
    raise ValueError("Неизвестный параметр модели.")


def _legacy_scope(config: AIConfig | None, provider: str, model: str) -> bool:
    if config is None:
        return False
    current_provider, current_model, _ = normalize_model_scope(config.provider, None)
    if current_provider == PROVIDER_KIE:
        current_model = getattr(config, "kie_model", None) or current_model
    else:
        current_model = getattr(config, f"{current_provider.lower()}_model", None) or current_model
    current_provider, current_model, _ = normalize_model_scope(current_provider, current_model)
    return current_provider == provider and current_model == model


async def _session_scalar(session, statement):
    scalar = getattr(session, "scalar", None)
    if scalar is not None:
        return await scalar(statement)
    execute = getattr(session, "execute", None)
    if execute is None:
        return None
    result = await execute(statement)
    if hasattr(result, "scalar_one_or_none"):
        return result.scalar_one_or_none()
    if hasattr(result, "scalar"):
        return result.scalar()
    return None


async def resolve_model_settings(
    session,
    provider: str | None,
    model: str | None,
    channel: str = "chat",
    *,
    config: AIConfig | None = None,
) -> ResolvedModelSettings:
    normalized_provider, normalized_model, normalized_channel = normalize_model_scope(provider, model, channel)
    row = None
    scoped_rows_exist = False
    scoped_table_available = True
    if normalized_model:
        statement = select(AIModelSettings).where(
            AIModelSettings.provider == normalized_provider,
            AIModelSettings.model == normalized_model,
            AIModelSettings.channel == normalized_channel,
        )
        try:
            row = await _session_scalar(session, statement)
        except OperationalError as exc:
            if "no such table" not in str(exc).lower() and "does not exist" not in str(exc).lower():
                raise
            scoped_table_available = False
            row = None
    if row is None and scoped_table_available:
        try:
            scoped_rows_exist = bool(await _session_scalar(session, select(AIModelSettings.id).limit(1)))
        except OperationalError as exc:
            if "no such table" not in str(exc).lower() and "does not exist" not in str(exc).lower():
                raise
            scoped_table_available = False
    if row is not None and all(
        hasattr(row, field)
        for field in ("max_output_tokens", "temperature", "reasoning_effort")
    ):
        return ResolvedModelSettings(
            normalized_provider,
            normalized_model,
            normalized_channel,
            row.max_output_tokens,
            row.temperature,
            normalize_reasoning_effort(normalized_provider, row.reasoning_effort),
            "model",
        )
    if config is None:
        config = await session.get(AIConfig, 1)
    if normalized_channel == "chat" and not scoped_rows_exist and _legacy_scope(config, normalized_provider, normalized_model):
        return ResolvedModelSettings(
            normalized_provider,
            normalized_model,
            normalized_channel,
            getattr(config, "max_output_tokens", None),
            getattr(config, "temperature", None),
            reasoning_from_legacy(getattr(config, "deepseek_thinking_enabled", None)) if normalized_provider == PROVIDER_DEEPSEEK else REASONING_AUTO,
            "legacy",
        )
    return ResolvedModelSettings(normalized_provider, normalized_model, normalized_channel, None, None, REASONING_AUTO, "default")


async def get_or_create_model_settings(session, provider: str | None, model: str | None, channel: str = "chat", *, config: AIConfig | None = None) -> AIModelSettings:
    normalized_provider, normalized_model, normalized_channel = normalize_model_scope(provider, model, channel)
    row = await session.scalar(
        select(AIModelSettings).where(
            AIModelSettings.provider == normalized_provider,
            AIModelSettings.model == normalized_model,
            AIModelSettings.channel == normalized_channel,
        )
    )
    if row is not None:
        return row
    resolved = await resolve_model_settings(session, normalized_provider, normalized_model, normalized_channel, config=config)
    row = AIModelSettings(
        provider=normalized_provider,
        model=normalized_model,
        channel=normalized_channel,
        max_output_tokens=resolved.max_output_tokens,
        temperature=resolved.temperature,
        reasoning_effort=resolved.reasoning_effort if normalized_provider == PROVIDER_DEEPSEEK else None,
    )
    session.add(row)
    await session.flush()
    return row


def wire_max_output_tokens(settings: ResolvedModelSettings) -> int | None:
    capabilities = get_generation_capabilities(settings.provider, settings.model, settings.channel)
    if settings.max_output_tokens is not None:
        return validate_model_setting(
            settings.provider,
            settings.model,
            "max_output_tokens",
            settings.max_output_tokens,
            reasoning_effort=settings.reasoning_effort,
        )
    if capabilities.max_output_tokens_required:
        return capabilities.output_limit
    return None


def wire_temperature(settings: ResolvedModelSettings) -> float | None:
    capabilities = get_generation_capabilities(settings.provider, settings.model, settings.channel)
    if not capabilities.temperature or (
        capabilities.temperature_ignored_for_reasoning
        and settings.reasoning_effort != REASONING_NONE
    ):
        return None
    if settings.temperature is None:
        return None
    return validate_model_setting(settings.provider, settings.model, "temperature", settings.temperature)


def wire_reasoning_effort(settings: ResolvedModelSettings) -> str | None:
    if settings.provider != PROVIDER_DEEPSEEK:
        return None
    effort = normalize_reasoning_effort(settings.provider, settings.reasoning_effort)
    return None if effort == REASONING_AUTO else effort
