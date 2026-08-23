import html
import logging
import re
import traceback
from typing import Any, Sequence

from aiogram import Bot

from database import get_all_admin_ids


# Patterns for scrubbing secrets from error strings, URLs, and headers
_SECRET_PATTERNS = [
    (re.compile(r"([?&]key=)[^&\s'\"]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(sk-[A-Za-z0-9_\-]{8,})", re.IGNORECASE), r"sk-[REDACTED]"),
    (re.compile(r"(AIza[0-9A-Za-z-_]{20,})", re.IGNORECASE), r"AIza[REDACTED]"),
    (re.compile(r"(api[_-]?key[\"'\s:=]+)[A-Za-z0-9_\-]{8,}", re.IGNORECASE), r"\1[REDACTED]"),
]


def sanitize_secret_values(text: str) -> str:
    """Scrub known secret tokens, query params, and API keys from diagnostic strings."""
    if not text:
        return ""
    result = str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _user_ref(user_id: int | None, username: str | None = None, full_name: str | None = None) -> str:
    if user_id is None:
        return "неизвестно"
    link = f"<a href='tg://user?id={user_id}'>перейти в профиль</a>"
    if username:
        return f"@{html.escape(username)} ({link})"
    name = html.escape(full_name) if full_name else str(user_id)
    return f"{name} ({link})"


def _shorten(value: str, limit: int = 1400) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]} ... [truncated]"


def exception_summary(exception: Exception) -> str:
    message = str(exception).strip()
    return message or type(exception).__name__


def classify_ai_error(exception: Exception | None, provider: str | None = None) -> tuple[str, str]:
    """
    Classify low-level provider exceptions into standardized error codes
    and concise, human-readable Russian descriptions.
    """
    if exception is None:
        return "unknown", "Неизвестная ошибка"

    exc_type = type(exception).__name__
    text = str(exception).lower()
    full_repr = f"{exc_type}: {text}"

    if "отключена провайдером" in text or "modelunavailableerror" in exc_type.lower():
        return "retired_model_unsupported", "Модель отключена провайдером и не поддерживается"

    if any(m in text for m in ("не указан api ключ", "api ключ openai не установлен", "api ключ не настроен")):
        return "missing_config", "API-ключ или конфигурация провайдера не настроены"

    if any(m in text for m in ("invalid api key", "invalid_api_key", "authenticationerror", "unauthorized", "status code 401")):
        return "auth_invalid_key", "Неверный или неактивный API-ключ (401 Unauthorized)"

    if any(m in text for m in ("user location is not supported", "location not supported", "geoblock", "geo-block", "country, region, or territory", "status code 403")):
        if "location" in text or "country" in text or "region" in text or "territory" in text:
            return "geo_blocked", "Блокировка доступа по региону (Geo-Block / 403 Forbidden)"
        return "auth_forbidden", "Доступ запрещен провайдером (403 Forbidden)"

    if any(m in text for m in ("insufficientbalanceerror", "insufficient credits", "credit balance", "billing", "quota", "purchase credits", "code\": 402", "status code 402")):
        return "insufficient_balance", "Недостаточно средств, кредитов или квоты на балансе (402)"

    if any(m in text for m in ("ratelimiterror", "rate limit", "too many requests", "status code 429", "code\": 429")):
        return "rate_limited", "Превышен лимит запросов к API (429 Rate Limit)"

    if any(m in text for m in ("timeouterror", "timeout", "timed out", "apitimeouterror")):
        return "timeout", "Превышено время ожидания ответа от API (Timeout)"

    if any(m in text for m in ("500", "502", "503", "504", "internalservererror", "service unavailable", "overloaded", "server error")):
        return "provider_5xx", "Внутренняя ошибка или перегрузка сервиса провайдера (5xx)"

    if any(m in text for m in ("connecterror", "connection error", "readerror", "network", "api connection", "remotedisconnected")):
        return "network_error", "Ошибка сетевого соединения с API"

    if any(m in text for m in ("пустой ответ", "empty", "no response data")):
        return "empty_response", "Провайдер вернул пустой ответ"

    clean_summary = exception_summary(exception)
    return "general_error", clean_summary


async def notify_admins_about_error(
    bot: Bot,
    *,
    title: str,
    user_id: int | None = None,
    username: str | None = None,
    full_name: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    stage: str | None = None,
    details: str | None = None,
    extra: dict[str, Any] | None = None,
    provider_attempts: Sequence[dict[str, str | None]] | None = None,
    exception: Exception | None = None,
    include_traceback: bool = True,
    logger: logging.Logger | None = None,
    level: int = logging.ERROR,
) -> None:
    log = logger or logging.getLogger(__name__)
    if exception is not None and include_traceback:
        raw_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    else:
        raw_trace = ""
    trace = sanitize_secret_values(raw_trace)
    
    effective_details = details.strip() if details and details.strip() else None
    if not effective_details and exception is not None and not provider_attempts:
        effective_details = exception_summary(exception)
    if effective_details:
        effective_details = sanitize_secret_values(effective_details)

    log_lines = [title]
    if provider:
        log_lines.append(f"provider={provider}")
    if model:
        log_lines.append(f"model={model}")
    if stage:
        log_lines.append(f"stage={stage}")
    if effective_details:
        log_lines.append(f"details={effective_details}")
    if provider_attempts:
        attempts_log = "; ".join(
            f"{attempt.get('provider') or 'unknown'}={sanitize_secret_values(attempt.get('error') or 'unknown error')}"
            for attempt in provider_attempts
        )
        log_lines.append(f"attempts={attempts_log}")
    if extra:
        log_lines.append(f"extra={extra}")

    if exception is not None:
        log.log(level, " | ".join(log_lines), exc_info=exception)
    else:
        log.log(level, " | ".join(log_lines))

    message_lines = [f"⚠️ <b>{html.escape(title)}</b>"]
    if user_id is not None:
        message_lines.append(f"Пользователь: {_user_ref(user_id, username, full_name)}")
    if provider:
        message_lines.append(f"Провайдер: <code>{html.escape(provider)}</code>")
    if model:
        message_lines.append(f"Модель: <code>{html.escape(model)}</code>")
    if stage:
        message_lines.append(f"Этап: <code>{html.escape(stage)}</code>")
    if effective_details:
        message_lines.append(f"Ошибка: <code>{html.escape(_shorten(effective_details, 1800))}</code>")
    if provider_attempts:
        message_lines.append("Попытки провайдеров:")
        for index, attempt in enumerate(provider_attempts, start=1):
            attempt_provider = html.escape(attempt.get("provider") or "неизвестно")
            attempt_model = attempt.get("model")
            model_suffix = f" (<code>{html.escape(attempt_model)}</code>)" if attempt_model else ""
            raw_err = attempt.get("error") or "неизвестная ошибка"
            attempt_error = html.escape(_shorten(sanitize_secret_values(raw_err), 900))
            message_lines.append(
                f"{index}. <b>{attempt_provider}</b>{model_suffix}\n"
                f"Ошибка: <code>{attempt_error}</code>"
            )
    if extra:
        safe_extra = _shorten(sanitize_secret_values("\n".join(f"{k}={v}" for k, v in extra.items())), 1200)
        message_lines.append(f"Контекст: <code>{html.escape(safe_extra)}</code>")
    if trace and trace.strip() and trace.strip() != "NoneType: None":
        message_lines.append(f"Traceback: <code>{html.escape(_shorten(trace, 1800))}</code>")

    admin_text = "\n".join(message_lines)
    admin_ids = await get_all_admin_ids()
    for admin_id in admin_ids:
        try:
            await bot.send_message(admin_id, admin_text, parse_mode="HTML")
        except Exception as send_exc:
            log.error("Failed to deliver admin error notification admin_id=%s error=%s", admin_id, send_exc)
