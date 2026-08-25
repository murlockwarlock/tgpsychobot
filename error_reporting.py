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
    (
        re.compile(
            r"((?:password|passwd|secret|signature(?:value)?|token|access_token|client_secret|"
            r"merchant_password|merchant_pass[12]|pass[12]|authorization|shop_id)\s*[:=]\s*[\"']?)[^\"'\s,}&]+",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]",
    ),
]

_SENSITIVE_EXTRA_KEY = re.compile(
    r"(?:password|passwd|secret|signature|token|api[_-]?key|authorization|credential)",
    re.IGNORECASE,
)


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


def exception_chain(exception: Exception | None, *, include_context: bool = True) -> list[Exception]:
    """Return a finite outer-to-inner exception chain."""
    chain: list[Exception] = []
    seen: set[int] = set()
    current = exception
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
        if current is None and include_context:
            current = chain[-1].__context__
    return chain


def root_cause_exception(
    exception: Exception | None,
    *,
    include_context: bool = True,
) -> Exception | None:
    """Choose the deepest meaningful exception without following cycles."""
    chain = exception_chain(exception, include_context=include_context)
    if not chain:
        return None
    meaningful = [item for item in chain if str(item).strip()]
    return meaningful[-1] if meaningful else chain[-1]


def exception_summary(exception: Exception, *, include_context: bool = True) -> str:
    root = root_cause_exception(exception, include_context=include_context) or exception
    message = str(root).strip()
    return message or type(root).__name__


_ERROR_CLASS_DESCRIPTIONS = {
    "network_ssl": "Ошибка SSL/TLS при обращении к провайдеру",
    "network_connection": "Ошибка сетевого соединения с провайдером",
    "timeout": "Провайдер не ответил вовремя",
    "auth": "Провайдер отклонил учетные данные или авторизацию",
    "forbidden_geo": "Провайдер запретил доступ или ограничил его по региону",
    "rate_limit": "Провайдер ограничил частоту запросов",
    "insufficient_balance_quota": "Недостаточно баланса, кредитов или квоты провайдера",
    "provider_rejection": "Провайдер отклонил запрос или платеж",
    "provider_5xx": "Внутренняя ошибка или перегрузка сервиса провайдера",
    "empty_response": "Провайдер вернул пустой ответ",
    "invalid_response": "Провайдер вернул пустой или некорректный ответ",
    "configuration": "Ошибка конфигурации приложения или провайдера",
    "application_internal": "Внутренняя ошибка приложения",
    "unknown": "Неизвестная ошибка",
}


def _exception_status_codes(exception: Exception, *, include_context: bool = True) -> set[int]:
    codes: set[int] = set()
    for item in exception_chain(exception, include_context=include_context):
        candidates = [getattr(item, "status_code", None), getattr(item, "status", None)]
        response = getattr(item, "response", None)
        if response is not None:
            candidates.extend([getattr(response, "status_code", None), getattr(response, "status", None)])
        content = getattr(item, "content", None)
        if isinstance(content, dict):
            candidates.extend([content.get("status_code"), content.get("status"), content.get("code")])
        for candidate in candidates:
            try:
                if isinstance(candidate, int):
                    codes.add(candidate)
                elif isinstance(candidate, str) and candidate.isdigit():
                    codes.add(int(candidate))
            except (TypeError, ValueError):
                continue
    return codes


def classify_external_error(
    exception: Exception | None,
    provider: str | None = None,
    *,
    include_context: bool = True,
) -> tuple[str, str]:
    """Classify an external-call failure using the complete exception chain."""
    if exception is None:
        return "unknown", _ERROR_CLASS_DESCRIPTIONS["unknown"]

    chain = exception_chain(exception, include_context=include_context)
    type_text = " ".join(type(item).__name__ for item in chain).lower()
    error_text = " ".join(str(item) for item in chain).lower()
    combined = f"{type_text} {error_text}"
    status_codes = _exception_status_codes(exception, include_context=include_context)

    if any(marker in combined for marker in (
        "sslerror",
        "ssl:",
        "tls",
        "certificate verify failed",
        "unexpected_eof",
        "wrong version number",
    )):
        code = "network_ssl"
    elif (
        any(marker in combined for marker in ("timeouterror", "readtimeout", "connecttimeout", "apitimeouterror"))
        or "timeout" in combined
        or "timed out" in combined
    ):
        code = "timeout"
    elif any(marker in combined for marker in (
        "connecterror",
        "connectionerror",
        "connection refused",
        "connection reset",
        "readerror",
        "remoteprotocolerror",
        "remotedisconnected",
        "network error",
        "networkerror",
    )):
        code = "network_connection"
    elif (
        status_codes & {401}
        or any(marker in combined for marker in (
            "unauthorized",
            "authenticationerror",
            "invalid api key",
            "invalid_api_key",
            "invalid credentials",
            "authentication failed",
        ))
    ):
        code = "auth"
    elif (
        status_codes & {403}
        or any(marker in combined for marker in (
            "forbidden",
            "user location is not supported",
            "location not supported",
            "geo-block",
            "geoblock",
            "country, region, or territory",
        ))
    ):
        code = "forbidden_geo"
    elif (
        status_codes & {429}
        or any(marker in combined for marker in ("ratelimiterror", "rate limit", "too many requests"))
    ):
        code = "rate_limit"
    elif (
        status_codes & {402}
        or any(marker in combined for marker in (
            "insufficientbalance",
            "insufficient balance",
            "insufficient credits",
            "credit balance",
            "quota",
            "purchase credits",
            "billing",
            "balance exhausted",
        ))
    ):
        code = "insufficient_balance_quota"
    elif (
        status_codes & {400, 409, 422}
        or "badrequesterror" in type_text
        or any(marker in combined for marker in (
            "payment_method_not_found",
            "provider rejected",
            "rejected by provider",
            "declined",
            "decline",
            "invalid parameter",
        ))
    ):
        code = "provider_rejection"
    elif (
        any(500 <= status < 600 for status in status_codes)
        or any(marker in combined for marker in (
            "internalservererror",
            "service unavailable",
            "server error",
            "server exception",
            "overloaded",
        ))
    ):
        code = "provider_5xx"
    elif any(marker in combined for marker in (
        "empty response",
        "empty content",
        "empty data",
        "no response data",
        "returned no ",
    )):
        code = "empty_response"
    elif (
        any(marker in type_text for marker in ("airesponseerror", "jsondecodeerror", "invalidresponse"))
        or any(marker in combined for marker in (
            "invalid response",
            "no file url",
            "no taskid",
            "cannot decode",
        ))
    ):
        code = "invalid_response"
    elif any(marker in combined for marker in (
        "api key",
        "api ключ",
        "not configured",
        "не настро",
        "configuration",
        "конфигурац",
        "unsupported provider",
        "unsupported model",
        "unknown provider",
        "неизвестный провайдер",
        "неподдерживаемый провайдер",
        "model unavailable",
        "modelunavailableerror",
        "модель недоступна",
        "missing config",
    )):
        code = "configuration"
    elif any(type(item).__name__ in {"AttributeError", "KeyError", "TypeError", "AssertionError"} for item in chain):
        code = "application_internal"
    else:
        code = "unknown"

    return code, _ERROR_CLASS_DESCRIPTIONS[code]


def classify_ai_error(exception: Exception | None, provider: str | None = None) -> tuple[str, str]:
    """
    Classify low-level provider exceptions into standardized error codes
    and concise, human-readable Russian descriptions.
    """
    if exception is None:
        return "unknown", "Неизвестная ошибка"

    exc_type = type(exception).__name__
    text = " ".join(str(item) for item in exception_chain(exception)).lower()

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

    if any(m in text for m in ("sslerror", "unexpected_eof", "tls", "certificate verify failed")):
        return "network_ssl", _ERROR_CLASS_DESCRIPTIONS["network_ssl"]

    if any(m in text for m in ("connecterror", "connection error", "readerror", "network", "api connection", "remotedisconnected")):
        return "network_error", "Ошибка сетевого соединения с API"

    if any(m in text for m in ("пустой ответ", "empty", "no response data")):
        return "empty_response", "Провайдер вернул пустой ответ"

    generic_code, generic_description = classify_external_error(exception, provider=provider)
    legacy_code = {
        "auth": "auth_invalid_key",
        "forbidden_geo": "geo_blocked" if any(marker in text for marker in ("location", "country", "region", "territory")) else "auth_forbidden",
        "rate_limit": "rate_limited",
        "insufficient_balance_quota": "insufficient_balance",
        "timeout": "timeout",
        "provider_5xx": "provider_5xx",
        "network_connection": "network_error",
        "empty_response": "empty_response",
        "invalid_response": "invalid_response",
        "configuration": "missing_config",
    }.get(generic_code, "general_error")
    if legacy_code != "general_error":
        return legacy_code, generic_description
    clean_summary = exception_summary(exception)
    return "general_error", clean_summary


def _safe_extra_text(extra: dict[str, Any]) -> str:
    safe_items = []
    for key, value in extra.items():
        key_text = str(key)
        safe_value = "[REDACTED]" if _SENSITIVE_EXTRA_KEY.search(key_text) else sanitize_secret_values(str(value))
        safe_items.append(f"{key_text}={safe_value}")
    return "\n".join(safe_items)


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
    classification_override: str | None = None,
    include_traceback: bool = True,
    logger: logging.Logger | None = None,
    level: int = logging.ERROR,
) -> None:
    log = logger or logging.getLogger(__name__)
    root_exception = root_cause_exception(exception)
    classification = classification_override
    classification_description = None
    root_summary = None
    outcome = getattr(exception, "ai_outcome", None) if exception is not None else None
    if exception is not None:
        detected_classification, detected_description = classify_external_error(exception, provider=provider)
        if classification is None:
            classification = detected_classification
            classification_description = detected_description
        else:
            classification_description = _ERROR_CLASS_DESCRIPTIONS.get(
                classification,
                detected_description,
            )
        root_summary = sanitize_secret_values(exception_summary(root_exception or exception))

    if exception is not None and include_traceback:
        raw_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    else:
        raw_trace = ""
    trace = sanitize_secret_values(raw_trace)

    effective_details = details.strip() if details and details.strip() else None
    if exception is not None and effective_details:
        outer_summary = str(exception).strip()
        if effective_details in {outer_summary, root_summary}:
            effective_details = None
    if effective_details:
        effective_details = sanitize_secret_values(effective_details)

    log_lines = [title]
    if provider:
        log_lines.append(f"provider={provider}")
    if model:
        log_lines.append(f"model={model}")
    if stage:
        log_lines.append(f"stage={stage}")
    if classification:
        log_lines.append(f"classification={classification}")
    if root_exception is not None:
        log_lines.append(f"exception_class={type(root_exception).__name__}")
    if root_summary:
        log_lines.append(f"root_cause={root_summary}")
    if outcome:
        log_lines.append(f"outcome={sanitize_secret_values(str(outcome))}")
    if effective_details:
        log_lines.append(f"details={effective_details}")
    if provider_attempts:
        attempts_log = "; ".join(
            f"{attempt.get('provider') or 'unknown'}"
            f"[{attempt.get('status') or 'FAILED'}]"
            f"/{attempt.get('classification') or 'unknown'}="
            f"{sanitize_secret_values(attempt.get('error') or 'unknown error')}"
            for attempt in provider_attempts
        )
        log_lines.append(f"attempts={attempts_log}")
    if extra:
        log_lines.append(f"extra={_safe_extra_text(extra)}")

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
    if classification:
        message_lines.append(f"Классификация: <code>{html.escape(classification)}</code>")
    if root_exception is not None:
        message_lines.append(f"Исключение: <code>{html.escape(type(root_exception).__name__)}</code>")
    if root_summary:
        message_lines.append(f"Корневая причина: <code>{html.escape(_shorten(root_summary, 1800))}</code>")
    if outcome:
        message_lines.append(f"Результат: <code>{html.escape(_shorten(str(outcome), 300))}</code>")
    if classification_description and not effective_details:
        message_lines.append(f"Описание: <code>{html.escape(classification_description)}</code>")
    if effective_details:
        message_lines.append(f"Ошибка: <code>{html.escape(_shorten(effective_details, 1800))}</code>")
    if provider_attempts:
        message_lines.append("Попытки провайдеров:")
        for index, attempt in enumerate(provider_attempts, start=1):
            attempt_provider = html.escape(attempt.get("provider") or "неизвестно")
            attempt_model = attempt.get("model")
            model_suffix = f" (<code>{html.escape(attempt_model)}</code>)" if attempt_model else ""
            attempt_status = html.escape(attempt.get("status") or "FAILED")
            attempt_classification = attempt.get("classification")
            classification_suffix = (
                f" · классификация: <code>{html.escape(attempt_classification)}</code>"
                if attempt_classification
                else ""
            )
            attempt_exception_class = attempt.get("exception_class")
            exception_suffix = (
                f" · класс: <code>{html.escape(attempt_exception_class)}</code>"
                if attempt_exception_class
                else ""
            )
            raw_err = attempt.get("error") or "неизвестная ошибка"
            attempt_error = html.escape(_shorten(sanitize_secret_values(raw_err), 900))
            message_lines.append(
                f"{index}. <b>{attempt_provider}</b>{model_suffix} · статус: <code>{attempt_status}</code>"
                f"{classification_suffix}{exception_suffix}\n"
                f"Ошибка: <code>{attempt_error}</code>"
            )
    if extra:
        safe_extra = _shorten(_safe_extra_text(extra), 1200)
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
