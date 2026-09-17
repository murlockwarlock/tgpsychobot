import html
import json
import logging
import os
import re
import traceback
from datetime import datetime, timedelta
from typing import Any, Sequence

from aiogram import Bot

from alert_cooldown import KeyedAlertCooldown
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
    "output_budget_exhausted": "Исчерпан лимит токенов вывода (output budget exhausted)",
    "provider_rejection": "Провайдер отклонил запрос или платеж",
    "provider_5xx": "Внутренняя ошибка или перегрузка сервиса провайдера",
    "empty_response": "Провайдер вернул пустой ответ",
    "invalid_response": "Провайдер вернул пустой или некорректный ответ",
    "configuration": "Ошибка конфигурации приложения или провайдера",
    "application_internal": "Внутренняя ошибка приложения",
    "unknown": "Неизвестная ошибка",
}


def _extract_transport_http_status(exception: Exception, *, include_context: bool = True) -> int | None:
    for item in exception_chain(exception, include_context=include_context):
        resp = getattr(item, "response", None)
        if resp is not None:
            sc = getattr(resp, "status_code", None)
            if isinstance(sc, int) and 100 <= sc <= 599:
                return sc
        for attr in ("http_status", "status_code", "status"):
            val = getattr(item, attr, None)
            if isinstance(val, int) and 100 <= val <= 599:
                return val
    return None


def _extract_provider_code(exception: Exception, *, include_context: bool = True) -> int | str | None:
    for item in exception_chain(exception, include_context=include_context):
        code = getattr(item, "provider_code", None)
        if code is not None:
            return code
        content = getattr(item, "content", None)
        if isinstance(content, dict):
            c = content.get("code") or content.get("provider_code")
            if c is not None:
                return c
    return None


def _exception_status_codes(exception: Exception, *, include_context: bool = True) -> set[int]:
    codes: set[int] = set()
    for item in exception_chain(exception, include_context=include_context):
        candidates = [getattr(item, "status_code", None), getattr(item, "status", None), getattr(item, "http_status", None)]
        response = getattr(item, "response", None)
        if response is not None:
            candidates.extend([getattr(response, "status_code", None), getattr(response, "status", None)])
        for candidate in candidates:
            try:
                if isinstance(candidate, int) and 100 <= candidate <= 599:
                    codes.add(candidate)
                elif isinstance(candidate, str) and candidate.isdigit() and 100 <= int(candidate) <= 599:
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
    """Classify an external-call failure using the complete exception chain with deterministic precedence."""
    if exception is None:
        return "unknown", _ERROR_CLASS_DESCRIPTIONS["unknown"]

    chain = exception_chain(exception, include_context=include_context)
    type_text = " ".join(type(item).__name__ for item in chain).lower()
    error_text = " ".join(str(item) for item in chain).lower()
    combined = f"{type_text} {error_text}"
    transport_status = _extract_transport_http_status(exception, include_context=include_context)
    status_codes = _exception_status_codes(exception, include_context=include_context)
    if transport_status is not None:
        status_codes.add(transport_status)

    # 1. Transport Timeout Wrapping (Highest Precedence)
    # Incident invariant: ReadTimeout -> SSLWantReadError MUST classify as "timeout", not "network_ssl".
    if (
        any(marker in combined for marker in ("timeouterror", "readtimeout", "connecttimeout", "apitimeouterror"))
        or any(type(item).__name__.lower() in ("timeouterror", "readtimeout", "connecttimeout", "apitimeouterror") for item in chain)
        or "timeout" in combined
        or "timed out" in combined
    ):
        code = "timeout"
        return code, _ERROR_CLASS_DESCRIPTIONS[code]

    # 2. Authoritative Explicit Classification Attribute
    for item in [exception] + chain:
        explicit_cls = getattr(item, "classification", None)
        if explicit_cls and explicit_cls in _ERROR_CLASS_DESCRIPTIONS:
            code = explicit_cls
            return code, _ERROR_CLASS_DESCRIPTIONS[code]

    # 3. Transport HTTP Status (Non-2xx is authoritative over body provider_code)
    # Conflict rule: HTTP 503 + provider_code 402 -> provider_5xx (transport status wins)
    if transport_status is not None and transport_status != 200:
        if 500 <= transport_status < 600:
            code = "provider_5xx"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif transport_status == 429:
            code = "rate_limit"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif transport_status == 401:
            code = "auth"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif transport_status == 403:
            if any(m in combined for m in ("api key", "unauthorized", "invalid key", "authentication")):
                code = "auth"
            else:
                code = "forbidden_geo"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif transport_status == 402:
            code = "insufficient_balance_quota"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif transport_status in {400, 409, 422}:
            code = "provider_rejection"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]

    # 4. Body Provider Code (authoritative when transport HTTP is 200 or absent)
    # Conflict rule: HTTP 200 + provider_code 402 -> insufficient_balance_quota
    provider_code = _extract_provider_code(exception, include_context=include_context)
    if provider_code is not None:
        p_code_str = str(provider_code).strip()
        if p_code_str == "401":
            code = "auth"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif p_code_str == "402":
            code = "insufficient_balance_quota"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif p_code_str == "403":
            code = "forbidden_geo"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif p_code_str == "429":
            code = "rate_limit"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif p_code_str in {"400", "409", "422"}:
            code = "provider_rejection"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]
        elif p_code_str in {"500", "502", "503", "504"}:
            code = "provider_5xx"
            return code, _ERROR_CLASS_DESCRIPTIONS[code]

    # 5. Standalone SSL & Connection Exceptions
    if any(marker in combined for marker in (
        "sslerror",
        "ssl:",
        "tls",
        "certificate verify failed",
        "unexpected_eof",
        "wrong version number",
        "sslwantreaderror",
        "sslwantwriteerror",
    )):
        code = "network_ssl"
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

    # 6. Generic Type / Keyword Heuristics
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
    elif any(marker in combined for marker in ("output budget exhausted", "output_budget_exhausted")):
        code = "output_budget_exhausted"
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


_terminal_failure_cooldown = KeyedAlertCooldown(timedelta(minutes=30))
_output_budget_cooldown = KeyedAlertCooldown(timedelta(minutes=30))


def extract_error_metadata(
    exception: Exception | None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Extract structured failure diagnostics and provider metadata without fragile regexes."""
    if exception is None:
        return {
            "error_type": None,
            "error_message": None,
            "error_classification": None,
            "http_status": None,
            "finish_reason": None,
            "diagnostics": None,
            "provider_response_payload": None,
        }

    chain = exception_chain(exception, include_context=True)
    root = root_cause_exception(exception) or exception
    error_type = type(root).__name__ if root is not None else type(exception).__name__
    raw_msg = str(exception).strip() or str(root).strip()
    error_message = sanitize_secret_values(raw_msg)

    http_status = getattr(exception, "http_status", None)
    if http_status is None:
        http_status = _extract_transport_http_status(exception)

    finish_reason = getattr(exception, "finish_reason", None)
    diagnostics = getattr(exception, "diagnostics", None)
    provider_response_payload = getattr(exception, "provider_response_payload", None)

    for item in chain:
        if finish_reason is None and hasattr(item, "finish_reason"):
            finish_reason = getattr(item, "finish_reason")
        if diagnostics is None and hasattr(item, "diagnostics"):
            diagnostics = getattr(item, "diagnostics")
        if provider_response_payload is None and hasattr(item, "provider_response_payload"):
            provider_response_payload = getattr(item, "provider_response_payload")
        if provider_response_payload is None and hasattr(item, "response"):
            resp = getattr(item, "response")
            if hasattr(resp, "text"):
                provider_response_payload = getattr(resp, "text")

    classification = getattr(exception, "classification", None)
    if not classification:
        classification, _ = classify_external_error(exception, provider=provider)

    return {
        "error_type": error_type,
        "error_message": error_message,
        "error_classification": classification,
        "http_status": http_status,
        "finish_reason": finish_reason,
        "diagnostics": diagnostics,
        "provider_response_payload": provider_response_payload,
    }


async def _dispatch_admin_alert_text(bot: Bot | None, text: str) -> bool:
    """Best-effort alert delivery to all configured administrators."""
    try:
        admin_ids = await get_all_admin_ids()
        if not admin_ids:
            return False

        if bot is not None:
            for admin_id in admin_ids:
                try:
                    await bot.send_message(admin_id, text, parse_mode="HTML")
                except Exception as exc:
                    logging.getLogger(__name__).error("Failed to send admin alert to %s: %s", admin_id, exc)
            return True

        bot_token = os.getenv("BOT_TOKEN")
        if not bot_token:
            return False

        from telegram_client import create_telegram_bot
        async with create_telegram_bot(bot_token) as t_bot:
            for admin_id in admin_ids:
                try:
                    await t_bot.send_message(admin_id, text, parse_mode="HTML")
                except Exception as exc:
                    logging.getLogger(__name__).error("Failed to send admin alert to %s: %s", admin_id, exc)
            return True
    except Exception as exc:
        logging.getLogger(__name__).error("Admin alert delivery failed: %s", exc)
        return False


async def send_terminal_ai_failure_alert(
    bot: Bot | None = None,
    *,
    platform: str = "telegram",
    user_id: int | None = None,
    user: Any | None = None,
    chat_id: int | None = None,
    bot_name: str | None = None,
    request_type: str = "chat",
    primary_provider: str | None = None,
    primary_model: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    title: str | None = None,
    stage: str | None = None,
    details: str | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    exception: Exception | None = None,
    classification: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    ai_log_ids: Sequence[int | str] | None = None,
    provider_attempts: Sequence[dict[str, Any]] | None = None,
    attempts: Sequence[dict[str, Any]] | None = None,
    dialogue_id: int | None = None,
    topic_id: int | None = None,
    topic_name: str | None = None,
    username: str | None = None,
    full_name: str | None = None,
) -> bool:
    """Send realtime terminal AI failure alert to Telegram admins with 30m cooldown."""
    if user is not None:
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or user_id
        chat_id = getattr(user, "chat_id", None) or chat_id
        username = getattr(user, "username", None) or username
        full_name = getattr(user, "full_name", None) or getattr(user, "name", None) or full_name
        platform = getattr(user, "platform", platform) or platform
        dialogue_id = getattr(user, "dialogue_id", None) or dialogue_id
        topic_id = getattr(user, "topic_id", None) or topic_id
        topic_name = getattr(user, "topic_name", None) or topic_name

    primary_provider = primary_provider or provider or "unknown"
    primary_model = primary_model or model or "—"
    provider_attempts = provider_attempts or attempts
    error_message = error_message or details

    platform_norm = "MAX" if platform and platform.lower() == "max" else "Telegram"
    b_name = bot_name or os.getenv("BOT_NAME") or os.getenv("BOT_USERNAME") or "tgpsychobot"
    cls_str = classification or "provider_failure"
    primary_p = primary_provider or "primary"
    fb_p = fallback_provider or "none"

    if request_type == "vision":
        fingerprint = f"vision:{platform_norm}:{b_name}:{primary_p}:{fb_p}:{cls_str}"
        alert_title = title or f"🚨 <b>[{platform_norm}] Сбой ИИ при анализе изображения</b>\n\n"
    else:
        fingerprint = f"{platform_norm}:{b_name}:{primary_p}:{fb_p}:{cls_str}"
        alert_title = title or f"🚨 <b>[{platform_norm}] Сбой ИИ в текстовом диалоге</b>\n\n"

    if not _terminal_failure_cooldown.should_send(fingerprint):
        return False

    id_label = f"MAX ID: <code>{user_id}</code>" if platform_norm == "MAX" else f"Telegram ID: <code>{user_id}</code>"
    chat_line = f"Chat ID: <code>{chat_id}</code>\n" if chat_id is not None else ""

    err_type_str = error_type or (type(exception).__name__ if exception is not None else "AIServiceError")
    raw_err = error_message or (str(exception) if exception is not None else "Unknown error")
    if request_type == "vision":
        from vision_reliability import sanitize_vision_request_payload
        raw_err = sanitize_vision_request_payload(raw_err)
    err_msg_str = sanitize_secret_values(str(raw_err))
    err_msg_short = _shorten(err_msg_str, 500)

    if ai_log_ids:
        logs_str = ", ".join(f"#{i}" if str(i).isdigit() else str(i) for i in ai_log_ids)
        label = "AI Log:" if len(ai_log_ids) == 1 else "AI Log IDs:"
        ai_log_line = f"\n{label} {logs_str}"
    else:
        ai_log_line = "\nAI Log: не удалось сохранить"

    fb_lines = ""
    if fallback_provider:
        fb_lines = f"Fallback: {html.escape(fallback_provider)} / {html.escape(fallback_model or '—')} ❌\n"

    attempts_block = ""
    if provider_attempts:
        lines = []
        for att in provider_attempts:
            p = att.get("provider", "Unknown")
            m = att.get("model", "—")
            st = att.get("status", "UNKNOWN")
            cls = att.get("classification")
            mark = "✅" if st in ("SUCCESS", "success", True) else "❌"
            cls_info = f" ({cls})" if cls else ""
            lines.append(f"• {html.escape(str(p))} / {html.escape(str(m))} {mark}{cls_info}")
        attempts_block = "\nПопытки:\n" + "\n".join(lines) + "\n"

    text = (
        f"{alert_title}"
        f"Бот: <code>{html.escape(b_name)}</code>\n"
        f"{id_label}\n"
        f"{chat_line}"
        f"Primary: {html.escape(primary_provider)} / {html.escape(primary_model)} ❌\n"
        f"{fb_lines}"
        f"{attempts_block}\n"
        f"Ошибка:\n"
        f"<code>{html.escape(err_type_str)}</code> / <code>{html.escape(err_msg_short)}</code>\n\n"
        f"Классификация:\n"
        f"<code>{html.escape(cls_str)}</code>\n"
        f"{ai_log_line}"
    )

    return await _dispatch_admin_alert_text(bot, text)


async def send_output_budget_exhausted_alert(
    bot: Bot | None = None,
    *,
    platform: str = "telegram",
    user_id: int | None = None,
    user: Any | None = None,
    chat_id: int | None = None,
    bot_name: str | None = None,
    request_type: str = "chat",
    provider: str = "Deepseek",
    model: str = "",
    finish_reason: str = "length",
    visible_content_length: int = 0,
    reasoning_content_length: int = 0,
    max_tokens: int | str = "65536",
    ai_log_id: int | str | None = None,
    details: str | None = None,
    exception: Exception | None = None,
    attempts: Sequence[dict[str, Any]] | None = None,
    provider_attempts: Sequence[dict[str, Any]] | None = None,
    dialogue_id: int | None = None,
    topic_id: int | None = None,
    topic_name: str | None = None,
    username: str | None = None,
    full_name: str | None = None,
) -> bool:
    """Send alert when AI provider exhausts output budget on tokens (30m cooldown)."""
    if user is not None:
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or user_id
        chat_id = getattr(user, "chat_id", None) or chat_id
        username = getattr(user, "username", None) or username
        full_name = getattr(user, "full_name", None) or getattr(user, "name", None) or full_name
        platform = getattr(user, "platform", platform) or platform

    platform_norm = "MAX" if platform and platform.lower() == "max" else "Telegram"
    b_name = bot_name or os.getenv("BOT_NAME") or os.getenv("BOT_USERNAME") or "tgpsychobot"

    if request_type == "vision":
        fingerprint = f"vision:{b_name}:{provider}:{model}:output_budget_exhausted"
        title = f"⚠️ <b>[Vision] {html.escape(provider)} исчерпал output budget</b>\n\n"
    else:
        fingerprint = f"{b_name}:{provider}:{model}:output_budget_exhausted"
        title = f"⚠️ <b>[AI] {html.escape(provider)} исчерпал output budget</b>\n\n"

    if not _output_budget_cooldown.should_send(fingerprint):
        return False

    ai_log_str = f"#{ai_log_id}" if ai_log_id is not None and str(ai_log_id).isdigit() else (str(ai_log_id) if ai_log_id else "не удалось сохранить")
    id_label = f"User ID: <code>{user_id if user_id is not None else 'не указан'}</code>\n"
    chat_line = f"Chat ID: <code>{chat_id}</code>\n" if chat_id is not None else ""

    text = (
        f"{title}"
        f"Бот: <code>{html.escape(b_name)}</code>\n"
        f"Платформа: <b>{platform_norm}</b>\n"
        f"{id_label}"
        f"{chat_line}"
        f"Model: <code>{html.escape(model)}</code>\n"
        f"finish_reason: <code>{html.escape(finish_reason)}</code>\n"
        f"visible_content_length: <code>{visible_content_length}</code>\n"
        f"reasoning_content_length: <code>{reasoning_content_length}</code>\n"
        f"max_tokens: <code>{max_tokens}</code>\n"
        f"AI Log: {ai_log_str}"
    )

    return await _dispatch_admin_alert_text(bot, text)


async def send_ai_fallback_used_alert(
    bot: Bot | None = None,
    *,
    platform: str = "telegram",
    user_id: int | None = None,
    user: Any | None = None,
    chat_id: int | None = None,
    bot_name: str | None = None,
    request_type: str = "chat",
    primary_provider: str | None = None,
    primary_model: str | None = None,
    failed_provider: str | None = None,
    failed_model: str | None = None,
    fallback_provider: str = "",
    fallback_model: str | None = None,
    primary_error: str | Exception | None = None,
    reason: str | Exception | None = None,
    failure_reason: str | Exception | None = None,
    ai_log_ids: Sequence[int | str] | None = None,
    provider_attempts: Sequence[dict[str, Any]] | None = None,
    attempts: Sequence[dict[str, Any]] | None = None,
    dialogue_id: int | None = None,
    topic_id: int | None = None,
    topic_name: str | None = None,
    username: str | None = None,
    full_name: str | None = None,
) -> bool:
    """Send realtime notification that reserve AI provider was used successfully."""
    if user is not None:
        user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or user_id
        chat_id = getattr(user, "chat_id", None) or chat_id
        username = getattr(user, "username", None) or username
        full_name = getattr(user, "full_name", None) or getattr(user, "name", None) or full_name
        platform = getattr(user, "platform", platform) or platform

    primary_provider = primary_provider or failed_provider or "Unknown"
    primary_model = primary_model or failed_model or "—"
    primary_error = primary_error or reason or failure_reason
    fallback_model = fallback_model or ""
    platform_norm = "MAX" if platform.lower() == "max" else "Telegram"
    b_name = bot_name or os.getenv("BOT_NAME") or os.getenv("BOT_USERNAME") or "tgpsychobot"
    id_label = f"MAX ID: <code>{user_id}</code>" if platform_norm == "MAX" else f"Telegram ID: <code>{user_id}</code>"
    chat_line = f"Chat ID: <code>{chat_id}</code>\n" if chat_id is not None else ""

    raw_err = str(primary_error) if primary_error is not None else "Не зафиксирована"
    if request_type == "vision":
        from vision_reliability import sanitize_vision_request_payload
        raw_err = sanitize_vision_request_payload(raw_err)
    err_str = sanitize_secret_values(str(raw_err))
    err_short = _shorten(err_str, 300)

    title_type = "при анализе изображения" if request_type == "vision" else "в текстовом диалоге"

    eff_attempts = provider_attempts or attempts
    attempts_block = ""
    if eff_attempts:
        lines = []
        for att in eff_attempts:
            p = att.get("provider", "Unknown")
            m = att.get("model", "—")
            st = att.get("status", "UNKNOWN")
            cls = att.get("classification")
            mark = "✅" if (st in ("SUCCESS", "success", True) or str(st).upper() == "SUCCESS") else "❌"
            cls_info = f" ({cls})" if cls else ""
            lines.append(f"• {html.escape(str(p))} / {html.escape(str(m))} {mark}{cls_info}")
        attempts_block = "\nПопытки:\n" + "\n".join(lines) + "\n"

    ai_log_line = ""
    if ai_log_ids:
        logs_str = ", ".join(f"#{i}" if str(i).isdigit() else str(i) for i in ai_log_ids)
        ai_log_line = f"\nAI Log IDs: {logs_str}"

    text = (
        f"ℹ️ <b>[{platform_norm}] Использован резервный AI-провайдер {title_type}</b>\n\n"
        f"Бот: <code>{html.escape(b_name)}</code>\n"
        f"{id_label}\n"
        f"{chat_line}"
        f"Основной: {html.escape(primary_provider)} / {html.escape(primary_model or '—')} ❌\n"
        f"Резервный: {html.escape(fallback_provider)} / {html.escape(fallback_model or '—')} ✅\n"
        f"{attempts_block}"
        f"Причина переключения:\n"
        f"<code>{html.escape(err_short)}</code>\n"
        f"{ai_log_line}"
    )
    return await _dispatch_admin_alert_text(bot, text)


