from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import mimetypes
import random
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx

from ai_request_context import AIRequestLayout, _capture_ai_request, build_openai_chat_messages
from provider_models import (
    DEEPGRAM_DEFAULT_MODEL,
    DEEPGRAM_TRANSCRIPTION_MODELS,
    OPENROUTER_MODEL_SPECS,
    PERPLEXITY_MODES,
    PERPLEXITY_STATIC_DIRECT_MODELS,
    is_perplexity_preset,
)


log = logging.getLogger(__name__)


class ProviderAdapterError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, category: str = "provider"):
        super().__init__(message)
        self.http_status = status
        self.classification = category
        self.category = category


_ADAPTER_CLASSIFICATION_MAP = {
    "auth": "auth",
    "quota": "insufficient_balance_quota",
    "insufficient_balance_quota": "insufficient_balance_quota",
    "rate_limit": "rate_limit",
    "timeout": "timeout",
    "server_error": "provider_5xx",
    "provider_5xx": "provider_5xx",
    "network": "network_connection",
    "network_connection": "network_connection",
    "invalid_request": "provider_rejection",
    "provider_rejection": "provider_rejection",
    "invalid_model": "configuration",
    "configuration": "configuration",
    "malformed_response": "invalid_response",
    "invalid_response": "invalid_response",
    "empty_response": "empty_response",
    "output_budget_exhausted": "output_budget_exhausted",
    "provider": "unknown",
    "unknown": "unknown",
}


def normalize_provider_error_classification(category: str | None) -> str:
    return _ADAPTER_CLASSIFICATION_MAP.get(str(category or "provider"), "unknown")


async def _mark_activity(activity_tracker: Any | None) -> None:
    if activity_tracker is None:
        return
    try:
        await activity_tracker.mark_outbound_attempt_once()
    except Exception as exc:
        log.warning("Failed to mark activity before provider request: %s", exc)


@dataclass(frozen=True)
class ProviderCitation:
    title: str
    url: str


def _capture_payload(payload: dict[str, Any]) -> dict[str, Any]:
    captured = copy.deepcopy(payload)
    for message in captured.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("image_url"), dict):
                item["image_url"]["url"] = "<redacted_media_url>"
    return captured


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


def build_openrouter_payload(
    layout: AIRequestLayout,
    model: str,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    spec = OPENROUTER_MODEL_SPECS.get(model)
    if spec is None or not spec.text:
        raise ProviderAdapterError("Недоступная модель OpenRouter", category="invalid_model")
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_openai_chat_messages(layout),
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def build_openrouter_vision_layout(
    layout: AIRequestLayout,
    image_bytes: bytes,
    *,
    mime_type: str = "image/jpeg",
    user_instruction: str = "",
) -> AIRequestLayout:
    if not image_bytes:
        raise ProviderAdapterError("Пустой файл изображения", category="invalid_request")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ProviderAdapterError("Изображение превышает лимит 20 МБ", category="invalid_request")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    content = [
        {"type": "text", "text": user_instruction or "Проанализируй изображение."},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
    ]
    return layout.with_current_user_content(content)


def build_perplexity_payload(
    layout: AIRequestLayout,
    preset_or_model: str | None = None,
    *,
    preset: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    target = preset_or_model or preset or model or ""
    is_preset = is_perplexity_preset(target) or is_perplexity_preset(preset)
    effective_preset = preset if preset is not None else (target if is_preset else None)
    effective_model = model if model is not None else (target if not is_preset else None)

    if not is_preset and (not effective_model or "/" not in effective_model):
        raise ProviderAdapterError(f"Недопустимая модель или режим Perplexity: '{target}'", category="invalid_model")

    # In Perplexity Agent API:
    # - Preset defines model/tool/search execution profile.
    # - Product instructions provide behavioral/clinical system instructions for psychobot,
    #   ensuring strict compliance with persona guardrails.
    # - Direct models specify the underlying LLM, and we explicitly attach the web_search tool
    #   to guarantee web grounding and citations per product design.
    instruction_blocks = layout.ordered_instruction_blocks
    instructions = "\n\n".join(block for block in instruction_blocks if block)
    if instructions:
        instructions += "\n\nИспользуй web_search для актуальных фактов и добавляй inline citations [n] к утверждениям, основанным на найденных источниках."

    # Multi-turn conversation handling:
    # Perplexity Agent API expects structured message items with 'role' and 'content'
    # when history is present, rather than flattening dialogue into a single string.
    if layout.history:
        structured_input: list[dict[str, Any]] = []
        for message in layout.history:
            content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
            structured_input.append({
                "role": message.role,
                "content": content,
            })
        if layout.current_user_content is not None:
            content = layout.current_user_content if isinstance(layout.current_user_content, str) else json.dumps(layout.current_user_content, ensure_ascii=False)
            structured_input.append({
                "role": "user",
                "content": content,
            })
        input_value: Any = structured_input
    else:
        if layout.current_user_content is not None:
            input_value = layout.current_user_content if isinstance(layout.current_user_content, str) else json.dumps(layout.current_user_content, ensure_ascii=False)
        else:
            input_value = ""

    payload: dict[str, Any] = {
        "input": input_value,
    }
    if effective_preset is not None:
        payload["preset"] = effective_preset
    if effective_model is not None:
        payload["model"] = effective_model
    if instructions:
        payload["instructions"] = instructions

    # Direct models: Explicitly enable web search tool and force tool_choice to ground the LLM with search citations.
    # Presets: Omit tools and tool_choice so that Perplexity manages its preset search & tool profile.
    if effective_model is not None and not is_preset:
        payload["tools"] = [{"type": "web_search"}]
        payload["tool_choice"] = {"type": "web_search"}

    # Anthropic models via Perplexity REQUIRE max_output_tokens.
    is_anthropic = bool(effective_model and effective_model.startswith("anthropic/"))
    if is_anthropic:
        resolved_tokens = max_output_tokens if max_output_tokens is not None else 8192
        payload["max_output_tokens"] = resolved_tokens
    elif max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    if temperature is not None and not is_preset:
        payload["temperature"] = float(temperature)

    if is_anthropic and "max_output_tokens" not in payload:
        raise ProviderAdapterError(
            "Для моделей Anthropic в Perplexity параметр max_output_tokens обязателен.",
            category="configuration",
        )

    return payload


def _is_balance_quota_evidence(msg: str, err_type: str = "", code: Any = None) -> bool:
    if code in {402, "insufficient_quota", "insufficient_credits", "insufficient_balance"}:
        return True
    combined = f"{msg} {err_type}".lower()
    balance_markers = (
        "insufficient_quota",
        "insufficient_credits",
        "insufficient_balance",
        "out of credits",
        "credit balance",
        "exhausted your credits",
        "balance too low",
        "quota exceeded",
        "billing",
    )
    if any(marker in combined for marker in balance_markers):
        return True
    if "credit" in combined or "balance" in combined or "quota" in combined:
        return True
    return False


def _classify_perplexity_error_dict(err: dict[str, Any]) -> str:
    code = err.get("code")
    msg = str(err.get("message") or "").lower()
    err_type = str(err.get("type") or "").lower()

    if _is_balance_quota_evidence(msg, err_type, code):
        return "insufficient_balance_quota"
    if code == 401 or "auth" in err_type or "invalid_api_key" in msg:
        return "auth"
    if code == 403:
        if any(k in msg or k in err_type for k in ("tier", "permission", "access", "unauthorized", "account")):
            return "auth"
        return "provider_rejection"
    if code == 429 or "rate_limit" in err_type or "rate limit" in msg:
        return "rate_limit"
    if code in {400, 404} or err_type in {"invalid_request_error", "validation_error"}:
        if "model" in msg and any(k in msg for k in ("not found", "unknown", "invalid model", "does not exist")):
            return "configuration"
        if "max_output_tokens" in msg or "max_tokens" in msg:
            return "configuration"
        if any(k in msg for k in ("content", "policy", "safety", "harmful", "moderation", "rejection", "rejected")):
            return "provider_rejection"
        if code == 404:
            return "configuration"
        return "provider_rejection"
    if (isinstance(code, int) and code >= 500) or "internal" in err_type or "server" in err_type or "backend" in msg:
        return "provider_5xx"
    return "provider_rejection"


def _classify_perplexity_http_error(status: int, data: dict[str, Any], raw_text: str) -> str:
    err = data.get("error") if isinstance(data.get("error"), dict) else {}
    msg = (str(err.get("message") or "") + " " + raw_text).lower()
    err_type = str(err.get("type") or "").lower()
    code = err.get("code") or status

    if status == 402 or _is_balance_quota_evidence(msg, err_type, code):
        return "insufficient_balance_quota"
    if status == 401 or "invalid_api_key" in msg or "auth" in err_type:
        return "auth"
    if status == 403:
        if any(k in msg or k in err_type for k in ("permission", "tier", "access", "unauthorized", "forbidden", "account")):
            return "auth"
        return "provider_rejection"
    if status == 429 or "rate_limit" in err_type:
        return "rate_limit"
    if status == 400:
        if "model" in msg and any(k in msg for k in ("not found", "unknown", "invalid model", "does not exist")):
            return "configuration"
        if "max_output_tokens" in msg or "max_tokens" in msg:
            return "configuration"
        if any(k in msg for k in ("content", "policy", "safety", "harmful", "moderation", "rejection", "rejected")):
            return "provider_rejection"
        return "provider_rejection"
    if status == 404:
        return "configuration"
    if status >= 500:
        return "provider_5xx"
    return "provider"


def _populate_perplexity_diagnostics(
    request_capture: dict[str, Any] | None,
    *,
    status_code: int | None = None,
    headers: Any = None,
    data: dict[str, Any] | None = None,
    error_dict: dict[str, Any] | None = None,
    retry_after: float | str | None = None,
    attempt_count: int | None = None,
    duration_ms: float | None = None,
    classification: str | None = None,
    is_retryable: bool | None = None,
    will_retry: bool | None = None,
) -> None:
    if request_capture is None:
        return
    if status_code is not None:
        request_capture["http_status"] = status_code
    if attempt_count is not None:
        request_capture["attempt_count"] = attempt_count
    if duration_ms is not None:
        request_capture["duration_ms"] = duration_ms
    if classification is not None:
        request_capture["classification"] = classification
    if is_retryable is not None:
        request_capture["is_retryable"] = is_retryable
    if will_retry is not None:
        request_capture["will_retry"] = will_retry
    if headers is not None:
        header_req_id = getattr(headers, "get", lambda k: None)("x-request-id")
        if header_req_id:
            request_capture["x_request_id"] = header_req_id
        if retry_after is None:
            retry_after_hdr = getattr(headers, "get", lambda k: None)("retry-after")
            if retry_after_hdr:
                retry_after = retry_after_hdr
    if retry_after is not None:
        request_capture["retry_after"] = retry_after

    payload = data or {}
    if payload.get("id"):
        request_capture["response_id"] = payload["id"]
    if payload.get("status"):
        request_capture["response_status"] = payload["status"]
    if payload.get("model"):
        request_capture["response_model"] = payload["model"]
    if payload.get("service_tier"):
        request_capture["service_tier"] = payload["service_tier"]
    if isinstance(payload.get("usage"), dict):
        request_capture["usage"] = payload["usage"]
    if payload.get("cost") is not None:
        request_capture["cost"] = payload["cost"]
    incomplete_details = payload.get("incomplete_details")
    if isinstance(incomplete_details, dict) and incomplete_details.get("reason"):
        request_capture["incomplete_reason"] = str(incomplete_details["reason"])

    err = error_dict or (payload.get("error") if isinstance(payload.get("error"), dict) else None)
    if isinstance(err, dict):
        if err.get("type"):
            request_capture["provider_error_type"] = err["type"]
        if err.get("code") is not None:
            request_capture["provider_error_code"] = err["code"]
        if err.get("message"):
            request_capture["provider_error_message"] = err["message"]


def _extract_perplexity_text(payload: dict[str, Any], request_capture: dict[str, Any] | None = None) -> str:
    status = payload.get("status")
    if not status or not isinstance(status, str):
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            classification="invalid_response",
            is_retryable=False,
            will_retry=False,
        )
        raise ProviderAdapterError("Отсутствует обязательный статус ответа Perplexity", category="invalid_response")

    normalized_status = status.strip().lower()
    if normalized_status == "failed":
        err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        msg = err.get("message") or "Запрос к Perplexity завершился ошибкой"
        cat = _classify_perplexity_error_dict(err)
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            error_dict=err,
            classification=cat,
            is_retryable=False,
            will_retry=False,
        )
        raise ProviderAdapterError(f"Perplexity статус failed: {msg}", category=cat)
    if normalized_status == "incomplete":
        incomplete_details = payload.get("incomplete_details") if isinstance(payload.get("incomplete_details"), dict) else {}
        reason = incomplete_details.get("reason")
        if reason == "max_output_tokens":
            cat = "output_budget_exhausted"
            err_msg = "Perplexity исчерпал лимит токенов вывода (incomplete: max_output_tokens)"
        else:
            cat = "invalid_response"
            err_msg = f"Perplexity вернул неполный ответ (incomplete: {reason or 'unknown'})"
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            classification=cat,
            is_retryable=False,
            will_retry=False,
        )
        if request_capture is not None and reason:
            request_capture["incomplete_reason"] = str(reason)
        raise ProviderAdapterError(err_msg, category=cat)
    if normalized_status == "cancelled":
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            classification="provider_rejection",
            is_retryable=False,
            will_retry=False,
        )
        raise ProviderAdapterError("Запрос к Perplexity был отменён (cancelled)", category="provider_rejection")
    if normalized_status in {"queued", "in_progress"}:
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            classification="invalid_response",
            is_retryable=False,
            will_retry=False,
        )
        raise ProviderAdapterError(f"Неожиданный промежуточный статус Perplexity: {normalized_status}", category="invalid_response")
    if normalized_status != "completed":
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            classification="invalid_response",
            is_retryable=False,
            will_retry=False,
        )
        raise ProviderAdapterError(f"Неизвестный статус ответа Perplexity: {normalized_status}", category="invalid_response")

    if payload.get("error"):
        err = payload["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        cat = _classify_perplexity_error_dict(err if isinstance(err, dict) else {})
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            error_dict=err if isinstance(err, dict) else None,
            classification=cat,
            is_retryable=False,
            will_retry=False,
        )
        raise ProviderAdapterError(f"Ошибка Perplexity в ответе: {msg}", category=cat)

    output = payload.get("output")
    if isinstance(output, list):
        for item in reversed(output):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            item_role = item.get("role")
            item_status = item.get("status")
            if item_type == "message" and item_role == "assistant" and item_status == "completed":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    _populate_perplexity_diagnostics(
                        request_capture,
                        status_code=200,
                        data=payload,
                        classification="success",
                        is_retryable=False,
                        will_retry=False,
                    )
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            parts.append(part["text"])
                        elif isinstance(part, str):
                            parts.append(part)
                    if parts:
                        text_val = "\n".join(parts).strip()
                        if text_val:
                            _populate_perplexity_diagnostics(
                                request_capture,
                                status_code=200,
                                data=payload,
                                classification="success",
                                is_retryable=False,
                                will_retry=False,
                            )
                            return text_val
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    _populate_perplexity_diagnostics(
                        request_capture,
                        status_code=200,
                        data=payload,
                        classification="success",
                        is_retryable=False,
                        will_retry=False,
                    )
                    return text_val.strip()

    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        _populate_perplexity_diagnostics(
            request_capture,
            status_code=200,
            data=payload,
            classification="success",
            is_retryable=False,
            will_retry=False,
        )
        return direct.strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        text = _extract_message_text(content)
        if text:
            _populate_perplexity_diagnostics(
                request_capture,
                status_code=200,
                data=payload,
                classification="success",
                is_retryable=False,
                will_retry=False,
            )
            return text
    _populate_perplexity_diagnostics(
        request_capture,
        status_code=200,
        data=payload,
        classification="empty_response",
        is_retryable=False,
        will_retry=False,
    )
    raise ProviderAdapterError("Perplexity вернул пустой ответ", category="empty_response")


def _is_http_url(raw_url: Any) -> bool:
    """Check if the provided raw_url is a valid HTTP or HTTPS URL."""
    value = str(raw_url or "").strip()
    if not value:
        return False
    try:
        parsed = urlsplit(value)
    except Exception:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _normalize_citation_url(raw_url: str) -> tuple[str, str]:
    """Conservative URL normalization helper for citation identity.
    Normalizes scheme (lower), hostname (lower), strips default ports (:80, :443),
    and strips trailing slash for non-root paths (e.g. /guide/ -> /guide).
    Returns (dedup_key, display_url).
    """
    raw_url = str(raw_url or "").strip()
    try:
        parsed = urlsplit(raw_url)
    except Exception:
        return raw_url, raw_url

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path
    if path and path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    dedup_key = urlunsplit((scheme, netloc, path, parsed.query, parsed.fragment))
    display_url = dedup_key
    return dedup_key, display_url


def format_perplexity_response(payload: dict[str, Any], request_capture: dict[str, Any] | None = None) -> str:
    text = _extract_perplexity_text(payload, request_capture=request_capture)
    output = payload.get("output")
    if not isinstance(output, list):
        output = []

    # Citation source extraction hierarchy:
    # 1. Canonical Agent API source: output[] items with type == "search_results"
    canonical_results: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "search_results":
            results = item.get("results")
            if isinstance(results, list):
                for res in results:
                    if isinstance(res, dict) and _is_http_url(res.get("url")):
                        canonical_results.append(res)
            content_items = item.get("content")
            if isinstance(content_items, list):
                for res in content_items:
                    if isinstance(res, dict) and _is_http_url(res.get("url")):
                        canonical_results.append(res)

    raw_results: list[dict[str, Any]] = []
    if canonical_results:
        raw_results = canonical_results
    else:
        # 2. Legacy compatibility fallback: output[] items with type == "tool_call"
        # Isolated for backward compatibility only when canonical search_results are absent.
        legacy_tool_results: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_call" and item.get("tool_name") in {"web_search", "search_results", None}:
                results = item.get("results")
                if isinstance(results, list):
                    for res in results:
                        if isinstance(res, dict) and _is_http_url(res.get("url")):
                            legacy_tool_results.append(res)
                content_items = item.get("content")
                if isinstance(content_items, list):
                    for res in content_items:
                        if isinstance(res, dict) and _is_http_url(res.get("url")):
                            legacy_tool_results.append(res)

        if legacy_tool_results:
            raw_results = legacy_tool_results
        else:
            # 3. Legacy compatibility fallback: top-level citations list
            # Isolated for backward compatibility only when canonical search_results are absent.
            citations_list = payload.get("citations")
            if isinstance(citations_list, list):
                for c in citations_list:
                    if isinstance(c, str) and _is_http_url(c):
                        raw_results.append({"url": c, "title": c})

    if not raw_results:
        return text

    url_to_display_id: dict[str, str] = {}
    alias_to_display_id: dict[str, str] = {}
    unique_citations: list[dict[str, str]] = []

    for idx, res in enumerate(raw_results, start=1):
        raw_url = str(res.get("url") or "").strip()
        norm_key, display_url = _normalize_citation_url(raw_url)
        title = str(res.get("title") or "").strip() or display_url

        if norm_key not in url_to_display_id:
            display_id = str(len(unique_citations) + 1)
            url_to_display_id[norm_key] = display_id
            unique_citations.append({"id": display_id, "title": title, "url": display_url})
        else:
            display_id = url_to_display_id[norm_key]

        upstream_id = str(res.get("id") or "").strip()
        if upstream_id:
            alias_to_display_id[upstream_id] = display_id
            if upstream_id.startswith("web:"):
                num = upstream_id.split(":", 1)[1]
                alias_to_display_id[f"web:{num}"] = display_id
                alias_to_display_id[num] = display_id

        # Also register web:idx and positional idx aliases in case assistant output references them
        alias_to_display_id[f"web:{idx}"] = display_id
        if str(idx) not in alias_to_display_id:
            alias_to_display_id[str(idx)] = display_id

    if not unique_citations:
        return text

    def _replace_marker(m: re.Match) -> str:
        key = m.group(1)
        # If the inline marker is already a sequential 1-based digit corresponding
        # to an existing unique citation (e.g. [1], [2]), preserve it as-is.
        if key.isdigit():
            val = int(key)
            if 1 <= val <= len(unique_citations):
                return f"[{val}]"
        # Otherwise, resolve through known aliases (e.g. upstream provider IDs, web:N)
        canonical = alias_to_display_id.get(key)
        if canonical:
            return f"[{canonical}]"
        return m.group(0)

    formatted_text = re.sub(r"\[([a-zA-Z0-9_\-:]+)\]", _replace_marker, text)

    footer_lines = [
        f"[{c['id']}] {c['title']} — {c['url']}"
        for c in unique_citations
    ]
    return f"{formatted_text}\n\nИсточники:\n" + "\n".join(footer_lines)


def _retryable_status(status: int) -> bool:
    return status in {408, 425, 429} or status >= 500


def _status_category(status: int) -> str:
    if status in {401, 403}:
        return "auth"
    if status == 402:
        return "quota"
    if status == 429:
        return "rate_limit"
    if status in {408, 425}:
        return "timeout"
    if status == 400:
        return "invalid_request"
    if status == 404:
        return "invalid_model"
    if status >= 500:
        return "server_error"
    return "provider"


async def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    provider: str,
    request_capture: dict | None,
    activity_tracker: Any | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    await _mark_activity(activity_tracker)
    if request_capture is not None:
        _capture_ai_request(request_capture, provider=provider, endpoint=url, payload=_capture_payload(payload))
    last_error: Exception | None = None
    deadline = time.monotonic() + max(float(timeout), 0.1)
    for attempt in range(retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderAdapterError(f"Таймаут обращения к {provider}", category="timeout") from last_error
        try:
            async with httpx.AsyncClient(timeout=remaining, trust_env=False) as client:
                response = await client.post(url, headers=headers, json=payload)
            try:
                data = response.json()
            except (TypeError, ValueError):
                data = {}
            if response.status_code >= 400:
                error = ProviderAdapterError(
                    f"{provider} API вернул HTTP {response.status_code}",
                    status=response.status_code,
                    category=_status_category(response.status_code),
                )
                if _retryable_status(response.status_code) and attempt < retries:
                    await asyncio.sleep(0)
                    continue
                raise error
            if not isinstance(data, dict):
                raise ProviderAdapterError(f"{provider} вернул некорректный JSON", status=response.status_code, category="malformed_response")
            if request_capture is not None:
                request_capture["http_status"] = response.status_code
            return data
        except ProviderAdapterError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(0)
                continue
            raise ProviderAdapterError(f"Ошибка сети {provider}", category="network") from exc
        except (OSError, TypeError, ValueError) as exc:
            raise ProviderAdapterError(f"Ошибка ответа {provider}", category="provider") from exc
    raise ProviderAdapterError(f"Ошибка обращения к {provider}", category="provider") from last_error


async def call_openrouter(
    api_key: str,
    layout: AIRequestLayout,
    model: str,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    timeout: float = 60.0,
    request_capture: dict | None = None,
    activity_tracker: Any | None = None,
) -> str:
    if not api_key:
        raise ProviderAdapterError("API ключ OpenRouter не задан", category="auth")
    payload = build_openrouter_payload(layout, model, temperature=temperature, max_output_tokens=max_output_tokens)
    data = await _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
        provider="OpenRouter",
        request_capture=request_capture,
        activity_tracker=activity_tracker,
    )
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderAdapterError("OpenRouter вернул пустой ответ", category="empty_response")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    text = _extract_message_text(content)
    if not text:
        raise ProviderAdapterError("OpenRouter вернул пустой текст", category="empty_response")
    if request_capture is not None and isinstance(data.get("usage"), dict):
        request_capture["usage"] = data["usage"]
    return text


_perplexity_jitter_provider: Callable[[float, float], float] = random.uniform


def _get_perplexity_jitter(min_val: float, max_val: float) -> float:
    return _perplexity_jitter_provider(min_val, max_val)


def set_perplexity_jitter_provider(provider: Callable[[float, float], float] | None = None) -> None:
    global _perplexity_jitter_provider
    _perplexity_jitter_provider = provider if provider is not None else random.uniform


async def _post_perplexity_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    request_capture: dict | None,
    activity_tracker: Any | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    await _mark_activity(activity_tracker)
    if request_capture is not None:
        _capture_ai_request(
            request_capture,
            provider="Perplexity",
            endpoint=url,
            payload=_capture_payload(payload),
        )
    last_error: Exception | None = None
    start_time = time.monotonic()
    deadline = start_time + max(float(timeout), 0.1)
    for attempt in range(max_attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            if request_capture is not None:
                _populate_perplexity_diagnostics(
                    request_capture,
                    attempt_count=attempt,
                    duration_ms=duration_ms,
                    classification="timeout",
                    is_retryable=False,
                    will_retry=False,
                )
            raise ProviderAdapterError("Таймаут обращения к Perplexity", category="timeout") from last_error
        try:
            async with httpx.AsyncClient(timeout=remaining, trust_env=False) as client:
                response = await client.post(url, headers=headers, json=payload)
            try:
                data = response.json()
            except (TypeError, ValueError):
                data = {}
            raw_text = response.text if hasattr(response, "text") else ""
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)

            if response.status_code >= 400:
                category = _classify_perplexity_http_error(response.status_code, data, raw_text)
                err_msg = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else ""
                if not err_msg:
                    err_msg = raw_text[:200]
                error = ProviderAdapterError(
                    f"Perplexity API вернул HTTP {response.status_code}: {err_msg}",
                    status=response.status_code,
                    category=category,
                )
                # Only true rate limits and 5xx server errors are retryable by policy.
                # Quota/credit exhaustion, auth errors, and configuration failures MUST NOT be retried.
                is_retryable = (category in {"rate_limit", "provider_5xx"}) and (response.status_code in {429, 500, 502, 503, 504})
                attempts_remain = (attempt + 1) < max_attempts
                delay = 0.0
                will_retry = False
                if is_retryable and attempts_remain:
                    retry_after_str = response.headers.get("retry-after")
                    if retry_after_str:
                        try:
                            delay = max(0.0, float(retry_after_str))
                        except (ValueError, TypeError):
                            delay = 0.5 * (2 ** attempt) + _get_perplexity_jitter(0.0, 0.25)
                    else:
                        delay = 0.5 * (2 ** attempt) + _get_perplexity_jitter(0.0, 0.25)

                    if time.monotonic() + delay <= deadline:
                        will_retry = True

                _populate_perplexity_diagnostics(
                    request_capture,
                    status_code=response.status_code,
                    headers=response.headers,
                    data=data if isinstance(data, dict) else {},
                    attempt_count=attempt + 1,
                    duration_ms=duration_ms,
                    classification=category,
                    is_retryable=is_retryable,
                    will_retry=will_retry,
                )

                if will_retry:
                    await asyncio.sleep(delay)
                    continue
                raise error

            if not isinstance(data, dict):
                _populate_perplexity_diagnostics(
                    request_capture,
                    status_code=response.status_code,
                    headers=response.headers,
                    attempt_count=attempt + 1,
                    duration_ms=duration_ms,
                    classification="invalid_response",
                    is_retryable=False,
                    will_retry=False,
                )
                raise ProviderAdapterError("Perplexity вернул некорректный JSON", status=response.status_code, category="invalid_response")

            _populate_perplexity_diagnostics(
                request_capture,
                status_code=response.status_code,
                headers=response.headers,
                data=data,
                attempt_count=attempt + 1,
                duration_ms=duration_ms,
                classification="success",
                is_retryable=False,
                will_retry=False,
            )
            return data
        except ProviderAdapterError:
            raise
        except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
            # SAFE pre-dispatch failure: request was not transmitted
            last_error = exc
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            category = "timeout" if isinstance(exc, httpx.ConnectTimeout) else "network_connection"
            is_retryable = True
            attempts_remain = (attempt + 1) < max_attempts
            delay = 0.0
            will_retry = False
            if attempts_remain:
                delay = 0.5 * (2 ** attempt) + _get_perplexity_jitter(0.0, 0.25)
                if time.monotonic() + delay <= deadline:
                    will_retry = True

            if request_capture is not None:
                _populate_perplexity_diagnostics(
                    request_capture,
                    attempt_count=attempt + 1,
                    duration_ms=duration_ms,
                    classification=category,
                    is_retryable=is_retryable,
                    will_retry=will_retry,
                )
            if will_retry:
                await asyncio.sleep(delay)
                continue
            raise ProviderAdapterError(f"Ошибка подключения к Perplexity: {exc}", category=category) from exc
        except (httpx.TimeoutException, TimeoutError) as exc:
            # UNCERTAIN post-dispatch timeout: do NOT automatically resend paid generation!
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            if request_capture is not None:
                _populate_perplexity_diagnostics(
                    request_capture,
                    attempt_count=attempt + 1,
                    duration_ms=duration_ms,
                    classification="timeout",
                    is_retryable=False,
                    will_retry=False,
                )
            raise ProviderAdapterError("Таймаут ожидания ответа Perplexity", category="timeout") from exc
        except httpx.NetworkError as exc:
            # UNCERTAIN post-dispatch network error: do NOT automatically resend paid generation!
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            if request_capture is not None:
                _populate_perplexity_diagnostics(
                    request_capture,
                    attempt_count=attempt + 1,
                    duration_ms=duration_ms,
                    classification="network_connection",
                    is_retryable=False,
                    will_retry=False,
                )
            raise ProviderAdapterError("Ошибка сети Perplexity", category="network_connection") from exc
        except Exception as exc:
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            if request_capture is not None:
                _populate_perplexity_diagnostics(
                    request_capture,
                    attempt_count=attempt + 1,
                    duration_ms=duration_ms,
                    classification="provider",
                    is_retryable=False,
                    will_retry=False,
                )
            raise ProviderAdapterError(f"Ошибка обращения к Perplexity: {exc}", category="provider") from exc
    raise ProviderAdapterError("Ошибка обращения к Perplexity", category="provider") from last_error


async def call_perplexity(
    api_key: str,
    layout: AIRequestLayout,
    preset_or_model: str | None = None,
    *,
    preset: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    timeout: float = 90.0,
    request_capture: dict | None = None,
    activity_tracker: Any | None = None,
    max_attempts: int = 2,
) -> str:
    if not api_key:
        raise ProviderAdapterError("API ключ Perplexity не задан", category="auth")
    payload = build_perplexity_payload(
        layout,
        preset_or_model=preset_or_model,
        preset=preset,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    data = await _post_perplexity_json(
        "https://api.perplexity.ai/v1/agent",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
        request_capture=request_capture,
        activity_tracker=activity_tracker,
        max_attempts=max_attempts,
    )
    return format_perplexity_response(data, request_capture=request_capture)


async def call_deepgram(
    api_key: str,
    file_bytes: bytes,
    filename: str,
    *,
    model: str = DEEPGRAM_DEFAULT_MODEL,
    timeout: float = 60.0,
    request_capture: dict | None = None,
    activity_tracker: Any | None = None,
) -> str:
    if not api_key:
        raise ProviderAdapterError("API ключ Deepgram не задан", category="auth")
    if not file_bytes:
        raise ProviderAdapterError("Пустой аудиофайл", category="invalid_request")
    if model not in DEEPGRAM_TRANSCRIPTION_MODELS:
        raise ProviderAdapterError("Недопустимая модель Deepgram", category="invalid_model")
    mime_type = mimetypes.guess_type(filename)[0] or "audio/ogg"
    params = {"model": model, "smart_format": "true"}
    if model == "nova-2":
        params["detect_language"] = "true"
    else:
        params["language"] = "multi"
    endpoint = "https://api.deepgram.com/v1/listen?" + urlencode(params)
    if request_capture is not None:
        payload = {"model": model, "smart_format": True}
        if model == "nova-2":
            payload["detect_language"] = True
        else:
            payload["language"] = "multi"
        _capture_ai_request(
            request_capture,
            provider="Deepgram",
            endpoint=endpoint,
            payload=payload,
        )
    last_error: Exception | None = None
    await _mark_activity(activity_tracker)
    deadline = time.monotonic() + max(float(timeout), 0.1)
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderAdapterError("Таймаут обращения к Deepgram", category="timeout") from last_error
        try:
            async with httpx.AsyncClient(timeout=remaining, trust_env=False) as client:
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Token {api_key}", "Content-Type": mime_type},
                    content=file_bytes,
                )
            if response.status_code >= 400:
                error = ProviderAdapterError(
                    f"Deepgram API вернул HTTP {response.status_code}",
                    status=response.status_code,
                    category=_status_category(response.status_code),
                )
                if _retryable_status(response.status_code) and attempt == 0:
                    await asyncio.sleep(0)
                    continue
                raise error
            data = response.json()
            transcript = (
                data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            if not isinstance(transcript, str) or not transcript.strip():
                raise ProviderAdapterError("Deepgram вернул пустую транскрипцию", status=response.status_code, category="empty_response")
            if request_capture is not None:
                request_capture["http_status"] = response.status_code
            return transcript.strip()
        except ProviderAdapterError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(0)
                continue
            raise ProviderAdapterError("Ошибка сети Deepgram", category="network") from exc
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise ProviderAdapterError("Deepgram вернул некорректный ответ", category="malformed_response") from exc
    raise ProviderAdapterError("Ошибка обращения к Deepgram", category="provider") from last_error


async def verify_openrouter_catalog(timeout: float = 20.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.get("https://openrouter.ai/api/v1/models")
    if response.status_code != 200:
        raise ProviderAdapterError("OpenRouter catalog unavailable", status=response.status_code, category="provider")
    payload = response.json()
    live = {str(item.get("id")): item for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")}
    result: dict[str, Any] = {}
    for model_id, spec in OPENROUTER_MODEL_SPECS.items():
        item = live.get(model_id)
        if item is None:
            result[model_id] = {"exists": False}
            continue
        architecture = item.get("architecture") or {}
        modalities = set(architecture.get("input_modalities") or [])
        live_context = item.get("context_length")
        live_output = (item.get("top_provider") or {}).get("max_completion_tokens")
        live_status = str(item.get("status") or "active")
        expiration_date = item.get("expiration_date")
        mismatches = []
        if spec.text != ("text" in modalities):
            mismatches.append("text")
        if ("image" in modalities) != spec.vision:
            mismatches.append("vision")
        if ("audio" in modalities) != spec.audio_input:
            mismatches.append("audio_input")
        if live_context != spec.context_limit:
            mismatches.append("context_limit")
        if live_output != spec.output_limit:
            mismatches.append("output_limit")
        if live_status.lower() in {"deprecated", "decommissioned", "disabled"}:
            mismatches.append("status")
        if expiration_date:
            try:
                if date.fromisoformat(str(expiration_date)) <= date.today():
                    mismatches.append("expiration_date")
            except ValueError:
                mismatches.append("expiration_date")
        result[model_id] = {
            "exists": True,
            "canonical_slug": item.get("canonical_slug"),
            "text": "text" in modalities,
            "vision": "image" in modalities,
            "audio_input": "audio" in modalities,
            "context_limit": live_context,
            "output_limit": live_output,
            "status": live_status,
            "expiration_date": expiration_date,
            "matches_static": not mismatches,
            "mismatches": mismatches,
        }
    return result
