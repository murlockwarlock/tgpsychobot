from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode

import httpx

from ai_request_context import AIRequestLayout, _capture_ai_request, build_openai_chat_messages
from provider_models import (
    DEEPGRAM_DEFAULT_MODEL,
    OPENROUTER_MODEL_SPECS,
    PERPLEXITY_MODES,
)


log = logging.getLogger(__name__)


class ProviderAdapterError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, category: str = "provider"):
        super().__init__(message)
        self.http_status = status
        self.classification = category


_ADAPTER_CLASSIFICATION_MAP = {
    "auth": "auth",
    "quota": "insufficient_balance_quota",
    "rate_limit": "rate_limit",
    "timeout": "timeout",
    "server_error": "provider_5xx",
    "network": "network_connection",
    "invalid_request": "provider_rejection",
    "invalid_model": "configuration",
    "malformed_response": "invalid_response",
    "empty_response": "empty_response",
    "provider": "unknown",
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
    preset: str,
    *,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if preset not in PERPLEXITY_MODES:
        raise ProviderAdapterError("Недопустимый режим Perplexity", category="invalid_model")
    instruction_blocks = layout.ordered_instruction_blocks
    instructions = "\n\n".join(block for block in instruction_blocks if block)
    if instructions:
        instructions += "\n\nИспользуй web_search для актуальных фактов и добавляй inline citations [n] к утверждениям, основанным на найденных источниках."
    input_parts: list[str] = []
    for message in layout.history:
        content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
        input_parts.append(f"{message.role}: {content}")
    if layout.current_user_content is not None:
        content = layout.current_user_content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        input_parts.append(f"user: {content}")
    tools: list[dict[str, Any]] = [{"type": "web_search"}]
    if preset in {"low", "medium"}:
        tools.append({"type": "fetch_url", "max_urls": 1})
    payload: dict[str, Any] = {
        "preset": preset,
        "input": "\n\n".join(part for part in input_parts if part),
        "tools": tools,
    }
    if instructions:
        payload["instructions"] = instructions
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    return payload


def _extract_perplexity_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
            elif isinstance(content, str):
                chunks.append(content)
        if chunks:
            return "\n".join(chunks).strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        text = _extract_message_text(content)
        if text:
            return text
    raise ProviderAdapterError("Perplexity вернул пустой ответ", category="empty_response")


def _extract_perplexity_citations(payload: dict[str, Any]) -> list[ProviderCitation]:
    found: list[ProviderCitation] = []
    seen: set[str] = set()
    output = payload.get("output")
    if not isinstance(output, list):
        output = []
    for item in output:
        if not isinstance(item, dict):
            continue
        results = item.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            if not url.startswith(("https://", "http://")) or url in seen:
                continue
            seen.add(url)
            title = str(result.get("title") or url).strip()
            found.append(ProviderCitation(title=title, url=url))
    citations = payload.get("citations")
    if isinstance(citations, list):
        for citation in citations:
            if isinstance(citation, str) and citation.startswith(("https://", "http://")) and citation not in seen:
                seen.add(citation)
                found.append(ProviderCitation(title=citation, url=citation))
    return found


def format_perplexity_response(payload: dict[str, Any]) -> str:
    text = _extract_perplexity_text(payload)
    citations = _extract_perplexity_citations(payload)
    if not citations:
        return text
    source_lines = [
        f"[{index}] {citation.title} — {citation.url}"
        for index, citation in enumerate(citations, start=1)
    ]
    return f"{text}\n\nИсточники:\n" + "\n".join(source_lines)


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


async def call_perplexity(
    api_key: str,
    layout: AIRequestLayout,
    preset: str,
    *,
    max_output_tokens: int | None = None,
    timeout: float = 90.0,
    request_capture: dict | None = None,
    activity_tracker: Any | None = None,
) -> str:
    if not api_key:
        raise ProviderAdapterError("API ключ Perplexity не задан", category="auth")
    payload = build_perplexity_payload(layout, preset, max_output_tokens=max_output_tokens)
    data = await _post_json(
        "https://api.perplexity.ai/v1/agent",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
        provider="Perplexity",
        request_capture=request_capture,
        activity_tracker=activity_tracker,
    )
    if request_capture is not None and isinstance(data.get("usage"), dict):
        request_capture["usage"] = data["usage"]
    return format_perplexity_response(data)


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
    if model != DEEPGRAM_DEFAULT_MODEL:
        raise ProviderAdapterError("Недопустимая модель Deepgram", category="invalid_model")
    mime_type = mimetypes.guess_type(filename)[0] or "audio/ogg"
    endpoint = "https://api.deepgram.com/v1/listen?" + urlencode(
        {"model": model, "smart_format": "true", "language": "multi"}
    )
    if request_capture is not None:
        _capture_ai_request(
            request_capture,
            provider="Deepgram",
            endpoint=endpoint,
            payload={"model": model, "language": "multi", "smart_format": True},
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
