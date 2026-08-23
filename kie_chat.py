from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

from ai_request_context import (
    AIRequestLayout,
    build_anthropic_system,
    build_gemini_contents,
    build_gemini_system_parts,
    build_openai_chat_messages,
    build_responses_input,
    normalize_request_messages,
)
from provider_models import (
    KIE_CHAT_PROTOCOL_ANTHROPIC,
    KIE_CHAT_PROTOCOL_GEMINI,
    KIE_CHAT_PROTOCOL_OPENAI,
    KIE_CHAT_PROTOCOL_RESPONSES,
    get_kie_chat_model_spec,
)


@dataclass(frozen=True)
class BuiltKIEChatRequest:
    endpoint: str
    headers: dict[str, str]
    payload: dict[str, Any]
    protocol: str
    stream: bool


def is_kie_error_payload(payload: Any) -> bool:
    """Return whether a JSON response is a KIE error envelope."""
    return (
        isinstance(payload, dict)
        and payload.get("code") not in (None, 200, "200")
    )


def is_kie_insufficient_balance(status_code: int, payload: Any) -> bool:
    """Recognize KIE credit failures without deciding how callers handle them."""
    if status_code == 402:
        return True
    if not isinstance(payload, dict):
        return False

    if str(payload.get("code", "")).strip() == "402":
        return True

    detail = payload.get("msg") or payload.get("message") or str(payload)
    lowered = str(detail).lower()
    return any(
        marker in lowered
        for marker in ("insufficient", "billing", "quota", "balance", "credit")
    )


def _value(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _text_messages(history: list[Any]) -> list[dict[str, str]]:
    messages = []
    for message in history:
        content = _value(message, "content")
        if not content:
            continue
        role = _value(message, "role", "user")
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": str(content)})
    return messages


def build_kie_chat_request(
    api_key: str,
    base_url: str,
    model: str,
    history: list[Any] | AIRequestLayout | None = None,
    system_prompt: str | AIRequestLayout | None = None,
    temperature: float = 0.7,
    *,
    request_layout: AIRequestLayout | None = None,
) -> BuiltKIEChatRequest:
    """Build a KIE request from the canonical semantic layout.

    ``history``/``system_prompt`` remain accepted for compatibility with
    older callers, but new code should pass ``request_layout``.  In
    particular, the adapter never needs a pre-concatenated full system
    prompt.
    """
    if request_layout is None:
        if isinstance(history, AIRequestLayout):
            request_layout = history
        elif isinstance(system_prompt, AIRequestLayout):
            request_layout = system_prompt
        else:
            request_layout = AIRequestLayout(
                stable_system_prompt=system_prompt or "",
                history=normalize_request_messages(history),
            )
    if not isinstance(request_layout, AIRequestLayout):  # pragma: no cover - defensive API guard
        raise TypeError("request_layout must be an AIRequestLayout")

    spec = get_kie_chat_model_spec(model)
    endpoint = f"{base_url.rstrip('/')}{spec.endpoint_path.format(model=model)}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if spec.protocol == KIE_CHAT_PROTOCOL_OPENAI:
        payload = {
            "model": model,
            "messages": build_openai_chat_messages(request_layout),
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": False,
        }
    elif spec.protocol == KIE_CHAT_PROTOCOL_ANTHROPIC:
        headers.update({
            "X-Api-Key": api_key,
            "anthropic-version": "2023-06-01",
        })
        anthropic_messages = [
            {"role": message.role, "content": message.content}
            for message in request_layout.history
        ]
        if request_layout.current_user_content is not None:
            anthropic_messages.append({
                "role": "user",
                "content": request_layout.current_user_content,
            })
        payload = {
            "model": model,
            "system": build_anthropic_system(request_layout),
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": False,
        }
    elif spec.protocol == KIE_CHAT_PROTOCOL_RESPONSES:
        payload = {
            "model": model,
            "instructions": request_layout.provider_stable_prefix,
            "input": build_responses_input(request_layout),
            "stream": spec.stream,
        }
    elif spec.protocol == KIE_CHAT_PROTOCOL_GEMINI:
        headers["X-Goog-Api-Key"] = api_key
        payload = {
            "stream": True,
            "contents": build_gemini_contents(request_layout),
            "systemInstruction": {
                "parts": build_gemini_system_parts(request_layout) or [{"text": ""}],
            },
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 4096,
            },
        }
    else:  # pragma: no cover - guarded by the authoritative catalog
        raise ValueError(f"Unsupported KIE chat protocol: {spec.protocol}")

    return BuiltKIEChatRequest(
        endpoint=endpoint,
        headers=headers,
        payload=payload,
        protocol=spec.protocol,
        stream=spec.stream,
    )


def _unwrap_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        nested = payload["data"]
        if not any(key in payload for key in ("choices", "content", "output", "candidates")):
            return nested
    return payload


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and not item.get("thought"):
            parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    return _content_text(message.get("content")) or _content_text(choice.get("text"))


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    return _content_text(payload.get("content"))


def _extract_responses_text(payload: dict[str, Any]) -> str:
    top_level = payload.get("output_text")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()

    parts = []
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "output_text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif item.get("type") == "message":
            text = _content_text(item.get("content"))
            if text:
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    parts = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        text = _content_text(content.get("parts")) if isinstance(content, dict) else ""
        if text:
            parts.append(text)
    return "".join(parts).strip()


def extract_kie_chat_text(payload: Any, protocol: str) -> str:
    payload = _unwrap_payload(payload)
    if not isinstance(payload, dict):
        return ""
    if protocol == KIE_CHAT_PROTOCOL_OPENAI:
        return _extract_openai_text(payload)
    if protocol == KIE_CHAT_PROTOCOL_ANTHROPIC:
        return _extract_anthropic_text(payload)
    if protocol == KIE_CHAT_PROTOCOL_RESPONSES:
        return _extract_responses_text(payload)
    if protocol == KIE_CHAT_PROTOCOL_GEMINI:
        return _extract_gemini_text(payload)
    return ""


def _iter_sse_payloads(text: str) -> Iterator[Any]:
    data_lines: list[str] = []

    def flush() -> Iterator[Any]:
        if not data_lines:
            return
        raw = "\n".join(data_lines).strip()
        data_lines.clear()
        if not raw or raw == "[DONE]":
            return
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            return

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
        elif not stripped and data_lines:
            yield from flush()
        elif not data_lines and stripped.startswith(("{", "[")):
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue
    if data_lines:
        yield from flush()


def _payload_items(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        yield payload


def _extract_stream_chunk_text(payload: dict[str, Any], protocol: str) -> str:
    if protocol == KIE_CHAT_PROTOCOL_GEMINI:
        parts = []
        for candidate in payload.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in content.get("parts", []) if isinstance(content, dict) else []:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought"):
                    parts.append(part["text"])
        return "".join(parts)
    if protocol == KIE_CHAT_PROTOCOL_ANTHROPIC:
        delta = payload.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            return delta["text"]
    return extract_kie_chat_text(payload, protocol)


def extract_kie_chat_stream_text(text: str, protocol: str) -> str:
    try:
        direct_payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        direct_payload = None
    if direct_payload is not None:
        direct_text = extract_kie_chat_text(direct_payload, protocol)
        if direct_text:
            return direct_text

    chunks = list(_iter_sse_payloads(text or ""))
    if protocol == KIE_CHAT_PROTOCOL_RESPONSES:
        deltas = []
        completed_text = []
        for payload in chunks:
            for item in _payload_items(payload):
                event_type = item.get("type")
                if event_type == "response.output_text.delta" and isinstance(item.get("delta"), str):
                    deltas.append(item["delta"])
                    continue
                if event_type == "response.output_text.done" and isinstance(item.get("text"), str):
                    completed_text.append(item["text"])
                    continue
                response = item.get("response")
                candidate = response if isinstance(response, dict) else item
                text_value = extract_kie_chat_text(candidate, protocol)
                if text_value:
                    completed_text.append(text_value)
        return "".join(deltas).strip() or "\n".join(completed_text).strip()

    parts = []
    for payload in chunks:
        for item in _payload_items(payload):
            item_text = _extract_stream_chunk_text(item, protocol)
            if item_text:
                parts.append(item_text)
    return "".join(parts).strip()


def extract_kie_chat_response_text(response: Any, protocol: str, stream: bool) -> str:
    if stream:
        text = getattr(response, "text", "") or ""
        parsed = extract_kie_chat_stream_text(text, protocol)
        if parsed:
            return parsed
    try:
        return extract_kie_chat_text(response.json(), protocol)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return ""
