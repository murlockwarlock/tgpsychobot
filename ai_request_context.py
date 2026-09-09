"""Canonical semantic layout for conversational AI requests.

The provider adapters in this project receive an :class:`AIRequestLayout` and
only translate it into the provider's wire format.  Keeping the semantic
blocks here prevents a provider-specific refactor from moving request data
into the stable system prompt.
"""

from __future__ import annotations

import copy
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterable

_SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "proxy_auth",
    "secret",
    "secret_key",
    "token",
    "x-api-key",
    "x_api_key",
}

_SENSITIVE_PARAM_PATTERN = re.compile(
    r"(?i)(?:key|api_key|token|access_token|secret|password)=([^&]+)"
)


def sanitize_endpoint_url(url: str | None) -> str:
    """Strip sensitive query parameters (e.g. ?key=...) from outbound endpoint URL."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.query:
            return url
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        sanitized_pairs = []
        for k, v in query_pairs:
            lower_k = k.lower()
            if lower_k in {"key", "api_key", "apikey", "token", "access_token", "secret", "password", "auth"}:
                continue
            sanitized_pairs.append((k, v))
        new_query = urllib.parse.urlencode(sanitized_pairs)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))
    except Exception:
        return _SENSITIVE_PARAM_PATTERN.sub("", url)


def _sanitize_payload_structure(obj: Any) -> Any:
    """Recursively sanitize credential keys in dict/list payload structures without altering message text."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            key_lower = str(k).lower().replace("-", "_")
            if key_lower in _SENSITIVE_KEY_NAMES:
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = _sanitize_payload_structure(v)
        return cleaned
    if isinstance(obj, (list, tuple)):
        return [_sanitize_payload_structure(item) for item in obj]
    return obj


def sanitize_ai_request_capture(
    *,
    provider: str,
    endpoint: str,
    payload: dict | list | Any,
) -> dict[str, Any]:
    return {
        "provider": str(provider),
        "endpoint": sanitize_endpoint_url(endpoint),
        "payload": _sanitize_payload_structure(copy.deepcopy(payload)),
    }


def _capture_ai_request(
    capture: dict | None,
    *,
    provider: str,
    endpoint: str,
    payload: dict | list | Any,
) -> None:
    """Record the exact provider request without credentials for the admin AI log."""
    if capture is None:
        return
    capture.clear()
    capture.update(
        sanitize_ai_request_capture(
            provider=provider,
            endpoint=endpoint,
            payload=payload,
        )
    )



@dataclass(frozen=True)
class AIRequestMessage:
    """A normalized dialogue message used by every conversational adapter."""

    role: str
    content: Any

    def __getitem__(self, key: str) -> Any:
        if key == "role":
            return self.role
        if key == "content":
            return self.content
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "role":
            return self.role
        if key == "content":
            return self.content
        return default


def _non_empty_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _text_blocks(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    else:
        values = value
    return tuple(
        text
        for item in values
        if (text := _non_empty_text(item))
    )


def _message_value(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def neutralize_stable_prompt(prompt: str | None) -> str:
    """Keep user-specific placeholders out of stable cacheable content."""
    stable_prompt = prompt or ""
    for placeholder, replacement in {
        "{user_name}": "[имя передано в служебном контексте]",
        "{user_gender}": "[пол передан в служебном контексте]",
        "{test_results}": "[результаты теста переданы в служебном контексте]",
        "{secret_answers}": "[ответы секретного теста переданы в служебном контексте]",
    }.items():
        stable_prompt = stable_prompt.replace(placeholder, replacement)
    return stable_prompt.strip()


def normalize_request_messages(messages: Iterable[Any] | None) -> tuple[AIRequestMessage, ...]:
    """Normalize history without changing message order or content shape."""
    normalized: list[AIRequestMessage] = []
    for message in messages or ():
        role = _message_value(message, "role", "user")
        content = _message_value(message, "content")
        if role not in {"user", "assistant"} or content in (None, "", []):
            continue
        normalized.append(AIRequestMessage(role=role, content=content))
    return tuple(normalized)


@dataclass(frozen=True)
class AIRequestLayout:
    """Canonical logical request order shared by text and vision adapters.

    The fields deliberately keep stable, dynamic, scenario, and transient
    context separate.  Adapters may serialize several dynamic blocks together
    only when their native protocol requires it; they must never fold those
    blocks into ``stable_system_prompt``.
    """

    stable_system_prompt: str = ""
    shared_instructions: tuple[str, ...] = ()
    runtime_context: tuple[str, ...] = ()
    scenario_context: tuple[str, ...] = ()
    request_context: tuple[str, ...] = ()
    history: tuple[AIRequestMessage, ...] = ()
    current_user_content: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stable_system_prompt", neutralize_stable_prompt(self.stable_system_prompt))
        object.__setattr__(
            self,
            "shared_instructions",
            tuple(neutralize_stable_prompt(block) for block in _text_blocks(self.shared_instructions)),
        )
        object.__setattr__(self, "runtime_context", _text_blocks(self.runtime_context))
        object.__setattr__(self, "scenario_context", _text_blocks(self.scenario_context))
        object.__setattr__(self, "request_context", _text_blocks(self.request_context))
        object.__setattr__(self, "history", normalize_request_messages(self.history))

    @property
    def dialogue_history(self) -> tuple[AIRequestMessage, ...]:
        """Readable alias for callers that use the semantic name."""
        return self.history

    @property
    def ordered_instruction_blocks(self) -> tuple[str, ...]:
        """Return blocks in the one required logical order."""
        return (
            ((self.stable_system_prompt,) if self.stable_system_prompt else ())
            + self.shared_instructions
            + self.runtime_context
            + self.scenario_context
            + self.request_context
        )

    @property
    def dynamic_instruction_blocks(self) -> tuple[str, ...]:
        """Return every block after the stable prefix, in canonical order."""
        return (
            self.shared_instructions
            + self.runtime_context
            + self.scenario_context
            + self.request_context
        )

    @property
    def provider_instruction_blocks(self) -> tuple[str, ...]:
        """Return the provider-facing blocks without changing logical order.

        Configured and shared instructions are one stable provider prefix,
        matching the production Chat Completions shape.  The runtime and
        scenario fields are one dynamic semantic region, followed by
        request-specific blocks.
        """
        blocks: list[str] = []
        stable_parts = (
            ((self.stable_system_prompt,) if self.stable_system_prompt else ())
            + self.shared_instructions
        )
        if stable_parts:
            blocks.append("\n\n".join(stable_parts))
        dynamic_parts = self.runtime_context + self.scenario_context
        if dynamic_parts:
            blocks.append("\n\n".join(dynamic_parts))
        blocks.extend(self.request_context)
        return tuple(blocks)

    @property
    def provider_stable_prefix(self) -> str:
        """Return the single stable block used by provider protocols with one prefix."""
        blocks = self.provider_instruction_blocks
        return blocks[0] if blocks else ""

    @property
    def provider_dynamic_instruction_blocks(self) -> tuple[str, ...]:
        """Return provider-facing dynamic blocks after the stable prefix."""
        blocks = self.provider_instruction_blocks
        if self.provider_stable_prefix and blocks:
            return blocks[1:]
        return blocks

    def with_history(self, history: Iterable[Any]) -> "AIRequestLayout":
        return AIRequestLayout(
            stable_system_prompt=self.stable_system_prompt,
            shared_instructions=self.shared_instructions,
            runtime_context=self.runtime_context,
            scenario_context=self.scenario_context,
            request_context=self.request_context,
            history=normalize_request_messages(history),
            current_user_content=self.current_user_content,
        )

    def with_current_user_content(self, content: Any) -> "AIRequestLayout":
        return AIRequestLayout(
            stable_system_prompt=self.stable_system_prompt,
            shared_instructions=self.shared_instructions,
            runtime_context=self.runtime_context,
            scenario_context=self.scenario_context,
            request_context=self.request_context,
            history=self.history,
            current_user_content=content,
        )

def build_openai_chat_messages(layout: AIRequestLayout) -> list[dict[str, Any]]:
    """Serialize the canonical layout for Chat Completions-compatible APIs."""
    messages = [
        {"role": "system", "content": block}
        for block in layout.provider_instruction_blocks
    ]
    messages.extend(
        {"role": message.role, "content": message.content}
        for message in layout.history
    )
    if layout.current_user_content is not None:
        messages.append({"role": "user", "content": layout.current_user_content})
    return messages


def build_anthropic_system(layout: AIRequestLayout) -> str | list[dict[str, str]]:
    """Serialize system blocks using Anthropic's structured system content."""
    blocks = layout.provider_instruction_blocks
    if not blocks:
        return ""
    return [{"type": "text", "text": block} for block in blocks]


def _gemini_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and "text" in item:
            parts.append({"text": item["text"]})
        elif "text" in item and set(item).issubset({"text"}):
            parts.append({"text": item["text"]})
        elif "inline_data" in item:
            parts.append({"inline_data": item["inline_data"]})
        else:
            parts.append(item)
    return parts


def build_gemini_system_parts(layout: AIRequestLayout) -> list[dict[str, str]]:
    """Serialize each logical instruction block as a distinct Gemini part."""
    return [{"text": block} for block in layout.provider_instruction_blocks]


def build_gemini_contents(layout: AIRequestLayout) -> list[dict[str, Any]]:
    contents = [
        {
            "role": "model" if message.role == "assistant" else "user",
            "parts": _gemini_parts(message.content),
        }
        for message in layout.history
    ]
    if layout.current_user_content is not None:
        contents.append({"role": "user", "parts": _gemini_parts(layout.current_user_content)})
    return contents


def _responses_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]
    return [
        item if isinstance(item, dict) and item.get("type") else {"type": "input_text", "text": str(item)}
        for item in content
    ]


def build_responses_input(layout: AIRequestLayout) -> list[dict[str, Any]]:
    """Serialize dynamic blocks and dialogue for the Responses protocol.

    Responses has one top-level ``instructions`` string, so the stable block
    is placed there.  The remaining canonical blocks use supported developer
    input messages and therefore remain physically after that stable prefix.
    """
    input_items: list[dict[str, Any]] = []
    for block in layout.provider_dynamic_instruction_blocks:
        input_items.append({
            "role": "developer",
            "content": [{"type": "input_text", "text": block}],
        })
    input_items.extend(
        {
            "role": message.role,
            "content": _responses_content(message.content),
        }
        for message in layout.history
    )
    if layout.current_user_content is not None:
        input_items.append({
            "role": "user",
            "content": _responses_content(layout.current_user_content),
        })
    return input_items
