from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime
import uuid

import anthropic
import httpx
import google.generativeai as genai  # noqa: F401
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import gemini_image

from prompt_blocks import MAX_CAPABILITIES
from ai_log_context import apply_ai_log_context, record_ai_attempt_log
from ai_request_builder import (
    ActivityTracker,
    build_conversational_request_layout,
    build_isolated_request_layout,
    get_user_ai_activity_gaps,
)
from .legacy import AIConfig, KnowledgeBase, Message as DBMessage, Topic, User, async_session_maker
from .legacy import AILog
from .logging_utils import configure_logging, get_ai_logger
from automation_engine import apply_service_data_blocks, build_runtime_automation_context
from user_metadata import extract_service_data
from memory_mode import MEMORY_MODE_TOPIC, build_history_scope, get_memory_mode, normalize_memory_mode
from result_history import ai_history_role_filter, select_ai_history_messages
from error_reporting import (
    classify_ai_error,
    classify_external_error,
    exception_summary,
    extract_error_metadata,
    send_ai_fallback_used_alert,
    send_output_budget_exhausted_alert,
    send_terminal_ai_failure_alert,
)
from vector_store import search_relevant_chunks
from provider_models import (
    CLAUDE_CHAT_MAX_TOKENS,
    DEEPSEEK_CHAT_MAX_TOKENS,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    GEMINI_CHAT_MAX_TOKENS,
    KIE_VISION_INITIAL_MAX_TOKENS,
    OPENAI_CHAT_MAX_TOKENS,
    PROVIDER_CLAUDE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    ModelUnavailableError,
    ensure_model_available,
    get_default_model,
    get_provider_vision_max_tokens,
    get_selectable_models,
    inspect_deepseek_response,
    is_retired_model,
    normalize_deepseek_model,
    should_omit_claude_sampling,
    validate_model_selection,
)
from vision_reliability import (
    VisionDeadlineTracker,
    VisionExecutionContext,
    attach_error_metadata,
    build_vision_httpx_timeout,
    order_kie_vision_candidates,
    resolve_effective_vision_fallback,
    run_coro_with_timeout,
    sanitize_vision_request_payload,
    sanitize_vision_text,
    should_retry_kie_vision_model,
    should_retry_kie_vision_upload,
    should_use_vision_provider_fallback,
)
from ai_request_context import (
    AIRequestLayout,
    _capture_ai_request,
    build_anthropic_system,
    build_gemini_contents,
    build_gemini_system_parts,
    build_openai_chat_messages,
    extract_effective_provider_and_model,
    neutralize_stable_prompt,
    normalize_request_messages,
)
from kie_chat import (
    build_kie_chat_request,
    extract_kie_chat_response_text,
    extract_kie_chat_text,
    is_kie_error_payload,
    is_kie_insufficient_balance,
)

configure_logging()
log = get_ai_logger("service")


class AIServiceError(RuntimeError):
    pass


class AIResponseError(AIServiceError):
    """Provider returned an invalid or empty text response."""
    pass


class InsufficientBalanceError(AIServiceError):
    pass


def _validate_text_response(response_text: object, *, provider: str) -> str:
    if not isinstance(response_text, str) or not response_text.strip():
        err = AIResponseError(f"{provider} returned an empty or invalid text response")
        err.provider_response_payload = str(response_text) if response_text else None
        err.classification = "empty_response"
        raise err
    return response_text



def _resolve_temperature(config, default: float = 0.7) -> float:
    value = getattr(config, "temperature", None) if config is not None else None
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_log_model(ai_config: AIConfig, provider: str | None) -> str:
    provider_key = (provider or "").strip().lower()
    model_field = "claude_model" if provider_key in {"claude", "anthropic"} else f"{provider_key}_model"
    model = getattr(ai_config, model_field, None)
    if not model:
        try:
            model = get_default_model(provider_key, channel="chat")
        except Exception:
            model = None
    if provider_key == "deepseek" and model:
        model = normalize_deepseek_model(str(model))
    return str(model or "—")


def _extract_effective_provider_and_model(
    request_capture: dict | None,
    default_provider: str,
    default_model: str,
) -> tuple[str, str]:
    return extract_effective_provider_and_model(
        request_capture,
        default_provider=default_provider,
        default_model=default_model,
        channel="chat",
    )


_CURRENT_AI_CONTEXT = object()


def _build_max_history_scope(
    user: User,
    memory_mode: str,
    topic_id: int | None | object = _CURRENT_AI_CONTEXT,
    dialogue_id: int | None = None,
):
    active_topic_id = user.current_topic_id if topic_id is _CURRENT_AI_CONTEXT else topic_id
    active_dialogue_id = dialogue_id or user.current_dialogue_id
    if memory_mode == MEMORY_MODE_TOPIC and active_topic_id is None:
        return (
            (DBMessage.user_id == user.id)
            & (DBMessage.dialogue_id == (active_dialogue_id or 1))
            & (DBMessage.topic_id.is_(None))
        )
    return build_history_scope(
        DBMessage,
        user.id,
        active_dialogue_id,
        active_topic_id,
        memory_mode,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_configured_system_prompt(ai_config: AIConfig, topic_prompt_text: str | None) -> str:
    system_prompt_text = topic_prompt_text

    if not system_prompt_text:
        if getattr(ai_config, "prompt_mode", "text") == "file" and getattr(ai_config, "prompt_filename", None):
            try:
                file_path = PROJECT_ROOT / "system_prompts" / ai_config.prompt_filename
                with open(file_path, "r", encoding="utf-8") as f:
                    system_prompt_text = f.read()
            except Exception:
                system_prompt_text = ai_config.system_prompt
        else:
            system_prompt_text = ai_config.system_prompt

    return system_prompt_text or ""


def _build_user_system_prompt(user: User, ai_config: AIConfig, topic: Topic | None | object = _CURRENT_AI_CONTEXT) -> str:
    active_topic = user.current_topic if topic is _CURRENT_AI_CONTEXT else topic
    system_prompt = _load_configured_system_prompt(
        ai_config,
        active_topic.system_prompt if active_topic and active_topic.system_prompt else None,
    )
    if not system_prompt:
        system_prompt = "Ты полезный ИИ-помощник."
    return neutralize_stable_prompt(system_prompt)


def _build_client_runtime_context(user: User) -> str:
    user_name = getattr(user, "name", None) or getattr(user, "first_name", None) or "Не указано"
    user_gender = getattr(user, "gender", None) or "Не указан"
    lines = ["ДАННЫЕ КЛИЕНТА:", f"ИМЯ: {user_name}", f"ПОЛ: {user_gender}"]
    if getattr(user, "age", None):
        lines.append(f"ВОЗРАСТ: {user.age}")
    return "\n".join(lines)


def _legacy_layout(
    messages: list[dict] | None,
    system_prompt: str | None = "",
) -> AIRequestLayout:
    normalized = normalize_request_messages(messages)
    current_content = None
    if normalized and normalized[-1].role == "user":
        current_content = normalized[-1].content
        normalized = normalized[:-1]
    return AIRequestLayout(
        stable_system_prompt=neutralize_stable_prompt(system_prompt),
        history=normalized,
        current_user_content=current_content,
    )


async def _call_openai(
    api_key: str,
    model: str,
    messages: list[dict] | None,
    temperature: float,
    *,
    request_layout: AIRequestLayout | None = None,
    request_capture: dict | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    target_model = model or "gpt-5.6-terra"
    ensure_model_available(PROVIDER_OPENAI, target_model)
    base_url = os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    payload: dict = {
        "model": target_model,
        "messages": build_openai_chat_messages(request_layout or _legacy_layout(messages)),
        "max_completion_tokens": OPENAI_CHAT_MAX_TOKENS,
    }
    if not target_model.startswith("gpt-5.6"):
        payload["temperature"] = temperature
    _capture_ai_request(
        request_capture,
        provider="OpenAI",
        endpoint=f"{base_url.rstrip('/')}/chat/completions",
        payload=payload,
    )
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    response = await client.chat.completions.create(**payload)
    return response.choices[0].message.content or ""


async def _call_deepseek(
    api_key: str,
    model: str,
    messages: list[dict] | None,
    temperature: float,
    *,
    request_layout: AIRequestLayout | None = None,
    request_capture: dict | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    normalized_model = normalize_deepseek_model(model)
    ensure_model_available(PROVIDER_DEEPSEEK, normalized_model)
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    payload = {
        "model": normalized_model,
        "messages": build_openai_chat_messages(request_layout or _legacy_layout(messages)),
        "max_tokens": DEEPSEEK_CHAT_MAX_TOKENS,
        "temperature": temperature,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    _capture_ai_request(
        request_capture,
        provider="Deepseek",
        endpoint=f"{base_url.rstrip('/')}/chat/completions",
        payload=payload,
    )
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    response = await client.chat.completions.create(
        **payload,
    )
    raw_payload_str = None
    try:
        if hasattr(response, "model_dump_json"):
            raw_payload_str = response.model_dump_json()
        elif hasattr(response, "to_dict"):
            raw_payload_str = json.dumps(response.to_dict(), default=str)
        elif isinstance(response, dict):
            raw_payload_str = json.dumps(response, default=str)
        else:
            raw_payload_str = str(response)
    except Exception:
        raw_payload_str = str(response)

    visible_content, diagnostics = inspect_deepseek_response(
        response,
        model=normalized_model,
        platform="max",
    )
    if diagnostics.output_budget_exhausted:
        log.warning(
            "DeepSeek output budget exhausted: provider=%s model=%s platform=%s finish_reason=%s "
            "visible_content_present=%s visible_content_length=%s "
            "reasoning_content_present=%s reasoning_content_length=%s "
            "output_budget_exhausted=%s",
            diagnostics.provider,
            diagnostics.model,
            diagnostics.platform,
            diagnostics.finish_reason,
            diagnostics.visible_content_present,
            diagnostics.visible_content_length,
            diagnostics.reasoning_content_present,
            diagnostics.reasoning_content_length,
            diagnostics.output_budget_exhausted,
        )
    if visible_content is not None:
        return visible_content

    if diagnostics.output_budget_exhausted:
        err = AIResponseError(
            f"Deepseek returned empty content (output budget exhausted: finish_reason={diagnostics.finish_reason}, reasoning_len={diagnostics.reasoning_content_length})"
        )
        err.http_status = 200
        err.finish_reason = diagnostics.finish_reason
        err.diagnostics = diagnostics
        err.provider_response_payload = raw_payload_str
        err.classification = "output_budget_exhausted"
        raise err

    err = AIResponseError("Deepseek returned an empty or invalid text response")
    err.http_status = 200
    err.finish_reason = diagnostics.finish_reason
    err.diagnostics = diagnostics
    err.provider_response_payload = raw_payload_str
    err.classification = "empty_response"
    raise err



async def _call_claude(
    api_key: str,
    model: str,
    messages: list[dict] | None,
    system_prompt: str,
    temperature: float,
    *,
    request_layout: AIRequestLayout | None = None,
    request_capture: dict | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    target_model = model or "claude-sonnet-5"
    ensure_model_available(PROVIDER_CLAUDE, target_model)
    layout = request_layout or _legacy_layout(messages, system_prompt)
    anthropic_messages = [
        {"role": message.role, "content": message.content}
        for message in layout.history
    ]
    if layout.current_user_content is not None:
        anthropic_messages.append({"role": "user", "content": layout.current_user_content})
    client = anthropic.AsyncAnthropic(api_key=api_key)
    payload: dict = {
        "model": target_model,
        "max_tokens": CLAUDE_CHAT_MAX_TOKENS,
        "system": build_anthropic_system(layout),
        "messages": anthropic_messages,
    }
    if not should_omit_claude_sampling(target_model):
        payload["temperature"] = temperature
    _capture_ai_request(
        request_capture,
        provider="Claude",
        endpoint="https://api.anthropic.com/v1/messages",
        payload=payload,
    )
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    response = await client.messages.create(**payload)
    return response.content[0].text


def _build_gemini_proxy_transport():
    """Build an httpx AsyncHTTPTransport using the GEMINI_PROXY env variable, if set."""
    raw_proxy = os.getenv("GEMINI_PROXY")
    if not raw_proxy:
        return None
    proxy = raw_proxy.strip().strip('"').strip("'")
    if not proxy:
        return None
    return httpx.AsyncHTTPTransport(proxy=proxy)


async def _call_gemini(
    api_key: str,
    model: str,
    messages: list[dict] | None,
    system_prompt: str,
    temperature: float,
    *,
    request_layout: AIRequestLayout | None = None,
    request_capture: dict | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    import httpx

    target_model = model or "gemini-3.7-flash"
    ensure_model_available(PROVIDER_GEMINI, target_model)
    layout = request_layout or _legacy_layout(messages, system_prompt)
    generation_config: dict = {"maxOutputTokens": GEMINI_CHAT_MAX_TOKENS}
    if not (target_model.startswith("gemini-3.7") or target_model.startswith("gemini-3.6")):
        generation_config["temperature"] = temperature

    payload = {
        "contents": build_gemini_contents(layout),
        "systemInstruction": {
            "parts": build_gemini_system_parts(layout) or [{"text": ""}],
        },
        "generationConfig": generation_config,
    }
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"
    url = f"{endpoint}?key={api_key}"
    _capture_ai_request(
        request_capture,
        provider="Gemini",
        endpoint=endpoint,
        payload=payload,
    )
    transport = _build_gemini_proxy_transport()
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    async with httpx.AsyncClient(timeout=60.0, transport=transport) as client:
        response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# KIE helpers
# ---------------------------------------------------------------------------

def _get_kie_base_url(config) -> str:
    return (getattr(config, "kie_base_url", None) or "https://api.kie.ai").rstrip("/")


def _get_kie_upload_base_url(config) -> str:
    return (getattr(config, "kie_upload_base_url", None) or "https://kieai.redpandaai.co").rstrip("/")


def _kie_model_base_url(base_url: str, model: str) -> str:
    return f"{base_url.rstrip('/')}/{model}/v1"


def _guess_filename(file_bytes: bytes, fallback_stem: str, fallback_ext: str) -> str:
    header = file_bytes[:16]
    ext = fallback_ext.lower().lstrip(".")
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        ext = "png"
    elif header.startswith(b"\xff\xd8\xff"):
        ext = "jpg"
    elif header.startswith(b"GIF8"):
        ext = "gif"
    elif header.startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
        ext = "webp"
    elif header.startswith(b"RIFF") and file_bytes[8:12] == b"WAVE":
        ext = "wav"
    elif header.startswith(b"OggS"):
        ext = "ogg"
    elif header.startswith(b"ID3") or header[:2] == b"\xff\xfb":
        ext = "mp3"
    elif header.startswith(b"%PDF"):
        ext = "pdf"
    return f"{fallback_stem}_{uuid.uuid4().hex[:12]}.{ext}"


def _extract_kie_chat_text(payload: dict) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            return "\n".join(p for p in parts if p).strip()
    return ""


def _validate_kie_json_response(status_code: int, payload: dict, *, context: str) -> dict:
    raw_detail = (
        payload.get("msg") or payload.get("message") or str(payload)
        if isinstance(payload, dict)
        else str(payload)
    )
    detail = str(raw_detail).strip() or f"HTTP {status_code} без описания"
    try:
        import json
        payload_str = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
    except Exception:
        payload_str = str(payload)

    if status_code != 200:
        code_val = payload.get("code") if isinstance(payload, dict) else None
        if is_kie_insufficient_balance(status_code, payload):
            err = InsufficientBalanceError(f"KIE API Error: {detail}")
        else:
            err = AIServiceError(f"{context}: status={status_code} message={detail}")
        err.http_status = status_code
        err.provider_code = code_val
        err.provider_response_payload = payload_str
        raise err

    code = payload.get("code")
    if code not in (None, 200, "200"):
        if is_kie_insufficient_balance(status_code, payload):
            err = InsufficientBalanceError(f"KIE API Error: {detail}")
        else:
            err = AIServiceError(f"{context}: {detail}")
        err.http_status = 200
        err.provider_code = code
        err.provider_response_payload = payload_str
        raise err

    return payload.get("data") if isinstance(payload.get("data"), dict) else payload


def _find_first_string_value(data, candidate_keys: tuple) -> str | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in candidate_keys and isinstance(value, str) and value.strip():
                return value.strip()
            found = _find_first_string_value(value, candidate_keys)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_first_string_value(item, candidate_keys)
            if found:
                return found
    return None


def _extract_kie_task_result(task_payload: dict) -> dict:
    response_payload = task_payload.get("response")
    if isinstance(response_payload, dict) and response_payload:
        return response_payload
    result_json = task_payload.get("resultJson")
    if isinstance(result_json, str) and result_json:
        try:
            return json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise AIServiceError(f"Cannot decode KIE resultJson: {exc}: {result_json}") from exc
    if isinstance(result_json, dict):
        return result_json
    return {}


async def _upload_file_to_kie(
    api_key: str,
    upload_base_url: str,
    file_bytes: bytes,
    filename: str,
    upload_path: str,
    *,
    timeout: float | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    url = f"{upload_base_url}/api/file-stream-upload"
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    form_data = {"uploadPath": upload_path, "fileName": filename}
    headers = {"Authorization": f"Bearer {api_key}"}
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before KIE upload: %s", act_err)
    client_timeout = build_vision_httpx_timeout(timeout) if timeout is not None else 120.0
    try:
        async with httpx.AsyncClient(timeout=client_timeout, trust_env=False) as client:
            response = await client.post(url, headers=headers, data=form_data, files=files)
        try:
            payload = response.json()
        except Exception:
            payload = {}

        if response.status_code >= 400:
            err = AIServiceError(f"Ошибка загрузки файла в KIE (HTTP {response.status_code}): {response.text}")
            err.http_status = response.status_code
            if isinstance(payload, dict) and "code" in payload:
                err.provider_code = payload.get("code")
            err.provider_response_payload = response.text
            from error_reporting import classify_external_error
            code, _ = classify_external_error(err)
            err.classification = code
            raise err

        data_payload = _validate_kie_json_response(response.status_code, payload, context="KIE upload failed")
        file_url = data_payload.get("downloadUrl") or data_payload.get("fileUrl")
        if not file_url:
            raise AIServiceError(f"KIE upload returned no file URL: {payload}")
        return file_url
    except (AIServiceError, InsufficientBalanceError):
        raise
    except Exception as e:
        log.error("KIE upload error: %s", e)
        raise AIServiceError(f"Ошибка загрузки файла в KIE: {exception_summary(e)}") from e


async def _call_kie_vision_inference(
    api_key: str,
    base_url: str,
    model: str,
    file_url: str,
    system_prompt: str,
    prompt: str,
    temperature: float = 0.7,
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    timeout: float = 25.0,
    request_capture: dict | None = None,
) -> str:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_KIE, target_model, channel="vision")
    layout = request_layout or AIRequestLayout(
        stable_system_prompt=neutralize_stable_prompt(system_prompt),
        history=normalize_request_messages(()),
    )
    layout = layout.with_current_user_content([
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": file_url}},
    ])
    payload = {
        "model": target_model,
        "messages": build_openai_chat_messages(layout),
        "max_tokens": KIE_VISION_INITIAL_MAX_TOKENS,
        "temperature": temperature,
        "stream": False,
    }
    endpoint = f"{_kie_model_base_url(base_url, target_model)}/chat/completions"
    _capture_ai_request(request_capture, provider="KIE", endpoint=endpoint, payload=payload)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    httpx_timeout = build_vision_httpx_timeout(timeout)
    async with httpx.AsyncClient(timeout=httpx_timeout, trust_env=False) as client:
        response = await client.post(
            endpoint,
            headers=headers,
            json=payload,
        )

    try:
        resp_json = response.json()
    except Exception:
        resp_json = {}

    if response.status_code >= 400:
        err = AIServiceError(f"KIE multimodal API error (HTTP {response.status_code}): {response.text}")
        err.http_status = response.status_code
        if isinstance(resp_json, dict) and "code" in resp_json:
            err.provider_code = resp_json.get("code")
        err.provider_response_payload = response.text
        raise err

    response_payload = _validate_kie_json_response(
        response.status_code, resp_json,
        context="Ошибка обращения к KIE multimodal API",
    )

    choices = response_payload.get("choices") if isinstance(response_payload, dict) else []
    first_choice = choices[0] if (choices and isinstance(choices, list)) else {}
    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    usage_data = response_payload.get("usage") if isinstance(response_payload, dict) else None
    provider_code_val = resp_json.get("code") if isinstance(resp_json, dict) else None

    safe_resp_payload = None
    try:
        import json
        safe_resp_payload = json.dumps(response_payload, ensure_ascii=False) if isinstance(response_payload, dict) else str(response_payload)
    except Exception:
        safe_resp_payload = None

    if request_capture is not None:
        request_capture["http_status"] = response.status_code
        request_capture["finish_reason"] = finish_reason
        if usage_data is not None:
            request_capture["usage"] = usage_data
        if provider_code_val is not None:
            request_capture["provider_code"] = provider_code_val
        if safe_resp_payload is not None:
            request_capture["provider_response_payload"] = safe_resp_payload

    diag_meta = {"usage": usage_data, "provider_code": provider_code_val} if (usage_data or provider_code_val) else None

    if not choices or not isinstance(choices, list):
        raise attach_error_metadata(
            AIResponseError("KIE multimodal request returned empty choices"),
            classification="empty_response",
            http_status=response.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if finish_reason in {"content_filter", "safety"}:
        raise attach_error_metadata(
            AIServiceError("KIE vision response rejected by safety filter"),
            classification="provider_rejection",
            finish_reason=finish_reason,
            http_status=response.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if finish_reason == "length":
        raise attach_error_metadata(
            AIResponseError("KIE vision response exceeded token budget"),
            classification="output_budget_exhausted",
            finish_reason="length",
            http_status=response.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    text = _extract_kie_chat_text(response_payload)
    if not text or not text.strip():
        raise attach_error_metadata(
            AIResponseError("KIE multimodal request returned empty content"),
            classification="empty_response",
            finish_reason=finish_reason,
            http_status=response.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )
    return text


async def _create_kie_task(api_key: str, base_url: str, model: str, input_payload: dict) -> str:
    url = f"{base_url}/api/v1/jobs/createTask"
    payload = {"model": model, "input": input_payload}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(url, headers=headers, json=payload)
        data = response.json()
        data_payload = _validate_kie_json_response(response.status_code, data, context="KIE task creation failed")
        task_id = data_payload.get("taskId")
        if not task_id:
            raise AIServiceError(f"KIE task creation returned no taskId: {data}")
        return task_id
    except (AIServiceError, InsufficientBalanceError):
        raise
    except Exception as e:
        logging.error("KIE create task error", exc_info=e)
        raise AIServiceError(f"Ошибка создания задачи KIE: {exception_summary(e)}") from e


async def _poll_kie_task(api_key: str, base_url: str, task_id: str, *, timeout_sec: int = 180) -> dict:
    url = f"{base_url}/api/v1/jobs/recordInfo"
    headers = {"Authorization": f"Bearer {api_key}"}
    delay = 2.0
    deadline = asyncio.get_running_loop().time() + timeout_sec
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        while True:
            response = await client.get(url, headers=headers, params={"taskId": task_id})
            payload = _validate_kie_json_response(
                response.status_code, response.json(),
                context=f"KIE task polling failed: task_id={task_id}",
            )
            state = (payload.get("state") or payload.get("status") or "").lower()
            success_flag = payload.get("successFlag")
            if state in {"success", "succeed", "succeeded"} or success_flag == 1:
                return payload
            if state in {"fail", "failed", "error"}:
                fail_msg = payload.get("failMsg") or payload.get("errorMessage") or "unknown task failure"
                raise AIServiceError(f"KIE task failed: task_id={task_id} message={fail_msg}")
            if asyncio.get_running_loop().time() >= deadline:
                raise AIServiceError(f"KIE task timed out: task_id={task_id} state={state}")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 8.0)


async def _get_kie_download_url(api_key: str, base_url: str, url: str) -> str:
    endpoint = f"{base_url}/api/v1/common/download-url"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(endpoint, headers=headers, json={"url": url})
        if response.status_code != 200:
            return url
        data = response.json()
        return data.get("data") or url
    except Exception:
        return url


async def _download_binary_file(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.get(url)
    if response.status_code != 200:
        raise AIServiceError(f"Result download failed: status={response.status_code} url={url}")
    return response.content


async def _call_kie_multimodal(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: list,
    temperature: float = 0.7,
    channel: str = "chat",
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_KIE, target_model, channel=channel)
    try:
        layout = request_layout or AIRequestLayout(
            stable_system_prompt=neutralize_stable_prompt(system_prompt),
            history=normalize_request_messages(()),
            current_user_content=user_content,
        )
        layout = layout.with_current_user_content(user_content)
        payload = {
            "model": target_model,
            "messages": build_openai_chat_messages(layout),
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if activity_tracker is not None:
            try:
                await activity_tracker.mark_outbound_attempt_once()
            except Exception as act_err:
                log.warning("Failed to mark activity before outbound call: %s", act_err)
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(
                f"{_kie_model_base_url(base_url, target_model)}/chat/completions",
                headers=headers,
                json=payload,
            )
        response_payload = _validate_kie_json_response(
            response.status_code, response.json(),
            context="Ошибка обращения к KIE multimodal API",
        )
        text = _extract_kie_chat_text(response_payload)
        if not text:
            raise AIServiceError("KIE multimodal request returned empty content")
        return text
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE multimodal error", exc_info=e)
        raise AIServiceError(f"Ошибка обращения к KIE multimodal API: {exception_summary(e)}") from e


def _select_image_generation_shape(prompt: str) -> tuple[str, str]:
    prompt_lc = (prompt or "").lower()
    portrait_markers = ("tarot", "card", "oracle", "poster", "cover", "vertical", "portrait orientation", "full body", "full-body", "phone wallpaper")
    landscape_markers = ("landscape orientation", "horizontal", "wide shot", "widescreen", "panoramic", "banner", "cinematic wide")
    if any(m in prompt_lc for m in portrait_markers):
        return "3:4", "1024x1536"
    if any(m in prompt_lc for m in landscape_markers):
        return "4:3", "1536x1024"
    return "1:1", "1024x1024"


def _build_kie_image_generation_input(model: str, prompt: str) -> dict:
    aspect_ratio, _ = _select_image_generation_shape(prompt)
    if model == "google/imagen4-fast":
        return {"prompt": prompt, "aspect_ratio": aspect_ratio, "num_images": "1"}
    if model in {"google/imagen4-ultra", "google/imagen4"}:
        return {"prompt": prompt, "aspect_ratio": aspect_ratio}
    if model == "bytedance/seedream-v4-text-to-image":
        return {"prompt": prompt, "image_size": "square_hd", "image_resolution": "1K", "max_images": 1}
    if model == "seedream/4.5-text-to-image":
        return {"prompt": prompt, "aspect_ratio": aspect_ratio, "quality": "basic"}
    raise AIServiceError(f"Неподдерживаемая KIE image generation model: {model}")


def _build_kie_image_edit_input(model: str, prompt: str, source_url: str) -> dict:
    aspect_ratio, _ = _select_image_generation_shape(prompt)
    if model == "google/nano-banana-edit":
        return {"prompt": prompt, "image_urls": [source_url], "output_format": "png", "image_size": "1:1"}
    if model == "bytedance/seedream-v4-edit":
        return {"prompt": prompt, "image_urls": [source_url], "image_size": "square_hd", "image_resolution": "1K", "max_images": 1}
    if model == "seedream/4.5-edit":
        return {"prompt": prompt, "image_urls": [source_url], "aspect_ratio": aspect_ratio, "quality": "basic"}
    raise AIServiceError(f"Неподдерживаемая KIE image edit model: {model}")


async def _transcribe_kie(api_key: str, base_url: str, upload_base_url: str, model: str, file_bytes: bytes, filename: str) -> str:
    ensure_model_available(PROVIDER_KIE, model, channel="transcription")
    try:
        file_url = await _upload_file_to_kie(api_key, upload_base_url, file_bytes, filename, "audio")
        if model == "elevenlabs/speech-to-text":
            task_id = await _create_kie_task(api_key, base_url, model, {
                "audio_url": file_url,
                "language_code": "ru",
                "tag_audio_events": False,
                "diarize": False,
            })
            task_payload = await _poll_kie_task(api_key, base_url, task_id, timeout_sec=60)
            result = _extract_kie_task_result(task_payload)
            transcription = _find_first_string_value(result, ("text", "transcript", "transcription", "content", "result"))
            if not transcription:
                raise AIServiceError(f"KIE STT returned no transcription text: task_id={task_id}")
            return transcription
        return await _call_kie_multimodal(
            api_key, base_url, model,
            "Ты — сервис точной транскрибации речи.",
            [
                {"type": "text", "text": "Сделай точную транскрипцию аудио. Язык речи: русский. Верни только текст без пояснений."},
                {"type": "image_url", "image_url": {"url": file_url}},
            ],
            temperature=0.0,
            channel="transcription",
        )
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE transcription error", exc_info=e)
        raise AIServiceError(f"Ошибка при транскрибации (KIE API): {exception_summary(e)}") from e


async def _analyze_kie(api_key: str, base_url: str, upload_base_url: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float = 0.7, history: list = None, shared_instructions: tuple[str, ...] = (), request_layout: AIRequestLayout | None = None, *, activity_tracker: ActivityTracker | None = None, timeout: float = 25.0) -> str:
    ensure_model_available(PROVIDER_KIE, model, channel="vision")
    try:
        file_url = await _upload_file_to_kie(
            api_key, upload_base_url, image_bytes,
            _guess_filename(image_bytes, "vision_input", "jpg"), "images",
            timeout=timeout,
            activity_tracker=activity_tracker,
        )
        layout = request_layout or AIRequestLayout(
            stable_system_prompt=system_prompt,
            shared_instructions=shared_instructions,
            history=normalize_request_messages(history),
        )
        return await _call_kie_vision_inference(
            api_key, base_url, model,
            file_url,
            system_prompt,
            prompt,
            temperature=temperature,
            request_layout=layout,
            activity_tracker=activity_tracker,
            timeout=timeout,
        )
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE vision error", exc_info=e)
        raise AIServiceError(f"Ошибка анализа изображения (KIE): {exception_summary(e)}") from e


async def _generate_kie(api_key: str, base_url: str, model: str, prompt: str) -> bytes:
    ensure_model_available(PROVIDER_KIE, model, channel="image_gen")
    attempts = 2
    last_exc: Exception = AIServiceError("KIE image generation failed without detailed error")
    for _ in range(attempts):
        try:
            task_id = await _create_kie_task(api_key, base_url, model, _build_kie_image_generation_input(model, prompt))
            task_payload = await _poll_kie_task(api_key, base_url, task_id)
            result = _extract_kie_task_result(task_payload)
            result_urls = result.get("resultUrls") or result.get("result_urls") or []
            if not result_urls:
                raise AIServiceError(f"KIE image generation returned no result URLs: task_id={task_id}")
            download_url = await _get_kie_download_url(api_key, base_url, result_urls[0])
            return await _download_binary_file(download_url)
        except AIServiceError as exc:
            last_exc = exc
            if "internal error" not in str(exc).lower():
                raise
            await asyncio.sleep(2)
    raise last_exc


async def _edit_kie(api_key: str, base_url: str, upload_base_url: str, model: str, prompt: str, image_bytes: bytes) -> bytes:
    ensure_model_available(PROVIDER_KIE, model, channel="image_edit")
    source_url = await _upload_file_to_kie(
        api_key, upload_base_url, image_bytes,
        _guess_filename(image_bytes, "image_edit_source", "jpg"), "images",
    )
    task_id = await _create_kie_task(api_key, base_url, model, _build_kie_image_edit_input(model, prompt, source_url))
    task_payload = await _poll_kie_task(api_key, base_url, task_id)
    result = _extract_kie_task_result(task_payload)
    result_urls = result.get("resultUrls") or result.get("result_urls") or []
    if not result_urls:
        raise AIServiceError(f"KIE image edit returned no result URLs: task_id={task_id}")
    download_url = await _get_kie_download_url(api_key, base_url, result_urls[0])
    return await _download_binary_file(download_url)


async def _call_kie_text_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict] | None,
    system_prompt: str,
    temperature: float,
    *,
    request_layout: AIRequestLayout | None = None,
    request_capture: dict | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    """Call KIE text chat using the model's documented protocol."""
    ensure_model_available(PROVIDER_KIE, model, channel="chat")
    layout = request_layout or _legacy_layout(messages, system_prompt)
    request = build_kie_chat_request(
        api_key,
        base_url,
        model,
        request_layout=layout,
        temperature=temperature,
    )
    _capture_ai_request(
        request_capture,
        provider="KIE",
        endpoint=request.endpoint,
        payload=request.payload,
    )
    try:
        if activity_tracker is not None:
            try:
                await activity_tracker.mark_outbound_attempt_once()
            except Exception as act_err:
                log.warning("Failed to mark activity before outbound call: %s", act_err)
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(
                request.endpoint,
                headers=request.headers,
                json=request.payload,
            )
        if response.status_code != 200:
            try:
                error_payload = response.json()
            except (TypeError, ValueError):
                error_payload = {"message": getattr(response, "text", "")}
            _validate_kie_json_response(
                response.status_code,
                error_payload,
                context="Ошибка при обращении к KIE Chat API",
            )

        if request.stream:
            try:
                response_payload = response.json()
            except (TypeError, ValueError):
                response_payload = None
            if is_kie_error_payload(response_payload):
                _validate_kie_json_response(
                    response.status_code,
                    response_payload,
                    context="Ошибка при обращении к KIE Chat API",
                )
            text = extract_kie_chat_response_text(response, request.protocol, stream=True)
        else:
            try:
                response_payload = _validate_kie_json_response(
                    response.status_code,
                    response.json(),
                    context="Ошибка при обращении к KIE Chat API",
                )
                text = extract_kie_chat_text(response_payload, request.protocol)
            except (TypeError, ValueError, json.JSONDecodeError):
                text = extract_kie_chat_response_text(response, request.protocol, stream=True)
        if not text:
            raise AIServiceError("KIE chat returned empty content")
        return text
    except (AIServiceError, InsufficientBalanceError):
        raise
    except Exception as e:
        log.error("KIE chat error: %s", e, exc_info=True)
        raise AIServiceError(f"Ошибка при обращении к KIE Chat API: {exception_summary(e)}") from e


def _resolve_provider(ai_config: AIConfig) -> tuple[str, float]:
    provider = (ai_config.provider or "").strip().lower()
    temperature = _resolve_temperature(ai_config)
    return provider, temperature


async def _dispatch_provider(
    ai_config: AIConfig,
    request_layout: AIRequestLayout | str,
    messages: list[dict] | None = None,
    *,
    request_capture: dict | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    provider, temperature = _resolve_provider(ai_config)
    layout = (
        request_layout
        if isinstance(request_layout, AIRequestLayout)
        else _legacy_layout(messages, request_layout)
    )

    timeout = float(getattr(ai_config, "fallback_timeout", 60) or 60.0)

    async def _invoke():
        if provider == "openai":
            if not ai_config.openai_api_key:
                raise AIServiceError("OpenAI API key не задан")
            return await _call_openai(
                ai_config.openai_api_key,
                ai_config.openai_model,
                [],
                temperature,
                request_layout=layout,
                request_capture=request_capture,
                activity_tracker=activity_tracker,
            )
        elif provider in {"claude", "anthropic"}:
            claude_key = getattr(ai_config, "claude_api_key", None) or getattr(ai_config, "anthropic_api_key", None)
            if not claude_key:
                raise AIServiceError("Claude API key не задан")
            return await _call_claude(
                claude_key,
                ai_config.claude_model,
                [],
                layout.stable_system_prompt,
                temperature,
                request_layout=layout,
                request_capture=request_capture,
                activity_tracker=activity_tracker,
            )
        elif provider == "gemini":
            if not ai_config.gemini_api_key:
                raise AIServiceError("Gemini API key не задан")
            return await _call_gemini(
                ai_config.gemini_api_key,
                ai_config.gemini_model,
                [],
                layout.stable_system_prompt,
                temperature,
                request_layout=layout,
                request_capture=request_capture,
                activity_tracker=activity_tracker,
            )
        elif provider == "deepseek":
            if not ai_config.deepseek_api_key:
                raise AIServiceError("DeepSeek API key не задан")
            return await _call_deepseek(
                ai_config.deepseek_api_key,
                ai_config.deepseek_model,
                [],
                temperature,
                request_layout=layout,
                request_capture=request_capture,
                activity_tracker=activity_tracker,
            )
        elif provider == "kie":
            if not ai_config.kie_api_key:
                raise AIServiceError("KIE API key не задан")
            base_url = _get_kie_base_url(ai_config)
            return await _call_kie_text_chat(
                ai_config.kie_api_key,
                base_url,
                ai_config.kie_model or "gemini-3-flash",
                [],
                layout.stable_system_prompt,
                temperature,
                request_layout=layout,
                request_capture=request_capture,
                activity_tracker=activity_tracker,
            )
        else:
            raise AIServiceError(f"Неподдерживаемый провайдер ИИ: {ai_config.provider}")

    try:
        result = await asyncio.wait_for(_invoke(), timeout=timeout)
    except asyncio.CancelledError:
        raise
    except (asyncio.TimeoutError, TimeoutError) as timeout_exc:
        err = AIServiceError(f"AI provider {provider} timed out after {timeout}s")
        err.classification = "timeout"
        raise err from timeout_exc
    return _validate_text_response(result, provider=provider)


def _looks_like_prompt_kb_entry(filename: str | None, indexed_content: str | None) -> bool:
    normalized_name = (filename or "").strip().lower()
    prompt_name_markers = (
        "prompt",
        "промпт",
        "system_prompt",
        "system-prompt",
        "system prompt",
    )
    if any(marker in normalized_name for marker in prompt_name_markers):
        return True

    normalized_head = (indexed_content or "")[:2000].strip().lower()
    if not normalized_head:
        return False

    if "system prompt" in normalized_head or "системный промпт" in normalized_head:
        return True
    return False


async def get_ai_response(
    user_id: int,
    user_prompt: str,
    *,
    topic_id_override: int | None | object = _CURRENT_AI_CONTEXT,
    dialogue_id_override: int | None = None,
    exclude_message_id: int | None = None,
    track_user_activity: bool = True,
    activity_tracker: ActivityTracker | None = None,
    minutes_since_last_visit: int | None = None,
    minutes_since_last_message: int | None = None,
    request_type: str = "chat",
) -> str:
    async with async_session_maker() as session:
        user = await session.scalar(
            select(User)
            .options(
                selectinload(User.current_topic).selectinload(Topic.knowledge_base_files),
                selectinload(User.subscription),
            )
            .where(User.id == user_id)
        )
        if not user:
            raise AIServiceError("Пользователь не найден")

        active_topic_id = user.current_topic_id if topic_id_override is _CURRENT_AI_CONTEXT else topic_id_override
        active_dialogue_id = dialogue_id_override or user.current_dialogue_id
        if active_topic_id == user.current_topic_id:
            active_topic = user.current_topic
        elif active_topic_id is not None:
            active_topic = await session.scalar(
                select(Topic)
                .options(selectinload(Topic.knowledge_base_files))
                .where(Topic.id == active_topic_id)
            )
        else:
            active_topic = None

        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            raise AIServiceError("AIConfig не найден")

        actual_provider = str(ai_config.provider or "—")
        actual_model = _resolve_log_model(ai_config, actual_provider)

        stable_system_prompt = _build_user_system_prompt(user, ai_config, active_topic)

        relevant_chunks = []
        if active_topic:
            doc_ids = [f.id for f in active_topic.knowledge_base_files]
            if doc_ids:
                relevant_chunks = await search_relevant_chunks(user_prompt, n_results=3, document_ids=doc_ids)
        else:
            gen_files_res = await session.execute(
                select(KnowledgeBase.id, KnowledgeBase.filename, KnowledgeBase.indexed_content).where(
                    KnowledgeBase.use_in_general_mode == True
                )
            )
            gen_doc_ids = [
                doc_id
                for doc_id, filename, indexed_content in gen_files_res.all()
                if not _looks_like_prompt_kb_entry(filename, indexed_content)
            ]
            if gen_doc_ids:
                relevant_chunks = await search_relevant_chunks(user_prompt, n_results=3, document_ids=gen_doc_ids)

        context = "\n\n".join(relevant_chunks)

        request_time = activity_tracker.request_time if activity_tracker is not None else datetime.utcnow()
        if minutes_since_last_visit is None or minutes_since_last_message is None:
            gap_visit, gap_msg = await get_user_ai_activity_gaps(
                session,
                user_id=user.id,
                topic_id=active_topic_id,
                now=request_time,
            )
            if minutes_since_last_visit is None:
                minutes_since_last_visit = gap_visit
            if minutes_since_last_message is None:
                minutes_since_last_message = gap_msg

        if activity_tracker is None:
            activity_tracker = ActivityTracker(
                async_session_maker,
                user_id=user.id,
                topic_id=active_topic_id,
                request_time=request_time,
                track_user_activity=track_user_activity,
            )

        request_layout = await build_conversational_request_layout(
            session,
            user=user,
            ai_config=ai_config,
            dialogue_id=active_dialogue_id,
            topic_id=active_topic_id,
            current_user_content=user_prompt,
            exclude_message_id=exclude_message_id,
            stable_system_prompt=stable_system_prompt,
            minutes_since_last_visit=minutes_since_last_visit,
            minutes_since_last_message=minutes_since_last_message,
            knowledge_context=context,
            service_capabilities=MAX_CAPABILITIES,
        )
        temperature = _resolve_temperature(ai_config)
        request_group_id = uuid.uuid4().hex[:12]
        primary_capture: dict = {}
        fallback_capture: dict = {}
        primary_start = time.monotonic()
        primary_succeeded = False
        primary_log_id = None
        fb_log_id = None

        try:
            result = await _dispatch_provider(
                ai_config,
                request_layout,
                request_capture=primary_capture,
                activity_tracker=activity_tracker,
            )
            primary_latency = int((time.monotonic() - primary_start) * 1000)
            actual_provider, actual_model = _extract_effective_provider_and_model(
                primary_capture,
                default_provider=actual_provider,
                default_model=actual_model,
            )
            visible_text, service_blocks, invalid_data_blocks = extract_service_data(result)
            if invalid_data_blocks:
                log.warning("AI returned %s invalid DATA block(s) for user %s", invalid_data_blocks, user_id)

            primary_log_id = await record_ai_attempt_log(
                session,
                user_id=user_id,
                platform="max",
                dialogue_id=active_dialogue_id,
                topic_id=active_topic_id,
                topic_name=active_topic.name if active_topic else None,
                request_type=request_type or "chat",
                provider=actual_provider,
                model=actual_model,
                prompt_summary=user_prompt if user_prompt else None,
                request_capture=primary_capture,
                raw_response=result,
                clean_text=visible_text,
                latency_ms=primary_latency,
                status="success",
                request_group_id=request_group_id,
                attempt_no=1,
                attempt_role="primary",
            )
            primary_succeeded = True
            log.info("AI response generated user_id=%s provider=%s topic_id=%s", user_id, actual_provider, active_topic_id)
        except (AIServiceError, Exception) as primary_err:
            primary_latency = int((time.monotonic() - primary_start) * 1000)
            primary_prov, primary_mod = _extract_effective_provider_and_model(
                primary_capture,
                default_provider=actual_provider,
                default_model=actual_model,
            )
            err_meta = extract_error_metadata(primary_err, provider=primary_prov)
            primary_log_id = await record_ai_attempt_log(
                session,
                user_id=user_id,
                platform="max",
                dialogue_id=active_dialogue_id,
                topic_id=active_topic_id,
                topic_name=active_topic.name if active_topic else None,
                request_type=request_type or "chat",
                provider=primary_prov,
                model=primary_mod,
                prompt_summary=user_prompt if user_prompt else None,
                request_capture=primary_capture,
                raw_response="",
                latency_ms=primary_latency,
                status="error",
                request_group_id=request_group_id,
                attempt_no=1,
                attempt_role="primary",
                error_type=err_meta["error_type"],
                error_message=err_meta["error_message"],
                error_classification=err_meta["error_classification"],
                http_status=err_meta["http_status"],
                finish_reason=err_meta["finish_reason"],
                diagnostics=err_meta["diagnostics"],
                provider_response_payload=err_meta["provider_response_payload"],
            )

            # Check output budget exhausted alert on primary
            if (
                getattr(primary_err, "classification", None) == "output_budget_exhausted"
                or err_meta["error_classification"] == "output_budget_exhausted"
                or (err_meta["diagnostics"] and getattr(err_meta["diagnostics"], "output_budget_exhausted", False))
            ):
                diag = err_meta["diagnostics"]
                await send_output_budget_exhausted_alert(
                    bot=None,
                    platform="max",
                    user_id=user_id,
                    provider=primary_prov,
                    model=primary_mod,
                    finish_reason=getattr(diag, "finish_reason", "length") if diag else "length",
                    visible_content_length=getattr(diag, "visible_content_length", 0) if diag else 0,
                    reasoning_content_length=getattr(diag, "reasoning_content_length", 0) if diag else 0,
                    max_tokens=DEEPSEEK_CHAT_MAX_TOKENS,
                    ai_log_id=primary_log_id,
                )

            # Try fallback provider if configured
            fb_provider = getattr(ai_config, "fallback_provider", None)
            fb_model = getattr(ai_config, "fallback_model", None)
            allow_fallback = getattr(ai_config, "allow_fallback", False)
            fallback_succeeded = False
            if allow_fallback and fb_provider and fb_model:
                fb_key = fb_provider.strip().lower()
                if fb_key in {"claude", "anthropic"}:
                    fb_api_key = getattr(ai_config, "claude_api_key", None) or getattr(ai_config, "anthropic_api_key", None)
                else:
                    fb_api_key = getattr(ai_config, f"{fb_key}_api_key", None)
                if fb_api_key:
                    log.warning("Primary provider '%s' failed (%s), falling back to '%s'", ai_config.provider, primary_err, fb_provider)
                    fb_start = time.monotonic()
                    fb_timeout = float(getattr(ai_config, "fallback_timeout", 60) or 60.0)

                    async def _invoke_fb():
                        if fb_key == "openai":
                            return await _call_openai(
                                fb_api_key, fb_model, [], temperature,
                                request_layout=request_layout,
                                request_capture=fallback_capture,
                                activity_tracker=activity_tracker,
                            )
                        elif fb_key in {"claude", "anthropic"}:
                            return await _call_claude(
                                fb_api_key, fb_model, [], stable_system_prompt, temperature,
                                request_layout=request_layout,
                                request_capture=fallback_capture,
                                activity_tracker=activity_tracker,
                            )
                        elif fb_key == "gemini":
                            return await _call_gemini(
                                fb_api_key, fb_model, [], stable_system_prompt, temperature,
                                request_layout=request_layout,
                                request_capture=fallback_capture,
                                activity_tracker=activity_tracker,
                            )
                        elif fb_key == "deepseek":
                            return await _call_deepseek(
                                fb_api_key, fb_model, [], temperature,
                                request_layout=request_layout,
                                request_capture=fallback_capture,
                                activity_tracker=activity_tracker,
                            )
                        elif fb_key == "kie":
                            return await _call_kie_text_chat(
                                fb_api_key, _get_kie_base_url(ai_config), fb_model, [],
                                stable_system_prompt, temperature,
                                request_layout=request_layout,
                                request_capture=fallback_capture,
                                activity_tracker=activity_tracker,
                            )
                        else:
                            raise AIServiceError(f"Неизвестный фолбэк провайдер: {fb_provider}")

                    try:
                        try:
                            result = await asyncio.wait_for(_invoke_fb(), timeout=fb_timeout)
                        except asyncio.CancelledError:
                            raise
                        except (asyncio.TimeoutError, TimeoutError) as timeout_exc:
                            err = AIServiceError(f"Fallback AI provider {fb_provider} timed out after {fb_timeout}s")
                            err.classification = "timeout"
                            raise err from timeout_exc

                        result = _validate_text_response(result, provider=fb_key)
                        fb_latency = int((time.monotonic() - fb_start) * 1000)
                        actual_provider, actual_model = _extract_effective_provider_and_model(
                            fallback_capture,
                            default_provider=str(fb_provider),
                            default_model=str(fb_model),
                        )
                        visible_text, service_blocks, invalid_data_blocks = extract_service_data(result)
                        if invalid_data_blocks:
                            log.warning("AI returned %s invalid DATA block(s) for user %s", invalid_data_blocks, user_id)

                        fb_log_id = await record_ai_attempt_log(
                            session,
                            user_id=user_id,
                            platform="max",
                            dialogue_id=active_dialogue_id,
                            topic_id=active_topic_id,
                            topic_name=active_topic.name if active_topic else None,
                            request_type=request_type or "chat",
                            provider=actual_provider,
                            model=actual_model,
                            prompt_summary=user_prompt if user_prompt else None,
                            request_capture=fallback_capture,
                            raw_response=result,
                            clean_text=visible_text,
                            latency_ms=fb_latency,
                            status="success",
                            request_group_id=request_group_id,
                            attempt_no=2,
                            attempt_role="fallback",
                        )
                        fallback_succeeded = True
                        log.info("Fallback response generated user_id=%s provider=%s", user_id, actual_provider)
                    except Exception as fb_err:
                        fb_latency = int((time.monotonic() - fb_start) * 1000)
                        fb_prov, fb_mod = _extract_effective_provider_and_model(
                            fallback_capture,
                            default_provider=str(fb_provider),
                            default_model=str(fb_model),
                        )
                        fb_err_meta = extract_error_metadata(fb_err, provider=fb_prov)
                        fb_log_id = await record_ai_attempt_log(
                            session,
                            user_id=user_id,
                            platform="max",
                            dialogue_id=active_dialogue_id,
                            topic_id=active_topic_id,
                            topic_name=active_topic.name if active_topic else None,
                            request_type=request_type or "chat",
                            provider=fb_prov,
                            model=fb_mod,
                            prompt_summary=user_prompt if user_prompt else None,
                            request_capture=fallback_capture,
                            raw_response="",
                            latency_ms=fb_latency,
                            status="error",
                            request_group_id=request_group_id,
                            attempt_no=2,
                            attempt_role="fallback",
                            error_type=fb_err_meta["error_type"],
                            error_message=fb_err_meta["error_message"],
                            error_classification=fb_err_meta["error_classification"],
                            http_status=fb_err_meta["http_status"],
                            finish_reason=fb_err_meta["finish_reason"],
                            diagnostics=fb_err_meta["diagnostics"],
                            provider_response_payload=fb_err_meta["provider_response_payload"],
                        )
                        log.error("Fallback provider '%s' also failed: %s", fb_provider, fb_err)
                        ai_log_ids = [i for i in (primary_log_id, fb_log_id) if i is not None]
                        await send_terminal_ai_failure_alert(
                            bot=None,
                            platform="max",
                            user_id=user_id,
                            primary_provider=primary_prov,
                            primary_model=primary_mod,
                            fallback_provider=fb_prov,
                            fallback_model=fb_mod,
                            exception=fb_err,
                            classification=fb_err_meta["error_classification"],
                            ai_log_ids=ai_log_ids if ai_log_ids else None,
                        )
                        service_err = AIServiceError(
                            f"Основной провайдер ({ai_config.provider}) и резервный ({fb_provider}) недоступны"
                        )
                        service_err.ai_log_ids = ai_log_ids
                        service_err.classification = getattr(fb_err, "classification", fb_err_meta["error_classification"])
                        raise service_err from fb_err

            if not fallback_succeeded:
                ai_log_ids = [i for i in (primary_log_id, fb_log_id) if i is not None]
                await send_terminal_ai_failure_alert(
                    bot=None,
                    platform="max",
                    user_id=user_id,
                    primary_provider=primary_prov,
                    primary_model=primary_mod,
                    fallback_provider=fb_provider,
                    fallback_model=fb_model,
                    exception=primary_err,
                    classification=err_meta["error_classification"],
                    ai_log_ids=ai_log_ids if ai_log_ids else None,
                )
                if isinstance(primary_err, AIServiceError):
                    log.exception("AI request failed user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, user.current_topic_id)
                    primary_err.ai_log_ids = ai_log_ids
                    raise
                log.exception("Unexpected AI request failure user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, user.current_topic_id)
                service_err = AIServiceError(f"Ошибка при обращении к AI-провайдеру: {primary_err}")
                service_err.ai_log_ids = ai_log_ids
                raise service_err from primary_err

        if service_blocks:
            try:
                await apply_service_data_blocks(
                    session,
                    user=user,
                    dialogue_id=active_dialogue_id,
                    topic_id=active_topic_id,
                    blocks=service_blocks,
                )
                await session.commit()
            except Exception as exc:
                await session.rollback()
                log.exception("Could not save shared AI service data for user %s: %s", user_id, exc)
                raise AIServiceError(f"Ошибка сохранения метаданных диалога: {exc}") from exc

        return visible_text


async def get_ai_response_direct(
    user_id: int,
    system_prompt: str,
    user_prompt: str,
    *,
    dialogue_id: int | None = None,
    topic_id: int | None = None,
    track_user_activity: bool = False,
    activity_tracker: ActivityTracker | None = None,
    minutes_since_last_visit: int | None = None,
    minutes_since_last_message: int | None = None,
) -> str:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            raise AIServiceError("Пользователь не найден")
        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            raise AIServiceError("AIConfig не найден")
        active_dialogue_id = dialogue_id or user.current_dialogue_id or 1
        active_topic_id = topic_id if topic_id is not None else user.current_topic_id

        request_time = activity_tracker.request_time if activity_tracker is not None else datetime.utcnow()
        if minutes_since_last_visit is None or minutes_since_last_message is None:
            gap_visit, gap_msg = await get_user_ai_activity_gaps(
                session,
                user_id=user.id,
                topic_id=active_topic_id,
                now=request_time,
            )
            if minutes_since_last_visit is None:
                minutes_since_last_visit = gap_visit
            if minutes_since_last_message is None:
                minutes_since_last_message = gap_msg

        if activity_tracker is None and track_user_activity:
            activity_tracker = ActivityTracker(
                async_session_maker,
                user_id=user.id,
                topic_id=active_topic_id,
                request_time=request_time,
                track_user_activity=True,
            )

        request_layout = await build_isolated_request_layout(
            session,
            user=user,
            ai_config=ai_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            dialogue_id=active_dialogue_id,
            topic_id=active_topic_id,
            minutes_since_last_visit=minutes_since_last_visit,
            minutes_since_last_message=minutes_since_last_message,
            service_capabilities=MAX_CAPABILITIES,
        )

        try:
            result = await _dispatch_provider(ai_config, request_layout, activity_tracker=activity_tracker)
            log.info("AI direct response generated user_id=%s provider=%s", user_id, ai_config.provider)
        except AIServiceError:
            log.exception("AI direct request failed user_id=%s provider=%s", user_id, ai_config.provider)
            raise
        except Exception as exc:
            log.exception("Unexpected AI direct request failure user_id=%s provider=%s", user_id, ai_config.provider)
            raise AIServiceError(f"Ошибка при прямом обращении к AI-провайдеру: {exc}") from exc

        visible_text, service_blocks, invalid_data_blocks = extract_service_data(result)
        if invalid_data_blocks:
            log.warning("Direct AI returned %s invalid DATA block(s) for user %s", invalid_data_blocks, user_id)
        if service_blocks:
            try:
                await apply_service_data_blocks(
                    session,
                    user=user,
                    dialogue_id=active_dialogue_id,
                    topic_id=active_topic_id,
                    blocks=service_blocks,
                )
                await session.commit()
            except Exception as exc:
                await session.rollback()
                log.exception("Could not save direct AI service data for user %s: %s", user_id, exc)
                raise AIServiceError(f"Ошибка сохранения метаданных диалога: {exc}") from exc
        return visible_text


# ---------------------------------------------------------------------------
# Voice Transcription
# ---------------------------------------------------------------------------

async def _transcribe_openai(api_key: str, file_bytes: bytes, filename: str) -> str:
    ensure_model_available(PROVIDER_OPENAI, "whisper-1", channel="transcription")
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
    transcription = await client.audio.transcriptions.create(model="whisper-1", file=(filename, file_bytes))
    return transcription.text


async def _transcribe_gemini(api_key: str, model: str, file_bytes: bytes, filename: str) -> str:
    import httpx

    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type or not mime_type.startswith("audio/"):
        mime_type = "audio/ogg"
    b64_data = base64.b64encode(file_bytes).decode()
    target_model = model or "gemini-3.7-flash"
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="transcription")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": "Сделай транскрипцию этой речи. Язык: русский. Верни только текст."},
                {"inline_data": {"mime_type": mime_type, "data": b64_data}},
            ]
        }]
    }
    async with httpx.AsyncClient(timeout=60.0, transport=_build_gemini_proxy_transport()) as http:
        resp = await http.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise AIServiceError("Gemini transcription returned empty candidates")
    return candidates[0]["content"]["parts"][0]["text"]


async def transcribe_audio(file_bytes: bytes, filename: str = "audio.ogg") -> str:
    """Transcribe audio bytes using the configured provider."""
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
    if not config:
        raise AIServiceError("AIConfig не найден")

    provider = (config.transcription_provider or "OpenAI").strip()
    if provider == "None":
        raise AIServiceError("Распознавание аудио отключено")
    if provider == "Gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для транскрибации не задан")
        gemini_stt_model = get_default_model(PROVIDER_GEMINI, channel="transcription")
        return await _transcribe_gemini(api_key, gemini_stt_model, file_bytes, filename)
    if provider == "KIE":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для транскрибации не задан")
        model = getattr(config, "kie_transcription_model", None) or "elevenlabs/speech-to-text"
        try:
            return await _transcribe_kie(
                api_key,
                _get_kie_base_url(config),
                _get_kie_upload_base_url(config),
                model,
                file_bytes,
                filename,
            )
        except AIServiceError as exc:
            if not config.gemini_api_key:
                raise
            log.warning("KIE transcription failed (%s), falling back to Gemini", exc)
            gemini_stt_model = get_default_model(PROVIDER_GEMINI, channel="transcription")
            return await _transcribe_gemini(
                config.gemini_api_key,
                gemini_stt_model,
                file_bytes,
                filename,
            )
    # Default: OpenAI
    api_key = config.openai_api_key
    if not api_key:
        raise AIServiceError("API ключ OpenAI для транскрибации не задан")
    return await _transcribe_openai(api_key, file_bytes, filename)


# ---------------------------------------------------------------------------
# Image Analysis (Vision)
# ---------------------------------------------------------------------------

async def _analyze_gemini(
    api_key: str,
    model: str,
    image_bytes: bytes,
    system_prompt: str,
    prompt: str,
    temperature: float,
    history: list = None,
    shared_instructions: tuple[str, ...] = (),
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    timeout: float = 25.0,
    request_capture: dict | None = None,
) -> str:
    b64_data = base64.b64encode(image_bytes).decode()
    target_model = model or "gemini-3.7-flash"
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="vision")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    
    layout = request_layout or AIRequestLayout(
        stable_system_prompt=system_prompt,
        shared_instructions=shared_instructions,
        history=normalize_request_messages(history),
    )
    layout = layout.with_current_user_content([
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}},
    ])
    contents = build_gemini_contents(layout)
    generation_config: dict = {"maxOutputTokens": get_provider_vision_max_tokens(PROVIDER_GEMINI)}
    if not (target_model.startswith("gemini-3.7") or target_model.startswith("gemini-3.6")):
        generation_config["temperature"] = temperature

    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": build_gemini_system_parts(layout) or [{"text": ""}],
        },
        "generationConfig": generation_config,
    }
    _capture_ai_request(request_capture, provider="Gemini", endpoint=url, payload=payload)
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    httpx_timeout = build_vision_httpx_timeout(timeout)
    async with httpx.AsyncClient(timeout=httpx_timeout, transport=_build_gemini_proxy_transport()) as http:
        resp = await http.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates", [])
    cand = candidates[0] if candidates else {}
    finish_reason = cand.get("finishReason")
    usage_meta = data.get("usageMetadata")

    safe_resp_payload = None
    try:
        import json
        safe_resp_payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        safe_resp_payload = None

    if request_capture is not None:
        request_capture["http_status"] = resp.status_code
        request_capture["finish_reason"] = finish_reason
        if usage_meta:
            request_capture["usage"] = usage_meta
        if safe_resp_payload is not None:
            request_capture["provider_response_payload"] = safe_resp_payload

    diag_meta = {"usage": usage_meta} if usage_meta else None

    if not candidates:
        prompt_feedback = data.get("promptFeedback", {})
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            raise attach_error_metadata(
                AIServiceError(f"Gemini vision blocked prompt: {block_reason}"),
                classification="provider_rejection",
                finish_reason=block_reason,
                http_status=resp.status_code,
                diagnostics=diag_meta,
                provider_response_payload=safe_resp_payload,
            )
        raise attach_error_metadata(
            AIResponseError("Gemini vision returned empty candidates"),
            classification="empty_response",
            http_status=resp.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        raise attach_error_metadata(
            AIServiceError(f"Gemini vision response rejected: {finish_reason}"),
            classification="provider_rejection",
            finish_reason=finish_reason,
            http_status=resp.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if finish_reason == "MAX_TOKENS":
        raise attach_error_metadata(
            AIResponseError("Gemini vision response exceeded token budget"),
            classification="output_budget_exhausted",
            finish_reason="MAX_TOKENS",
            http_status=resp.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    parts = cand.get("content", {}).get("parts", [])
    if not parts or not parts[0].get("text"):
        raise attach_error_metadata(
            AIResponseError("Gemini vision returned empty text"),
            classification="empty_response",
            finish_reason=finish_reason,
            http_status=resp.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )
    text = parts[0]["text"]
    if "Ошибка: Не удалось получить текст из ответа Gemini Vision." in text:
        raise attach_error_metadata(
            AIResponseError("Gemini vision returned internal fallback text"),
            classification="empty_response",
            finish_reason=finish_reason,
            http_status=resp.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )
    if not text.strip():
        raise attach_error_metadata(
            AIResponseError("Gemini vision returned whitespace text"),
            classification="empty_response",
            finish_reason=finish_reason,
            http_status=resp.status_code,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )
    return text


async def _analyze_openai(
    api_key: str,
    model: str,
    image_bytes: bytes,
    system_prompt: str,
    prompt: str,
    temperature: float,
    history: list = None,
    shared_instructions: tuple[str, ...] = (),
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    timeout: float = 25.0,
    request_capture: dict | None = None,
) -> str:
    target_model = model or "gpt-5.6-terra"
    ensure_model_available(PROVIDER_OPENAI, target_model, channel="vision")
    b64_data = base64.b64encode(image_bytes).decode()
    base_url = os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=timeout,
    )
    layout = request_layout or AIRequestLayout(
        stable_system_prompt=system_prompt,
        shared_instructions=shared_instructions,
        history=normalize_request_messages(history),
    )
    layout = layout.with_current_user_content([
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
    ])
    payload: dict = {
        "model": target_model,
        "messages": build_openai_chat_messages(layout),
        "max_completion_tokens": get_provider_vision_max_tokens(PROVIDER_OPENAI),
    }
    if not target_model.startswith("gpt-5.6"):
        payload["temperature"] = temperature
    _capture_ai_request(request_capture, provider="OpenAI", endpoint=f"{base_url.rstrip('/')}/chat/completions", payload=payload)
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    response = await client.chat.completions.create(**payload)

    choices = getattr(response, "choices", None)
    choice = choices[0] if choices else None
    finish_reason = getattr(choice, "finish_reason", None) if choice else None
    msg = getattr(choice, "message", None) if choice else None
    refusal = getattr(msg, "refusal", None) if msg else None
    content = (getattr(msg, "content", "") or "") if msg else ""

    usage_dict = None
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        try:
            usage_dict = usage_obj.model_dump()
        except Exception:
            usage_dict = getattr(usage_obj, "__dict__", str(usage_obj))

    safe_resp_payload = None
    try:
        import json
        safe_resp = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
            "usage": usage_dict,
        }
        safe_resp_payload = json.dumps(safe_resp, ensure_ascii=False)
    except Exception:
        safe_resp_payload = None

    if request_capture is not None:
        request_capture["http_status"] = 200
        request_capture["finish_reason"] = finish_reason
        if usage_dict is not None:
            request_capture["usage"] = usage_dict
        if safe_resp_payload is not None:
            request_capture["provider_response_payload"] = safe_resp_payload

    diag_meta = {"usage": usage_dict} if usage_dict else None

    if not choices:
        raise attach_error_metadata(
            AIResponseError("OpenAI vision returned empty choices"),
            classification="empty_response",
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    has_refusal = isinstance(refusal, str) and bool(refusal.strip())
    is_content_filter = (finish_reason == "content_filter")
    if has_refusal or is_content_filter:
        rejection_detail = refusal if has_refusal else finish_reason
        raise attach_error_metadata(
            AIServiceError(f"OpenAI vision response blocked: {rejection_detail}"),
            classification="provider_rejection",
            finish_reason="content_filter" if is_content_filter else "refusal",
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if finish_reason == "length":
        raise attach_error_metadata(
            AIResponseError("OpenAI vision response exceeded token budget"),
            classification="output_budget_exhausted",
            finish_reason="length",
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if not content or not content.strip():
        raise attach_error_metadata(
            AIResponseError("OpenAI vision returned empty or whitespace text"),
            classification="empty_response",
            finish_reason=str(finish_reason) if finish_reason else None,
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )
    return content


async def _analyze_claude(
    api_key: str,
    model: str,
    image_bytes: bytes,
    system_prompt: str,
    prompt: str,
    temperature: float,
    history: list = None,
    shared_instructions: tuple[str, ...] = (),
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    timeout: float = 25.0,
    request_capture: dict | None = None,
) -> str:
    target_model = model or "claude-sonnet-5"
    ensure_model_available(PROVIDER_CLAUDE, target_model, channel="vision")
    b64_data = base64.b64encode(image_bytes).decode()
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        max_retries=0,
        timeout=timeout,
    )
    layout = request_layout or AIRequestLayout(
        stable_system_prompt=system_prompt,
        shared_instructions=shared_instructions,
        history=normalize_request_messages(history),
    )
    layout = layout.with_current_user_content([
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_data}},
        {"type": "text", "text": prompt},
    ])
    claude_messages = [
        {"role": message.role, "content": message.content}
        for message in layout.history
    ]
    claude_messages.append({"role": "user", "content": layout.current_user_content})
    payload: dict = {
        "model": target_model,
        "max_tokens": get_provider_vision_max_tokens(PROVIDER_CLAUDE),
        "system": build_anthropic_system(layout),
        "messages": claude_messages,
    }
    if not should_omit_claude_sampling(target_model):
        payload["temperature"] = temperature
    _capture_ai_request(request_capture, provider="Claude", endpoint="https://api.anthropic.com/v1/messages", payload=payload)
    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            log.warning("Failed to mark activity before outbound call: %s", act_err)
    response = await client.messages.create(**payload)

    stop_reason = getattr(response, "stop_reason", None)
    usage_dict = None
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        try:
            usage_dict = usage_obj.model_dump()
        except Exception:
            usage_dict = getattr(usage_obj, "__dict__", str(usage_obj))

    first_block = response.content[0] if getattr(response, "content", None) else None
    content = (getattr(first_block, "text", "") or "") if first_block else ""

    safe_resp_payload = None
    try:
        import json
        safe_resp = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "stop_reason": stop_reason,
            "usage": usage_dict,
            "content": [{"type": "text", "text": content}] if content else [],
        }
        safe_resp_payload = json.dumps(safe_resp, ensure_ascii=False)
    except Exception:
        safe_resp_payload = None

    if request_capture is not None:
        request_capture["http_status"] = 200
        request_capture["finish_reason"] = stop_reason
        if usage_dict is not None:
            request_capture["usage"] = usage_dict
        if safe_resp_payload is not None:
            request_capture["provider_response_payload"] = safe_resp_payload

    diag_meta = {"usage": usage_dict} if usage_dict else None

    if not response or not getattr(response, "content", None):
        raise attach_error_metadata(
            AIResponseError("Claude vision returned empty content"),
            classification="empty_response",
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if stop_reason in {"refusal", "safety"}:
        raise attach_error_metadata(
            AIServiceError("Claude vision request refused by model"),
            classification="provider_rejection",
            finish_reason=stop_reason,
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if stop_reason == "max_tokens":
        raise attach_error_metadata(
            AIResponseError("Claude vision response exceeded token budget"),
            classification="output_budget_exhausted",
            finish_reason="max_tokens",
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )

    if not content or not content.strip():
        raise attach_error_metadata(
            AIResponseError("Claude vision returned empty or whitespace text"),
            classification="empty_response",
            finish_reason=str(stop_reason) if stop_reason else None,
            http_status=200,
            diagnostics=diag_meta,
            provider_response_payload=safe_resp_payload,
        )
    return content


async def analyze_image(
    user_id: int,
    image_bytes: bytes,
    prompt: str,
    *,
    activity_tracker: ActivityTracker | None = None,
    exclude_message_id: int | None = None,
    execution_context: VisionExecutionContext | None = None,
) -> str:
    """Analyze image with the configured vision provider."""
    async with async_session_maker() as session:
        user = await session.scalar(
            select(User)
            .options(selectinload(User.current_topic))
            .where(User.id == user_id)
        )
        if not user:
            raise AIServiceError("Пользователь не найден")
        config = await session.get(AIConfig, 1)
        if not config:
            raise AIServiceError("AIConfig не найден")

        if not getattr(config, "vision_provider", None) or config.vision_provider == "None":
            raise AIServiceError("Обработка изображений отключена администратором")

        primary_provider = (config.vision_provider or "Gemini").strip()
        if primary_provider not in (PROVIDER_OPENAI, PROVIDER_CLAUDE, PROVIDER_GEMINI, PROVIDER_KIE):
            err = AIServiceError(f"Неподдерживаемый провайдер для vision: {primary_provider}")
            err.classification = "configuration"
            raise err

        configured_model = getattr(config, "vision_model", None)
        if configured_model and is_retired_model(configured_model):
            primary_model = get_default_model(primary_provider, channel="vision")
        else:
            primary_model = configured_model or get_default_model(primary_provider, channel="vision")

        try:
            ensure_model_available(primary_provider, primary_model, channel="vision")
        except Exception as exc:
            err = AIServiceError(f"Недопустимая модель vision {configured_model} для {primary_provider}: {exc}")
            err.classification = "configuration"
            raise err from exc

        primary_api_key = None
        if primary_provider == PROVIDER_OPENAI:
            primary_api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
        elif primary_provider == PROVIDER_CLAUDE:
            primary_api_key = config.claude_api_key
        elif primary_provider == PROVIDER_GEMINI:
            primary_api_key = config.gemini_api_key
        elif primary_provider == PROVIDER_KIE:
            primary_api_key = getattr(config, "kie_api_key", None)

        if not primary_api_key:
            err = AIServiceError(f"API ключ для {primary_provider} (Vision) не установлен.")
            err.classification = "configuration"
            raise err

        active_dialogue_id = user.current_dialogue_id
        active_topic_id = user.current_topic_id

        request_time = activity_tracker.request_time if activity_tracker is not None else datetime.utcnow()
        gap_visit, gap_msg = await get_user_ai_activity_gaps(
            session,
            user_id=user.id,
            topic_id=active_topic_id,
            now=request_time,
        )

        if activity_tracker is None:
            activity_tracker = ActivityTracker(
                async_session_maker,
                user_id=user.id,
                topic_id=active_topic_id,
                request_time=request_time,
                track_user_activity=True,
            )

        photo_instructions = (
            "\n\nИНСТРУКЦИЯ ПО АНАЛИЗУ ФОТО:\n"
            "1. Если пользователь просит ИЗМЕНИТЬ это фото или 'сделать так же', добавь в конце: EDIT_IMG: <prompt on english>.\n"
            "2. Если нужно создать НОВОЕ фото с нуля, добавь в конце: GEN_IMG: <prompt on english>.\n"
            "3. ВАЖНО: Диалог уже начат. НЕ здоровайся, не представляйся и не используй вежливые вступления. Сразу переходи к сути разбора изображения."
        )

        stable_system_prompt = _build_user_system_prompt(user, config, user.current_topic)

        request_layout = await build_conversational_request_layout(
            session,
            user=user,
            ai_config=config,
            dialogue_id=active_dialogue_id,
            topic_id=active_topic_id,
            exclude_message_id=exclude_message_id,
            stable_system_prompt=stable_system_prompt,
            minutes_since_last_visit=gap_visit,
            minutes_since_last_message=gap_msg,
            service_capabilities=MAX_CAPABILITIES,
            modality_instructions=(photo_instructions,),
        )

    if execution_context is not None:
        execution_context = VisionExecutionContext(
            user_id=user.id,
            chat_id=execution_context.chat_id,
            dialogue_id=active_dialogue_id,
            topic_id=active_topic_id,
            topic_name=getattr(user.current_topic, "name", None) if user.current_topic else None,
            platform="max",
            bot_name=execution_context.bot_name or "MaxBot",
            bot=execution_context.bot,
            full_name=getattr(user, "name", None) or getattr(user, "first_name", None),
            username=getattr(user, "username", None),
        )

    deadline_tracker = VisionDeadlineTracker(total_deadline_sec=85.0)
    primary_provider = (config.vision_provider or "Gemini").strip()
    primary_model = config.vision_model or get_default_model(primary_provider, channel="vision")
    temperature = _resolve_temperature(config)

    allow_fallback = bool(getattr(config, "allow_vision_fallback", False))
    fallback_provider = getattr(config, "vision_fallback_provider", None)
    fallback_model = getattr(config, "vision_fallback_model", None)
    eff_allow_fallback, fb_provider, fb_model = resolve_effective_vision_fallback(
        primary_provider,
        allow_fallback,
        fallback_provider,
        fallback_model,
    )

    attempts: list[dict] = []
    last_exception: Exception | None = None
    raw_result: str | None = None
    recovered_by_model_retry: bool = False
    recovered_by_provider_fallback: bool = False
    output_budget_exhausted_occurred: bool = False

    def _classify_vision_error(exc: Exception | None) -> str:
        if exc is None:
            return "unknown"
        res = classify_external_error(exc)
        return str(res[0]) if isinstance(res, (tuple, list)) else str(res)

    attempt_counter = 0
    request_group_id = str(uuid.uuid4())

    def _record_attempt_summary(
        provider: str,
        model: str,
        success: bool,
        error: str | None = None,
        classification: str | None = None,
        stage: str | None = None,
        attempt_role: str | None = None,
        fallback_kind: str | None = None,
    ) -> None:
        status = "success" if success else "error"
        if attempt_role is None:
            if stage == "provider_fallback":
                attempt_role = "fallback"
                fallback_kind = fallback_kind or "provider"
            elif stage in ("inference_retry", "kie_alternate_model"):
                attempt_role = "fallback"
                fallback_kind = fallback_kind or "model"
            else:
                attempt_role = "primary"
        attempts.append({
            "provider": provider,
            "model": model,
            "status": status,
            "classification": classification,
            "error": sanitize_vision_text(error) if error else None,
            "attempt_no": len(attempts) + 1,
            "attempt_role": attempt_role,
            "fallback_kind": fallback_kind,
            "stage": stage,
            "success": success,
        })

    async def _record_attempt_log(
        provider: str,
        model: str,
        success: bool,
        error_msg: str | None,
        classification: Any | None,
        duration_ms: int,
        http_status: int | None,
        request_payload: dict | None,
        raw_response: str | None,
        diagnostics: dict | None = None,
        *,
        finish_reason: str | None = None,
        provider_response_payload: str | None = None,
    ):
        nonlocal attempt_counter
        if execution_context is None:
            return
        attempt_counter += 1

        if finish_reason is None and isinstance(request_payload, dict):
            finish_reason = request_payload.get("finish_reason")
        if http_status is None and isinstance(request_payload, dict):
            http_status = request_payload.get("http_status")
        if provider_response_payload is None and isinstance(request_payload, dict):
            provider_response_payload = request_payload.get("provider_response_payload")

        if diagnostics is None:
            diagnostics = {}
        if isinstance(diagnostics, dict) and isinstance(request_payload, dict):
            for k in ("usage", "finish_reason", "provider_code"):
                if k in request_payload and k not in diagnostics:
                    diagnostics[k] = request_payload[k]

        sanitized_payload = sanitize_vision_request_payload(request_payload)
        sanitized_response = sanitize_vision_text(raw_response)
        sanitized_error = sanitize_vision_text(error_msg)
        sanitized_diag = sanitize_vision_request_payload(diagnostics) if diagnostics else None
        sanitized_resp_payload = sanitize_vision_text(provider_response_payload) if provider_response_payload else None

        eff_cls = (
            str(classification[0])
            if isinstance(classification, (tuple, list))
            else (str(classification) if classification else None)
        )
        stage = diagnostics.get("stage") if isinstance(diagnostics, dict) else None
        if stage == "provider_fallback":
            role = "fallback"
            if isinstance(diagnostics, dict) and "fallback_kind" not in diagnostics:
                diagnostics["fallback_kind"] = "provider"
        elif stage in ("inference_retry", "kie_alternate_model"):
            role = "fallback"
            if isinstance(diagnostics, dict) and "fallback_kind" not in diagnostics:
                diagnostics["fallback_kind"] = "model"
        else:
            role = "primary"

        status = "success" if success else "error"
        clean_text = raw_response if success else None
        if finish_reason is None and isinstance(diagnostics, dict):
            finish_reason = diagnostics.get("finish_reason")

        async with async_session_maker() as audit_session:
            await record_ai_attempt_log(
                audit_session,
                user_id=execution_context.user_id,
                dialogue_id=execution_context.dialogue_id,
                topic_id=execution_context.topic_id,
                topic_name=execution_context.topic_name,
                platform=execution_context.platform,
                request_type="vision",
                provider=provider,
                model=model,
                prompt_summary=prompt[:100] if prompt else None,
                request_capture=sanitized_payload,
                raw_response=sanitized_response,
                clean_text=clean_text,
                latency_ms=duration_ms,
                status=status,
                request_group_id=request_group_id,
                attempt_no=attempt_counter,
                attempt_role=role,
                error_message=sanitized_error,
                error_classification=eff_cls,
                http_status=http_status,
                finish_reason=finish_reason,
                diagnostics=sanitized_diag,
                provider_response_payload=sanitized_resp_payload,
            )

    async def _execute_provider_call(
        call_provider: str,
        call_model: str,
        stage_budget: float,
        *,
        file_url: str | None = None,
        request_capture: dict | None = None,
    ) -> str:
        prov = call_provider.strip()
        if prov == PROVIDER_GEMINI:
            api_key = config.gemini_api_key
            if not api_key:
                raise AIServiceError("API ключ Gemini для vision не задан")
            return await _analyze_gemini(
                api_key,
                call_model,
                image_bytes,
                request_layout.stable_system_prompt,
                prompt,
                temperature,
                history=list(request_layout.history),
                request_layout=request_layout,
                activity_tracker=activity_tracker,
                timeout=stage_budget,
                request_capture=request_capture,
            )
        elif prov in {PROVIDER_CLAUDE, "Anthropic"}:
            api_key = config.claude_api_key
            if not api_key:
                raise AIServiceError("API ключ Claude для vision не задан")
            return await _analyze_claude(
                api_key,
                call_model,
                image_bytes,
                request_layout.stable_system_prompt,
                prompt,
                temperature,
                history=list(request_layout.history),
                request_layout=request_layout,
                activity_tracker=activity_tracker,
                timeout=stage_budget,
                request_capture=request_capture,
            )
        elif prov == PROVIDER_KIE:
            api_key = getattr(config, "kie_api_key", None)
            if not api_key:
                raise AIServiceError("API ключ KIE для vision не задан")
            if not file_url:
                raise AIServiceError("KIE vision requires uploaded file URL")
            return await _call_kie_vision_inference(
                api_key,
                _get_kie_base_url(config),
                call_model,
                file_url,
                request_layout.stable_system_prompt,
                prompt,
                temperature,
                request_layout=request_layout,
                activity_tracker=activity_tracker,
                timeout=stage_budget,
                request_capture=request_capture,
            )
        elif prov == PROVIDER_OPENAI:
            api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise AIServiceError("API ключ OpenAI для vision не задан")
            return await _analyze_openai(
                api_key,
                call_model,
                image_bytes,
                request_layout.stable_system_prompt,
                prompt,
                temperature,
                history=list(request_layout.history),
                request_layout=request_layout,
                activity_tracker=activity_tracker,
                timeout=stage_budget,
                request_capture=request_capture,
            )
        else:
            err = AIServiceError(f"Неподдерживаемый провайдер для vision: {prov}")
            err.classification = "configuration"
            raise err

    # --- PRIMARY ATTEMPT(S) ---
    if primary_provider == PROVIDER_KIE:
        kie_api_key = getattr(config, "kie_api_key", None)
        if not kie_api_key:
            raise AIServiceError("API ключ KIE для vision не задан")
        candidates = order_kie_vision_candidates(
            primary_model,
            selectable_models=get_selectable_models(PROVIDER_KIE, "vision"),
        )
        upload_base_url = _get_kie_upload_base_url(config)
        upload_stage_cap = 20.0
        upload_start = time.monotonic()
        file_url: str | None = None
        upload_err: Exception | None = None

        for upload_attempt_idx in range(2):
            rem_upload = upload_stage_cap - (time.monotonic() - upload_start)
            if rem_upload < 3.0 or deadline_tracker.remaining_time() <= 5.0:
                break
            up_budget = deadline_tracker.stage_budget(max_stage_budget=rem_upload, reserve_sec=5.0, min_required=3.0)
            if up_budget < 3.0:
                break
            t0 = time.monotonic()
            try:
                file_url = await run_coro_with_timeout(
                    _upload_file_to_kie(
                        kie_api_key,
                        upload_base_url,
                        image_bytes,
                        _guess_filename(image_bytes, "vision_input", "jpg"),
                        "images",
                        timeout=up_budget,
                        activity_tracker=activity_tracker,
                    ),
                    timeout_sec=up_budget,
                )
                upload_err = None
                break
            except Exception as exc:
                t_spent = int((time.monotonic() - t0) * 1000)
                classification = _classify_vision_error(exc)
                attach_error_metadata(exc, classification=classification, stage="kie_upload")
                upload_err = exc
                if upload_attempt_idx == 0 and should_retry_kie_vision_upload(classification):
                    log.warning("KIE upload transient failure (%s), retrying upload once...", exc)
                    continue
                break

        if file_url is None:
            up_cls = _classify_vision_error(upload_err) if upload_err else "timeout"
            meta = extract_error_metadata(upload_err) if upload_err else {}
            _record_attempt_summary(
                PROVIDER_KIE,
                candidates[0] if candidates else primary_model,
                False,
                error=str(upload_err or "KIE upload timeout"),
                classification=up_cls,
                stage="kie_upload",
            )
            await _record_attempt_log(
                PROVIDER_KIE,
                candidates[0] if candidates else primary_model,
                False,
                str(upload_err or "KIE upload timeout"),
                up_cls,
                int((time.monotonic() - upload_start) * 1000),
                meta.get("http_status"),
                None,
                None,
                {"stage": "kie_upload", "inference_http_started": False},
                finish_reason=meta.get("finish_reason"),
                provider_response_payload=meta.get("provider_response_payload"),
            )
            last_exception = upload_err or AIServiceError("Не удалось загрузить изображение в KIE")
        else:
            # Inference on Candidate 1
            cand1 = candidates[0]
            stage_budget = deadline_tracker.stage_budget(max_stage_budget=25.0, reserve_sec=5.0, min_required=3.0)
            t0 = time.monotonic()
            cand1_capture = {}
            cand1_diag = {"stage": "inference"}
            try:
                raw_result = await run_coro_with_timeout(
                    _execute_provider_call(PROVIDER_KIE, cand1, stage_budget, file_url=file_url, request_capture=cand1_capture),
                    timeout_sec=stage_budget,
                )
                dur = int((time.monotonic() - t0) * 1000)
                _record_attempt_summary(PROVIDER_KIE, cand1, True, stage="inference")
                for k in ("usage", "finish_reason", "provider_code"):
                    if k in cand1_capture:
                        cand1_diag[k] = cand1_capture[k]
                await _record_attempt_log(
                    PROVIDER_KIE,
                    cand1,
                    True,
                    None,
                    None,
                    dur,
                    cand1_capture.get("http_status") or 200,
                    cand1_capture,
                    raw_result,
                    cand1_diag,
                    finish_reason=cand1_capture.get("finish_reason"),
                    provider_response_payload=cand1_capture.get("provider_response_payload"),
                )
            except Exception as exc:
                dur = int((time.monotonic() - t0) * 1000)
                classification = _classify_vision_error(exc)
                if classification == "output_budget_exhausted":
                    output_budget_exhausted_occurred = True
                attach_error_metadata(exc, classification=classification, stage="inference")
                meta = extract_error_metadata(exc)
                _record_attempt_summary(
                    PROVIDER_KIE,
                    cand1,
                    False,
                    error=str(exc),
                    classification=classification,
                    stage="inference",
                )
                for k in ("usage", "finish_reason", "provider_code"):
                    if k in cand1_capture:
                        cand1_diag[k] = cand1_capture[k]
                    elif k in meta:
                        cand1_diag[k] = meta[k]
                await _record_attempt_log(
                    PROVIDER_KIE,
                    cand1,
                    False,
                    str(exc),
                    classification,
                    dur,
                    meta.get("http_status") or cand1_capture.get("http_status"),
                    cand1_capture,
                    None,
                    cand1_diag,
                    finish_reason=meta.get("finish_reason") or cand1_capture.get("finish_reason"),
                    provider_response_payload=meta.get("provider_response_payload") or cand1_capture.get("provider_response_payload"),
                )
                last_exception = exc

                # Same-KIE candidate 2 retry
                if len(candidates) > 1 and should_retry_kie_vision_model(classification):
                    cand2 = candidates[1]
                    if deadline_tracker.remaining_time() > 5.0:
                        cand2_budget = deadline_tracker.stage_budget(max_stage_budget=20.0, reserve_sec=5.0, min_required=3.0)
                        if cand2_budget >= 3.0:
                            log.info("Retrying KIE vision with alternate model %s (reusing file_url)", cand2)
                            t2_start = time.monotonic()
                            cand2_capture = {}
                            cand2_diag = {"stage": "inference_retry", "fallback_kind": "model"}
                            try:
                                raw_result = await run_coro_with_timeout(
                                    _execute_provider_call(PROVIDER_KIE, cand2, cand2_budget, file_url=file_url, request_capture=cand2_capture),
                                    timeout_sec=cand2_budget,
                                )
                                dur2 = int((time.monotonic() - t2_start) * 1000)
                                recovered_by_model_retry = True
                                _record_attempt_summary(PROVIDER_KIE, cand2, True, stage="inference_retry", fallback_kind="model")
                                for k in ("usage", "finish_reason", "provider_code"):
                                    if k in cand2_capture:
                                        cand2_diag[k] = cand2_capture[k]
                                await _record_attempt_log(
                                    PROVIDER_KIE,
                                    cand2,
                                    True,
                                    None,
                                    None,
                                    dur2,
                                    cand2_capture.get("http_status") or 200,
                                    cand2_capture,
                                    raw_result,
                                    cand2_diag,
                                    finish_reason=cand2_capture.get("finish_reason"),
                                    provider_response_payload=cand2_capture.get("provider_response_payload"),
                                )
                            except Exception as exc2:
                                dur2 = int((time.monotonic() - t2_start) * 1000)
                                cls2 = _classify_vision_error(exc2)
                                if cls2 == "output_budget_exhausted":
                                    output_budget_exhausted_occurred = True
                                attach_error_metadata(exc2, classification=cls2, stage="inference_retry")
                                meta2 = extract_error_metadata(exc2)
                                _record_attempt_summary(
                                    PROVIDER_KIE,
                                    cand2,
                                    False,
                                    error=str(exc2),
                                    classification=cls2,
                                    stage="inference_retry",
                                    fallback_kind="model",
                                    attempt_role="fallback",
                                )
                                for k in ("usage", "finish_reason", "provider_code"):
                                    if k in cand2_capture:
                                        cand2_diag[k] = cand2_capture[k]
                                    elif k in meta2:
                                        cand2_diag[k] = meta2[k]
                                await _record_attempt_log(
                                    PROVIDER_KIE,
                                    cand2,
                                    False,
                                    str(exc2),
                                    cls2,
                                    dur2,
                                    meta2.get("http_status") or cand2_capture.get("http_status"),
                                    cand2_capture,
                                    None,
                                    cand2_diag,
                                    finish_reason=meta2.get("finish_reason") or cand2_capture.get("finish_reason"),
                                    provider_response_payload=meta2.get("provider_response_payload") or cand2_capture.get("provider_response_payload"),
                                )
                                last_exception = exc2

    else:
        # Direct provider (OpenAI, Claude, Gemini)
        stage_budget = deadline_tracker.stage_budget(max_stage_budget=25.0, reserve_sec=5.0, min_required=3.0)
        t0 = time.monotonic()
        prim_capture = {}
        prim_diag = {"stage": "inference"}
        try:
            raw_result = await run_coro_with_timeout(
                _execute_provider_call(primary_provider, primary_model, stage_budget, request_capture=prim_capture),
                timeout_sec=stage_budget,
            )
            dur = int((time.monotonic() - t0) * 1000)
            _record_attempt_summary(primary_provider, primary_model, True, stage="inference")
            for k in ("usage", "finish_reason", "provider_code"):
                if k in prim_capture:
                    prim_diag[k] = prim_capture[k]
            await _record_attempt_log(
                primary_provider,
                primary_model,
                True,
                None,
                None,
                dur,
                prim_capture.get("http_status") or 200,
                prim_capture,
                raw_result,
                prim_diag,
                finish_reason=prim_capture.get("finish_reason"),
                provider_response_payload=prim_capture.get("provider_response_payload"),
            )
        except Exception as exc:
            dur = int((time.monotonic() - t0) * 1000)
            classification = _classify_vision_error(exc)
            if classification == "output_budget_exhausted":
                output_budget_exhausted_occurred = True
            attach_error_metadata(exc, classification=classification, stage="inference")
            meta = extract_error_metadata(exc)
            _record_attempt_summary(
                primary_provider,
                primary_model,
                False,
                error=str(exc),
                classification=classification,
                stage="inference",
            )
            for k in ("usage", "finish_reason", "provider_code"):
                if k in prim_capture:
                    prim_diag[k] = prim_capture[k]
                elif k in meta:
                    prim_diag[k] = meta[k]
            await _record_attempt_log(
                primary_provider,
                primary_model,
                False,
                str(exc),
                classification,
                dur,
                meta.get("http_status") or prim_capture.get("http_status"),
                prim_capture,
                None,
                prim_diag,
                finish_reason=meta.get("finish_reason") or prim_capture.get("finish_reason"),
                provider_response_payload=meta.get("provider_response_payload") or prim_capture.get("provider_response_payload"),
            )
            last_exception = exc

    # --- PROVIDER FALLBACK STAGE ---
    fallback_attempted = False
    fb_eligible = False
    fb_budget = 0.0
    if raw_result is None and eff_allow_fallback and fb_provider and fb_model and last_exception is not None:
        is_diff = str(fb_provider).strip().lower() != str(primary_provider).strip().lower()
        is_supported = str(fb_provider).strip() in (PROVIDER_OPENAI, PROVIDER_CLAUDE, PROVIDER_GEMINI, PROVIDER_KIE)
        is_valid_model = bool(fb_model)
        try:
            if is_valid_model and is_supported:
                validate_model_selection(fb_provider, fb_model, channel="vision_fallback")
            else:
                is_valid_model = False
        except Exception:
            is_valid_model = False

        has_key = False
        if fb_provider == PROVIDER_KIE:
            has_key = bool(getattr(config, "kie_api_key", None))
        elif fb_provider == PROVIDER_GEMINI:
            has_key = bool(getattr(config, "gemini_api_key", None))
        elif fb_provider in (PROVIDER_CLAUDE, "Anthropic"):
            has_key = bool(getattr(config, "claude_api_key", None))
        elif fb_provider == PROVIDER_OPENAI:
            has_key = bool(getattr(config, "openai_api_key", None) or os.getenv("OPENAI_API_KEY"))

        last_cls = _classify_vision_error(last_exception)
        primary_tokens = get_provider_vision_max_tokens(primary_provider)
        fallback_tokens = get_provider_vision_max_tokens(fb_provider) if is_supported else 0

        if is_diff and is_supported and is_valid_model and has_key and should_use_vision_provider_fallback(last_cls, primary_tokens, fallback_tokens):
            fb_budget = deadline_tracker.stage_budget(max_stage_budget=15.0, reserve_sec=5.0, min_required=3.0)
            if fb_budget >= 3.0:
                fb_eligible = True

    if fb_eligible:
        fallback_attempted = True
        log.info("Attempting vision provider fallback to %s/%s with budget %.1fs", fb_provider, fb_model, fb_budget)
        t_fb = time.monotonic()
        fb_stage_cap = 15.0
        fb_capture = {}
        fb_diag = {"stage": "provider_fallback", "fallback_kind": "provider"}
        try:
            if fb_provider == PROVIDER_KIE:
                fb_file_url = None
                fb_upload_err = None
                for fb_up_attempt in range(2):
                    elapsed = time.monotonic() - t_fb
                    rem_stage = fb_stage_cap - elapsed
                    if rem_stage < 3.0 or deadline_tracker.remaining_time() <= 5.0:
                        if fb_upload_err is not None:
                            raise fb_upload_err
                        raise AIServiceError("Недостаточно времени для загрузки в KIE (fallback)")
                    fb_upload_budget = deadline_tracker.stage_budget(max_stage_budget=rem_stage, reserve_sec=5.0, min_required=3.0)
                    if fb_upload_budget < 3.0:
                        if fb_upload_err is not None:
                            raise fb_upload_err
                        raise AIServiceError("Недостаточно времени для загрузки в KIE (fallback)")
                    try:
                        fb_file_url = await run_coro_with_timeout(
                            _upload_file_to_kie(
                                getattr(config, "kie_api_key", None),
                                _get_kie_upload_base_url(config),
                                image_bytes,
                                _guess_filename(image_bytes, "vision_fb", "jpg"),
                                "images",
                                timeout=fb_upload_budget,
                                activity_tracker=activity_tracker,
                            ),
                            timeout_sec=fb_upload_budget,
                        )
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as up_exc:
                        fb_upload_err = up_exc
                        if fb_up_attempt == 0:
                            up_cls = _classify_vision_error(up_exc)
                            if should_retry_kie_vision_upload(up_cls):
                                rem_after_err = fb_stage_cap - (time.monotonic() - t_fb)
                                rem_tracker = deadline_tracker.stage_budget(max_stage_budget=rem_after_err, reserve_sec=5.0, min_required=3.0)
                                if rem_after_err >= 3.0 and rem_tracker >= 3.0:
                                    log.info("Retrying KIE provider fallback upload after transient error: %s", up_exc)
                                    continue
                        raise up_exc

                if not fb_file_url:
                    if fb_upload_err:
                        raise fb_upload_err
                    raise AIServiceError("Не удалось получить URL файла KIE для fallback")

                fb_spent_after_up = time.monotonic() - t_fb
                fb_inf_rem = fb_stage_cap - fb_spent_after_up
                if fb_inf_rem < 3.0:
                    raise AIServiceError("Недостаточно времени для инференса KIE (fallback)")
                fb_inf_budget = deadline_tracker.stage_budget(max_stage_budget=fb_inf_rem, reserve_sec=5.0, min_required=3.0)
                if fb_inf_budget < 3.0:
                    raise AIServiceError("Недостаточно времени для инференса KIE (fallback)")
                raw_result = await run_coro_with_timeout(
                    _execute_provider_call(
                        fb_provider,
                        fb_model,
                        fb_inf_budget,
                        file_url=fb_file_url,
                        request_capture=fb_capture,
                    ),
                    timeout_sec=fb_inf_budget,
                )
            else:
                raw_result = await run_coro_with_timeout(
                    _execute_provider_call(fb_provider, fb_model, fb_budget, request_capture=fb_capture),
                    timeout_sec=fb_budget,
                )

            dur_fb = int((time.monotonic() - t_fb) * 1000)
            recovered_by_provider_fallback = True
            _record_attempt_summary(fb_provider, fb_model, True, stage="provider_fallback", fallback_kind="provider")
            for k in ("usage", "finish_reason", "provider_code"):
                if k in fb_capture:
                    fb_diag[k] = fb_capture[k]
            await _record_attempt_log(
                fb_provider,
                fb_model,
                True,
                None,
                None,
                dur_fb,
                fb_capture.get("http_status") or 200,
                fb_capture,
                raw_result,
                fb_diag,
                finish_reason=fb_capture.get("finish_reason"),
                provider_response_payload=fb_capture.get("provider_response_payload"),
            )

            if execution_context is not None:
                try:
                    await send_ai_fallback_used_alert(
                        bot=execution_context.bot,
                        primary_provider=primary_provider,
                        primary_model=primary_model,
                        fallback_provider=fb_provider,
                        fallback_model=fb_model,
                        failure_reason=str(last_exception),
                        user=execution_context,
                        dialogue_id=execution_context.dialogue_id,
                        topic_id=execution_context.topic_id,
                        topic_name=execution_context.topic_name,
                        request_type="vision",
                        attempts=attempts,
                        platform=execution_context.platform,
                    )
                except Exception as alert_err:
                    log.warning("Failed to send fallback alert: %s", alert_err)

        except Exception as fb_exc:
            dur_fb = int((time.monotonic() - t_fb) * 1000)
            fb_cls = _classify_vision_error(fb_exc)
            if fb_cls == "output_budget_exhausted":
                output_budget_exhausted_occurred = True
            attach_error_metadata(fb_exc, classification=fb_cls, stage="provider_fallback")
            fb_meta = extract_error_metadata(fb_exc)
            _record_attempt_summary(
                fb_provider,
                fb_model,
                False,
                error=str(fb_exc),
                classification=fb_cls,
                stage="provider_fallback",
                fallback_kind="provider",
            )
            for k in ("usage", "finish_reason", "provider_code"):
                if k in fb_capture:
                    fb_diag[k] = fb_capture[k]
                elif k in fb_meta:
                    fb_diag[k] = fb_meta[k]
            await _record_attempt_log(
                fb_provider,
                fb_model,
                False,
                str(fb_exc),
                fb_cls,
                dur_fb,
                fb_meta.get("http_status") or fb_capture.get("http_status"),
                fb_capture,
                None,
                fb_diag,
                finish_reason=fb_meta.get("finish_reason") or fb_capture.get("finish_reason"),
                provider_response_payload=fb_meta.get("provider_response_payload") or fb_capture.get("provider_response_payload"),
            )
            last_exception = fb_exc

    # --- OUTCOME HANDLING ---
    if raw_result is None:
        last_cls = _classify_vision_error(last_exception) if last_exception else "unknown"
        if execution_context is not None and last_exception is not None:
            try:
                if last_cls == "output_budget_exhausted" and not fallback_attempted:
                    await send_output_budget_exhausted_alert(
                        bot=execution_context.bot,
                        user=execution_context,
                        dialogue_id=execution_context.dialogue_id,
                        topic_id=execution_context.topic_id,
                        topic_name=execution_context.topic_name,
                        provider=primary_provider,
                        model=primary_model,
                        details=str(last_exception),
                        exception=last_exception,
                        request_type="vision",
                        attempts=attempts,
                        platform=execution_context.platform,
                    )
                else:
                    await send_terminal_ai_failure_alert(
                        bot=execution_context.bot,
                        title="Терминальный сбой анализа изображения",
                        user=execution_context,
                        dialogue_id=execution_context.dialogue_id,
                        topic_id=execution_context.topic_id,
                        topic_name=execution_context.topic_name,
                        provider=primary_provider,
                        model=primary_model,
                        stage="vision_orchestrator",
                        classification=last_cls,
                        details=str(last_exception),
                        exception=last_exception,
                        request_type="vision",
                        attempts=attempts,
                        platform=execution_context.platform,
                    )
                last_exception.admin_alert_handled = True
            except Exception as alert_err:
                log.warning("Failed to send terminal vision alert: %s", alert_err)

        if isinstance(last_exception, AIServiceError):
            final_exc = last_exception
        elif last_exception is not None:
            final_exc = AIServiceError(f"Анализ изображения не удался: {last_exception}")
            final_exc.__cause__ = last_exception
        else:
            final_exc = AIServiceError("Анализ изображения не удался")

        attach_error_metadata(final_exc, classification=last_cls, stage="vision_orchestrator")
        if execution_context is not None:
            final_exc.admin_alert_handled = True
        raise final_exc

    visible_text, service_blocks, invalid_data_blocks = extract_service_data(raw_result)
    if invalid_data_blocks:
        log.warning("Vision AI returned %s invalid DATA block(s) for user %s", invalid_data_blocks, user_id)
    if service_blocks:
        async with async_session_maker() as session:
            user = await session.get(User, user_id)
            if user:
                try:
                    await apply_service_data_blocks(
                        session,
                        user=user,
                        dialogue_id=active_dialogue_id,
                        topic_id=active_topic_id,
                        blocks=service_blocks,
                    )
                    await session.commit()
                except Exception as exc:
                    await session.rollback()
                    log.exception("Could not save vision service data for user %s: %s", user_id, exc)
                    raise AIServiceError(f"Ошибка сохранения метаданных анализа изображения: {exc}") from exc
    return visible_text


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------

async def _generate_gemini(api_key: str, model: str, prompt: str) -> bytes:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="image_gen")
    try:
        return await gemini_image.generate_image(api_key, target_model, prompt)
    except gemini_image.GeminiImageResponseError as exc:
        raise AIResponseError(str(exc)) from exc
    except gemini_image.GeminiImageError as exc:
        raise AIServiceError(str(exc)) from exc


async def _generate_openai(api_key: str, prompt: str) -> bytes:
    import httpx

    model = get_default_model(PROVIDER_OPENAI, channel="image_gen")
    ensure_model_available(PROVIDER_OPENAI, model, channel="image_gen")
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
    response = await client.images.generate(model=model, prompt=prompt, n=1, size="1024x1024")
    if not response.data:
        raise AIServiceError("OpenAI image generation returned no data")
    img_data = response.data[0]
    if img_data.b64_json:
        return base64.b64decode(img_data.b64_json)
    if img_data.url:
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(img_data.url)
            resp.raise_for_status()
            return resp.content
    raise AIServiceError("OpenAI image generation returned no image data")


async def generate_image(prompt: str) -> bytes:
    """Generate image from text prompt using configured provider."""
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
    if not config:
        raise AIServiceError("AIConfig не найден")

    provider = getattr(config, "image_generation_provider", None) or config.vision_provider or "OpenAI"
    provider_key = provider.strip().lower()

    if provider_key == "gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для генерации не задан")
        model = getattr(config, "image_generation_model", None) or get_default_model(PROVIDER_GEMINI, channel="image_gen")
        return await _generate_gemini(api_key, model, prompt)
    if provider_key == "kie":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для генерации не задан")
        model = getattr(config, "image_generation_model", None) or get_default_model(PROVIDER_KIE, channel="image_gen")
        ensure_model_available(PROVIDER_KIE, model, channel="image_gen")
        return await _generate_kie(api_key, _get_kie_base_url(config), model, prompt)
    # Default: OpenAI
    api_key = config.openai_api_key
    if not api_key:
        raise AIServiceError("API ключ OpenAI для генерации не задан")
    return await _generate_openai(api_key, prompt)


# ---------------------------------------------------------------------------
# Image Editing
# ---------------------------------------------------------------------------

async def _edit_gemini(api_key: str, model: str, prompt: str, image_bytes: bytes) -> bytes:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="image_edit")
    try:
        return await gemini_image.edit_image(api_key, target_model, prompt, image_bytes)
    except gemini_image.GeminiImageResponseError as exc:
        raise AIResponseError(str(exc)) from exc
    except gemini_image.GeminiImageError as exc:
        raise AIServiceError(str(exc)) from exc


async def edit_image(prompt: str, image_bytes: bytes) -> bytes:
    """Edit image using the configured provider."""
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
    if not config:
        raise AIServiceError("AIConfig не найден")

    provider = getattr(config, "image_edit_provider", None) or config.vision_provider or "KIE"
    provider_key = provider.strip().lower()

    if provider_key == "gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для редактирования не задан")
        model = getattr(config, "image_edit_model", None) or get_default_model(PROVIDER_GEMINI, channel="image_edit")
        return await _edit_gemini(api_key, model, prompt, image_bytes)
    if provider_key == "kie":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для редактирования не задан")
        model = getattr(config, "image_edit_model", None) or get_default_model(PROVIDER_KIE, channel="image_edit")
        ensure_model_available(PROVIDER_KIE, model, channel="image_edit")
        return await _edit_kie(api_key, _get_kie_base_url(config), _get_kie_upload_base_url(config), model, prompt, image_bytes)
    raise AIServiceError(f"Редактирование изображений не поддерживается для провайдера: {provider}")
