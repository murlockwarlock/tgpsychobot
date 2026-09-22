import logging
import google.generativeai as genai
import anthropic
import os
import io
import mimetypes
import base64
import asyncio
import json
import uuid
import html
import httpx
import re
from typing import Any, Iterable
from types import SimpleNamespace
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from openai import AsyncOpenAI, AuthenticationError, RateLimitError, BadRequestError
import gemini_image

import time
from datetime import datetime

from database import (async_session_maker, AIConfig, Message as DBMessage, User, Topic,
                     UserSubscription, KnowledgeBase, SubscriptionConfig, AILog)
from ai_request_builder import (
    ActivityTracker,
    build_conversational_request_layout,
    build_isolated_request_layout,
    get_user_ai_activity_gaps,
)
from media_scope import load_available_media
from memory_mode import get_memory_mode, is_global_memory_mode
from prompt_blocks import (
    DEFAULT_SERVICE_PROMPT_TEMPLATE,
    DEFAULT_SHORT_RESPONSE_INSTRUCTION,
    TELEGRAM_CAPABILITIES,
    build_media_instruction_block,
    format_available_media_text,
    render_prompt_block,
)
from automation_engine import apply_service_data_blocks, build_runtime_automation_context
from result_history import (
    TEST_RESULT_ROLE,
    ai_history_role_filter,
    select_ai_history_messages,
)
from error_reporting import (
    classify_external_error,
    exception_summary,
    extract_error_metadata,
    notify_admins_about_error,
    root_cause_exception,
    send_ai_fallback_used_alert,
    send_output_budget_exhausted_alert,
    send_terminal_ai_failure_alert,
)
from ai_log_context import record_ai_attempt_log
from vector_store import search_relevant_chunks
from user_metadata import extract_service_data
from provider_models import (
    CLAUDE_CHAT_MAX_TOKENS,
    DEEPSEEK_CHAT_MAX_TOKENS,
    DEFAULT_KIE_TRANSCRIPTION_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    GEMINI_CHAT_MAX_TOKENS,
    effective_chat_output_tokens,
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
    VisionExecutionContext,
    VisionDeadlineTracker,
    run_coro_with_timeout,
    build_vision_httpx_timeout,
    attach_error_metadata,
    sanitize_vision_request_payload,
    sanitize_vision_text,
    should_retry_kie_vision_upload,
    should_retry_kie_vision_model,
    should_use_vision_provider_fallback,
    resolve_effective_vision_fallback,
    order_kie_vision_candidates,
)
from kie_chat import (
    build_kie_chat_request,
    extract_kie_chat_response_text,
    extract_kie_chat_text,
    is_kie_error_payload,
    is_kie_insufficient_balance,
)
from subscription_context import active_subscription_flag
from translation_service import resolve_user_effective_locale
from ai_request_context import (
    AIRequestLayout,
    AIRequestMessage,
    _capture_ai_request,
    build_anthropic_system,
    build_gemini_contents,
    build_gemini_system_parts,
    build_openai_chat_messages,
    extract_effective_provider_and_model,
    neutralize_stable_prompt,
    normalize_request_messages,
)
from ai_log_context import apply_ai_log_context

class InsufficientBalanceError(Exception):
    pass


class AIServiceError(Exception):
    """Transient AI provider error (network, 5xx, etc.) — show friendly message to user."""
    pass


class AIResponseError(AIServiceError):
    """Provider returned an invalid or empty payload."""
    pass


def _validate_text_response(response_text: object, *, provider: str) -> str:
    if not isinstance(response_text, str) or not response_text.strip():
        err = AIResponseError(f"{provider} returned an empty or invalid text response")
        err.provider_response_payload = str(response_text) if response_text else None
        err.classification = "empty_response"
        raise err
    return response_text



_CURRENT_AI_CONTEXT = object()


_MISSING_CURRENT_CONTENT = object()


def _clean_request_history(history: Iterable[Any] | None) -> tuple[AIRequestMessage, ...]:
    """Normalize history while preserving the existing service-data cleanup."""
    cleaned: list[AIRequestMessage] = []
    for message in normalize_request_messages(history):
        content = message.content
        if message.role == "assistant" and isinstance(content, str) and content:
            content, _, _ = extract_service_data(content)
        if content not in (None, "", []):
            cleaned.append(AIRequestMessage(role=message.role, content=content))
    return tuple(cleaned)


def _legacy_request_layout(
    *,
    history: Iterable[Any] | None,
    system_prompt: str | None,
    context: str | None = "",
    runtime_context: str | None = "",
    current_user_content: Any = _MISSING_CURRENT_CONTENT,
) -> AIRequestLayout:
    """Adapt the pre-layout call signature without joining semantic blocks."""
    cleaned_history = _clean_request_history(history)
    current_content = current_user_content
    if current_content is _MISSING_CURRENT_CONTENT:
        if cleaned_history and cleaned_history[-1].role == "user":
            current_content = cleaned_history[-1].content
            cleaned_history = cleaned_history[:-1]
        else:
            current_content = None

    request_blocks = ()
    if context and context.strip():
        request_blocks = (f"РЕЛЕВАНТНЫЕ ДАННЫЕ ИЗ БАЗЫ ЗНАНИЙ:\n{context.strip()}",)
    return AIRequestLayout(
        stable_system_prompt=neutralize_stable_prompt(system_prompt),
        runtime_context=(runtime_context,) if runtime_context and runtime_context.strip() else (),
        request_context=request_blocks,
        history=cleaned_history,
        current_user_content=current_content,
    )


def _coerce_request_layout(
    request_layout: AIRequestLayout | None,
    *,
    history: Iterable[Any] | None,
    system_prompt: str | None,
    context: str | None = "",
    runtime_context: str | None = "",
    current_user_content: Any = _MISSING_CURRENT_CONTENT,
) -> AIRequestLayout:
    if request_layout is None:
        return _legacy_request_layout(
            history=history,
            system_prompt=system_prompt,
            context=context,
            runtime_context=runtime_context,
            current_user_content=current_user_content,
        )
    return AIRequestLayout(
        stable_system_prompt=request_layout.stable_system_prompt,
        shared_instructions=request_layout.shared_instructions,
        runtime_context=request_layout.runtime_context,
        scenario_context=request_layout.scenario_context,
        request_context=request_layout.request_context,
        history=_clean_request_history(request_layout.history),
        current_user_content=request_layout.current_user_content,
    )


def _extract_vision_user_prompt(
    user_prompt: str | None = None,
    request_layout: AIRequestLayout | None = None,
    prompt: str | None = None,
) -> str:
    if user_prompt and user_prompt.strip():
        return user_prompt.strip()
    if request_layout is not None and request_layout.current_user_content:
        content = request_layout.current_user_content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, (list, tuple)):
            for item in content:
                if isinstance(item, dict):
                    text_val = item.get("text")
                    if text_val and str(text_val).strip():
                        return str(text_val).strip()
                elif isinstance(item, str) and item.strip():
                    return item.strip()
    if prompt and prompt.strip():
        if request_layout is None or prompt.strip() != (request_layout.stable_system_prompt or "").strip():
            return prompt.strip()
    return "Опиши это изображение подробно."


def _build_async_transport_from_env(env_var_name: str, use_proxy: bool = True):
    if not use_proxy:
        return None
    import httpx

    raw_proxy = os.getenv(env_var_name)
    if not raw_proxy:
        return None

    proxy = raw_proxy.strip().strip('"').strip("'")
    if not proxy:
        return None

    return httpx.AsyncHTTPTransport(proxy=proxy)


def _normalize_provider_name(provider: str | None) -> str:
    return provider.strip().lower() if provider else ""


def _normalize_config_value(value: str | None) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _resolve_temperature(config: Any, default: float = 0.7) -> float:
    if config is None:
        return default
    val = getattr(config, "temperature", None)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


async def _notify_ai_fallback_used(
    bot,
    *,
    user: User | None,
    primary_provider: str,
    primary_model: str | None,
    fallback_provider: str,
    fallback_model: str | None,
    error: Exception,
) -> None:
    if bot is None:
        return

    try:
        classification, classified_error = classify_external_error(error, provider=primary_provider)
        primary_root = root_cause_exception(error) or error
        await notify_admins_about_error(
            bot,
            title="Основной AI-провайдер не сработал — использован резервный",
            user_id=getattr(user, "id", None),
            username=getattr(user, "username", None),
            full_name=getattr(user, "full_name", None),
            provider=primary_provider,
            model=primary_model,
            stage="ai_provider_fallback",
            details=classified_error,
            provider_attempts=(
                {
                    "provider": primary_provider,
                    "model": primary_model,
                    "status": "FAILED",
                    "classification": classification,
                    "exception_class": type(primary_root).__name__,
                    "error": exception_summary(error),
                },
                {
                    "provider": fallback_provider,
                    "model": fallback_model,
                    "status": "SUCCESS",
                    "classification": None,
                    "error": "Ответ получен",
                },
            ),
            extra={
                "fallback_provider": fallback_provider,
                "fallback_model": fallback_model,
                "fallback_status": "SUCCESS",
            },
            exception=error,
            include_traceback=False,
            level=logging.WARNING,
        )
    except Exception as notify_error:
        logging.error("Failed to send AI fallback admin notification: %s", notify_error)


def _get_kie_base_url(ai_config: AIConfig) -> str:
    return (getattr(ai_config, "kie_base_url", None) or "https://api.kie.ai").rstrip("/")


def _get_kie_upload_base_url(ai_config: AIConfig) -> str:
    return (getattr(ai_config, "kie_upload_base_url", None) or "https://kieai.redpandaai.co").rstrip("/")


def _nonempty_provider_reason(value: object, fallback: str) -> str:
    reason = str(value).strip() if value is not None else ""
    return reason or fallback


def _build_ai_attempt(
    provider: str | None,
    model: str | None,
    error: Exception | None = None,
    *,
    status: str = "FAILED",
    classification: str | None = None,
    include_context: bool = True,
) -> dict[str, str | None]:
    if error is None:
        return {
            "provider": provider,
            "model": model,
            "status": status,
            "classification": classification,
            "error": "Ответ получен" if status == "SUCCESS" else "Неизвестная ошибка",
        }
    error_classification, _ = classify_external_error(
        error,
        provider=provider,
        include_context=include_context,
    )
    root = root_cause_exception(error, include_context=include_context) or error
    return {
        "provider": provider,
        "model": model,
        "status": status,
        "classification": classification or error_classification,
        "exception_class": type(root).__name__,
        "error": exception_summary(error, include_context=include_context),
    }


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

    unique_id = uuid.uuid4().hex[:12]
    return f"{fallback_stem}_{unique_id}.{ext}"


def _guess_image_media_type(file_bytes: bytes) -> str:
    header = file_bytes[:16]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"GIF8"):
        return "image/gif"
    if header.startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _extract_kie_chat_text(payload: dict) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(part for part in text_parts if part).strip()
    return ""


def _extract_openai_chat_text(response, *, provider: str) -> str:
    if response is None:
        raise AIResponseError(f"{provider} вернул пустой ответ")

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIResponseError(f"{provider} вернул ответ без choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise AIResponseError(f"{provider} вернул пустой content")

    return content


def _is_kie_transient_failure(error: Exception | str) -> bool:
    text = str(error).lower()
    markers = [
        "connecterror",
        "readerror",
        "remoteprotocolerror",
        "server exception",
        "temporarily",
        "timed out",
        "timeout",
        "maintained",
        "maintenance",
        "internal error",
        "try again later",
        "server is currently being maintained",
    ]
    return any(marker in text for marker in markers)


async def _prepare_kie_transcription_audio(
    file_bytes: bytes,
    filename: str,
) -> tuple[bytes, str]:
    if not filename.lower().endswith((".ogg", ".oga", ".opus")):
        return file_bytes, filename

    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        wav_bytes, stderr = await process.communicate(file_bytes)
    except Exception as exc:
        raise AIServiceError(f"Не удалось запустить конвертацию OGG для KIE: {exception_summary(exc)}") from exc

    if process.returncode != 0 or not wav_bytes:
        ffmpeg_error = stderr.decode("utf-8", errors="replace").strip()
        raise AIServiceError(
            f"Не удалось конвертировать OGG в WAV для KIE: {ffmpeg_error or f'ffmpeg exit {process.returncode}'}"
        )

    stem = os.path.splitext(filename)[0]
    return wav_bytes, f"{stem}.wav"


async def _retry_kie_transcription_step(step: str, operation, attempts: int = 3):
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except InsufficientBalanceError:
            raise
        except AIServiceError as exc:
            if attempt >= attempts or not _is_kie_transient_failure(exc):
                raise
            logging.warning(
                "Retrying KIE transcription step=%s attempt=%s/%s error=%s",
                step,
                attempt + 1,
                attempts,
                exc,
            )
            await asyncio.sleep(attempt)


def _validate_kie_json_response(status_code: int, payload: dict, *, context: str) -> dict:
    raw_detail = (
        payload.get("msg") or payload.get("message") or str(payload)
        if isinstance(payload, dict)
        else str(payload)
    )
    detail = _nonempty_provider_reason(raw_detail, f"HTTP {status_code} без описания")
    if status_code != 200:
        if is_kie_insufficient_balance(status_code, payload):
            raise InsufficientBalanceError(f"KIE API Error: {detail}")
        raise AIServiceError(f"{context}: status={status_code} message={detail}")

    code = payload.get("code")
    if code not in (None, 200, "200"):
        if is_kie_insufficient_balance(status_code, payload):
            err = InsufficientBalanceError(f"KIE API Error: {detail}")
        else:
            err = AIServiceError(f"{context}: {detail}")
        err.http_status = status_code
        err.provider_code = code
        try:
            import json
            err.provider_response_payload = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
        except Exception:
            err.provider_response_payload = str(payload)
        raise err

    return payload.get("data") if isinstance(payload.get("data"), dict) else payload


def _extract_text_from_openai_message(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(part for part in text_parts if part).strip()
    return ""


def _find_first_string_value(data, candidate_keys: tuple[str, ...]) -> str | None:
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


def _load_configured_system_prompt(ai_config: AIConfig, topic_prompt_text: str | None) -> str:
    system_prompt_text = topic_prompt_text

    if not system_prompt_text:
        if ai_config.prompt_mode == 'file' and ai_config.prompt_filename:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(script_dir, "system_prompts", ai_config.prompt_filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    system_prompt_text = f.read()
            except Exception:
                system_prompt_text = ai_config.system_prompt
        else:
            system_prompt_text = ai_config.system_prompt

    return system_prompt_text or ""


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

    return (
        "{user_name}" in normalized_head
        and ("роль" in normalized_head or "ты ведёшь диалог как" in normalized_head)
    )


async def _build_request_context_blocks(
    session,
    *,
    user: User,
    dialogue_id: int,
    topic_id: int | None,
    subscription_config: SubscriptionConfig | None = None,
    load_subscription_config: bool = True,
    include_subscription_status: bool = True,
    test_context: str = "",
    short_response_instruction: str = "",
    knowledge_context: str = "",
    global_memory_context: str = "",
    scenario_context: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Build the dynamic, scenario, and request-specific semantic blocks."""
    user_name = user.name or user.first_name or "Не указано"
    user_gender = user.gender or "Не указан"
    client_lines = ["ДАННЫЕ КЛИЕНТА:", f"ИМЯ: {user_name}", f"ПОЛ: {user_gender}"]
    if user.age:
        client_lines.append(f"ВОЗРАСТ: {user.age}")
    if subscription_config is None and load_subscription_config:
        subscription_config = await session.get(SubscriptionConfig, 1)
    if include_subscription_status:
        subscription_flag = active_subscription_flag(subscription_config, user.subscription)
        if subscription_flag:
            client_lines.append(subscription_flag)

    if scenario_context is None:
        scenario_context = await build_runtime_automation_context(
            session,
            user_id=user.id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
        )

    runtime_parts = ["\n".join(client_lines)]
    if short_response_instruction and short_response_instruction.strip():
        runtime_parts.append(short_response_instruction.strip())

    request_parts = []
    if test_context and test_context.strip():
        request_parts.append(test_context.strip())
    if knowledge_context and knowledge_context.strip():
        request_parts.append("РЕЛЕВАНТНЫЕ ДАННЫЕ ИЗ БАЗЫ ЗНАНИЙ:\n" + knowledge_context.strip())
    if global_memory_context and global_memory_context.strip():
        request_parts.append(global_memory_context.strip())

    scenario_parts = ()
    if scenario_context and scenario_context.strip():
        scenario_parts = (scenario_context.strip(),)
    return tuple(runtime_parts), scenario_parts, tuple(request_parts)


async def build_ai_request_layout(
    session,
    *,
    user: User,
    dialogue_id: int,
    topic_id: int | None,
    stable_system_prompt: str,
    shared_instructions: Iterable[str] = (),
    history: Iterable[Any] | None = None,
    current_user_content: Any = None,
    subscription_config: SubscriptionConfig | None = None,
    load_subscription_config: bool = True,
    include_subscription_status: bool = True,
    test_context: str = "",
    short_response_instruction: str = "",
    knowledge_context: str = "",
    global_memory_context: str = "",
    scenario_context: str | None = None,
) -> AIRequestLayout:
    """Build the canonical request layout for a conversational request."""
    runtime_parts, scenario_parts, request_parts = await _build_request_context_blocks(
        session,
        user=user,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
        subscription_config=subscription_config,
        load_subscription_config=load_subscription_config,
        include_subscription_status=include_subscription_status,
        test_context=test_context,
        short_response_instruction=short_response_instruction,
        knowledge_context=knowledge_context,
        global_memory_context=global_memory_context,
        scenario_context=scenario_context,
    )
    return AIRequestLayout(
        stable_system_prompt=neutralize_stable_prompt(stable_system_prompt),
        shared_instructions=tuple(shared_instructions),
        runtime_context=runtime_parts,
        scenario_context=scenario_parts,
        request_context=request_parts,
        history=_clean_request_history(history),
        current_user_content=current_user_content,
    )


async def build_runtime_context(
    session,
    *,
    user: User,
    dialogue_id: int,
    topic_id: int | None,
    subscription_config: SubscriptionConfig | None = None,
    test_context: str = "",
    available_media_text: str = "",
    short_response_instruction: str = "",
    knowledge_context: str = "",
    global_memory_context: str = "",
) -> str:
    """Backward-compatible text view of the dynamic request blocks.

    New provider calls use :func:`build_ai_request_layout`; this helper stays
    for existing callers that need a display/log string.
    """
    runtime_parts, scenario_parts, request_parts = await _build_request_context_blocks(
        session,
        user=user,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
        subscription_config=subscription_config,
        test_context=test_context,
        short_response_instruction=short_response_instruction,
        knowledge_context=knowledge_context,
        global_memory_context=global_memory_context,
    )
    return "\n\n".join((*runtime_parts, *scenario_parts, *request_parts))


async def generate_response(
    user_id: int,
    user_prompt: str,
    bot=None,
    *,
    response_capture: dict | None = None,
    topic_id_override: int | None | object = _CURRENT_AI_CONTEXT,
    dialogue_id_override: int | None = None,
    exclude_message_id: int | None = None,
    track_user_activity: bool = True,
    activity_tracker: ActivityTracker | None = None,
    minutes_since_last_visit: int | None = None,
    minutes_since_last_message: int | None = None,
    preferred_response_locale: str | None = None,
) -> str:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            return "Ошибка: Пользователь не найден."

        user_name = user.name if user.name else "Незнакомец"
        user_gender = user.gender if user.gender else "unknown"

    return await get_ai_response(
        user_id,
        user_prompt,
        user_name,
        user_gender,
        bot=bot,
        response_capture=response_capture,
        topic_id_override=topic_id_override,
        dialogue_id_override=dialogue_id_override,
        exclude_message_id=exclude_message_id,
        track_user_activity=track_user_activity,
        activity_tracker=activity_tracker,
        minutes_since_last_visit=minutes_since_last_visit,
        minutes_since_last_message=minutes_since_last_message,
        preferred_response_locale=preferred_response_locale,
    )


async def _call_gemini_api(
    api_key: str,
    model: str,
    history: list,
    context: str,
    system_prompt: str,
    temperature: float = 0.7,
    timeout: float = 60.0,
    request_capture: dict | None = None,
    runtime_context: str = "",
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    max_output_tokens: int | None = None,
) -> str:
    import httpx
    try:
        if not api_key:
            raise AIServiceError("API key for Gemini is not configured.")
        target_model = model if model else "gemini-3.7-flash"
        ensure_model_available(PROVIDER_GEMINI, target_model)

        transport = _build_async_transport_from_env("GEMINI_PROXY")
        layout = _coerce_request_layout(
            request_layout,
            history=history,
            system_prompt=system_prompt,
            context=context,
            runtime_context=runtime_context,
        )
        contents = build_gemini_contents(layout)
        if not contents or contents[-1]['role'] != 'user':
            raise AIResponseError("Gemini request history must end with a user message")

        generation_config: dict[str, Any] = {
            "maxOutputTokens": max_output_tokens or GEMINI_CHAT_MAX_TOKENS,
        }
        if not (target_model.startswith("gemini-3.7") or target_model.startswith("gemini-3.6")):
            generation_config["temperature"] = temperature

        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": build_gemini_system_parts(layout) or [{"text": ""}],
            },
            "generationConfig": generation_config,
        }
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"
        url = f"{endpoint}?key={api_key}"
        _capture_ai_request(request_capture, provider="Gemini", endpoint=endpoint, payload=payload)
        if activity_tracker is not None:
            try:
                await activity_tracker.mark_outbound_attempt_once()
            except Exception as act_err:
                logging.warning("Failed to mark activity before outbound call: %s", act_err)
        async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=timeout) as client:
            response = await client.post(url, json=payload, headers={'Content-Type': 'application/json'})
            if response.status_code != 200:
                error_data = response.json()
                error_payload = error_data.get('error', {}) if isinstance(error_data, dict) else {}
                error_msg = _nonempty_provider_reason(
                    error_payload.get('message') if isinstance(error_payload, dict) else None,
                    _nonempty_provider_reason(
                        getattr(response, 'text', None),
                        f"HTTP {response.status_code} без описания",
                    ),
                )
                if "location" in error_msg.lower():
                    raise InsufficientBalanceError(f"Geo-Block: {error_msg}")
                raise AIServiceError(f"Ошибка API Gemini: {error_msg}")
            data = response.json()
            candidates = data.get('candidates', [])
            if not candidates:
                raise AIResponseError("Gemini returned an empty response or blocked content")
            return _validate_text_response(
                candidates[0]['content']['parts'][0]['text'],
                provider="Gemini",
            )
    except Exception as e:
        if any(word in str(e).lower() for word in ["billing", "quota", "location", "geo-block"]):
            raise InsufficientBalanceError(f"Gemini API Error: {exception_summary(e)}") from e
        raise AIServiceError(f"Ошибка при обращении к Gemini: {exception_summary(e)}") from e


async def _call_kie_chat(
    api_key: str,
    base_url: str,
    model: str,
    history: list,
    context: str,
    system_prompt: str,
    temperature: float = 0.7,
    request_capture: dict | None = None,
    runtime_context: str = "",
    *,
    timeout: float = 120.0,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    max_output_tokens: int | None = None,
) -> str:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_KIE, target_model, channel="chat")
    if not api_key:
        raise AIServiceError("API key for KIE is not configured.")
    try:
        layout = _coerce_request_layout(
            request_layout,
            history=history,
            system_prompt=system_prompt,
            context=context,
            runtime_context=runtime_context,
        )

        request = build_kie_chat_request(
            api_key,
            base_url,
            target_model,
            request_layout=layout,
            temperature=temperature,
            max_output_tokens=max_output_tokens or 4096,
        )
        _capture_ai_request(
            request_capture,
            provider="KIE",
            endpoint=request.endpoint,
            payload=request.payload,
        )
        if activity_tracker is not None:
            try:
                await activity_tracker.mark_outbound_attempt_once()
            except Exception as act_err:
                logging.warning("Failed to mark activity before outbound call: %s", act_err)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
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
            raise AIResponseError("KIE chat returned empty content")
        return _validate_text_response(text, provider="KIE")
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE chat error", exc_info=e)
        raise AIServiceError(f"Ошибка при обращении к KIE Chat API: {exception_summary(e)}") from e


async def _upload_file_to_kie(
    api_key: str,
    upload_base_url: str,
    file_bytes: bytes,
    filename: str,
    upload_path: str,
    *,
    timeout: float | httpx.Timeout = 120.0,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    url = f"{upload_base_url}/api/file-stream-upload"
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    data = {"uploadPath": upload_path, "fileName": filename}
    headers = {"Authorization": f"Bearer {api_key}"}

    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            logging.warning("Failed to mark activity before outbound KIE upload: %s", act_err)

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
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
            raise err

        data_payload = _validate_kie_json_response(
            response.status_code,
            payload,
            context="KIE upload failed",
        )
        file_url = data_payload.get("downloadUrl") or data_payload.get("fileUrl")
        if not file_url:
            raise AIResponseError(f"KIE upload returned no file URL: {payload}")
        return file_url
    except (AIServiceError, AIResponseError, InsufficientBalanceError):
        raise
    except Exception as e:
        logging.error("KIE upload error", exc_info=e)
        raise AIServiceError(f"Ошибка загрузки файла в KIE: {exception_summary(e)}") from e



async def _call_kie_multimodal(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: list,
    temperature: float = 0.7,
    history: list | None = None,
    request_capture: dict | None = None,
    channel: str = "chat",
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
) -> str:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_KIE, target_model, channel=channel)
    try:
        layout = _coerce_request_layout(
            request_layout,
            history=history,
            system_prompt=system_prompt,
            current_user_content=user_content,
        ).with_current_user_content(user_content)
        payload = {
            "model": target_model,
            "messages": build_openai_chat_messages(layout),
            "max_tokens": 4096,
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
                logging.warning("Failed to mark activity before outbound KIE multimodal call: %s", act_err)
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(
                endpoint,
                headers=headers,
                json=payload,
            )
        response_payload = _validate_kie_json_response(
            response.status_code,
            response.json(),
            context="Ошибка обращения к KIE multimodal API",
        )
        text = _extract_kie_chat_text(response_payload)
        if not text:
            raise AIResponseError("KIE multimodal request returned empty content")
        return text
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE multimodal error", exc_info=e)
        raise AIServiceError(f"Ошибка обращения к KIE multimodal API: {exception_summary(e)}") from e


async def _create_kie_task(api_key: str, base_url: str, model: str, input_payload: dict) -> str:
    url = f"{base_url}/api/v1/jobs/createTask"
    payload = {"model": model, "input": input_payload}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(url, headers=headers, json=payload)
        data = response.json()
        data_payload = _validate_kie_json_response(
            response.status_code,
            data,
            context="KIE task creation failed",
        )
        task_id = data_payload.get("taskId")
        if not task_id:
            raise AIResponseError(f"KIE task creation returned no taskId: {data}")
        return task_id
    except (AIServiceError, AIResponseError):
        raise
    except Exception as e:
        logging.error("KIE create task error", exc_info=e)
        raise AIServiceError(f"Ошибка создания задачи KIE: {exception_summary(e)}") from e


def _extract_kie_task_result(task_payload: dict) -> dict:
    response_payload = task_payload.get("response")
    if isinstance(response_payload, dict) and response_payload:
        return response_payload
    result_json = task_payload.get("resultJson")
    if isinstance(result_json, str) and result_json:
        try:
            return json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise AIResponseError(f"Cannot decode KIE resultJson: {exception_summary(exc)}") from exc
    if isinstance(result_json, dict):
        return result_json
    return {}


async def _poll_kie_task(api_key: str, base_url: str, task_id: str, *, timeout_sec: int = 180) -> dict:
    url = f"{base_url}/api/v1/jobs/recordInfo"
    headers = {"Authorization": f"Bearer {api_key}"}
    delay = 2.0
    deadline = asyncio.get_running_loop().time() + timeout_sec

    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        while True:
            response = await client.get(url, headers=headers, params={"taskId": task_id})
            payload = _validate_kie_json_response(
                response.status_code,
                response.json(),
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


async def _download_binary_file(url: str) -> bytes:
    import httpx

    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.get(url)
    if response.status_code != 200:
        raise AIServiceError(f"Result download failed: status={response.status_code} url={url}")
    return response.content


async def _get_kie_download_url(api_key: str, base_url: str, url: str) -> str:
    import httpx

    endpoint = f"{base_url}/api/v1/common/download-url"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"url": url}

    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            return url
        data = response.json()
        return data.get("data") or url
    except Exception:
        return url


async def get_kie_remaining_credits(api_key: str, base_url: str) -> float:
    endpoint = f"{base_url}/api/v1/chat/credit"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        for attempt in range(1, 4):
            try:
                response = await client.get(endpoint, headers=headers)
                if response.status_code != 200:
                    raise AIServiceError(
                        f"KIE credits check failed: status={response.status_code} body={response.text}"
                    )

                payload = response.json()
                data = payload.get("data")
                if isinstance(data, (int, float, str)):
                    return float(data)
                if data is None:
                    raise AIResponseError(f"KIE credits response has no data field: {payload}")
                for key in ("remainingCredits", "remaining_credits", "credits", "balance", "creditBalance"):
                    value = data.get(key)
                    if value is not None:
                        return float(value)
                raise AIResponseError(f"KIE credits response has no remaining credits field: {payload}")
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= 3:
                    logging.error("KIE credits check transport error", exc_info=exc)
                    raise AIServiceError(
                        f"Ошибка проверки остатка кредитов KIE: {exception_summary(exc)} после 3 попыток"
                    ) from exc
                logging.warning(
                    "Retrying KIE credits check attempt=%s/3 error=%s",
                    attempt + 1,
                    exception_summary(exc),
                )
                await asyncio.sleep(attempt)
            except (AIServiceError, AIResponseError):
                raise
            except Exception as exc:
                logging.error("KIE credits check error", exc_info=exc)
                raise AIServiceError(
                    f"Ошибка проверки остатка кредитов KIE: {exception_summary(exc)}"
                ) from exc


async def _call_claude_api(
    api_key: str,
    model: str,
    history: list,
    context: str,
    system_prompt: str,
    temperature: float = 0.7,
    timeout: float = 60.0,
    request_capture: dict | None = None,
    runtime_context: str = "",
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    max_output_tokens: int | None = None,
):
    try:
        if not api_key:
            raise AIServiceError("API key for Claude is not configured.")
        target_model = model if model else "claude-sonnet-5"
        ensure_model_available(PROVIDER_CLAUDE, target_model)
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)

        layout = _coerce_request_layout(
            request_layout,
            history=history,
            system_prompt=system_prompt,
            context=context,
            runtime_context=runtime_context,
        )
        claude_history = [
            {"role": message.role, "content": message.content}
            for message in layout.history
        ]
        if layout.current_user_content is not None:
            claude_history.append({"role": "user", "content": layout.current_user_content})

        payload: dict[str, Any] = {
            "model": target_model,
            "max_tokens": max_output_tokens or CLAUDE_CHAT_MAX_TOKENS,
            "system": build_anthropic_system(layout),
            "messages": claude_history,
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
                logging.warning("Failed to mark activity before outbound call: %s", act_err)
        message = await client.messages.create(
            **payload,
        )
        return message.content[0].text
    except anthropic.AuthenticationError as e:
        raise InsufficientBalanceError(f"Claude API Error: {exception_summary(e)}") from e
    except Exception as e:
        error_text = str(e).lower()
        if any(marker in error_text for marker in ["credit balance", "billing", "quota", "purchase credits", "insufficient"]):
            raise InsufficientBalanceError(f"Claude API Error: {exception_summary(e)}") from e
        logging.error(f"Claude API error: {e}")
        raise AIServiceError(f"Ошибка при обращении к Claude API: {exception_summary(e)}") from e


async def _call_claude_vision(
    api_key: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    history: list = None,
    temperature: float = 0.7,
    request_context: str = "",
    request_capture: dict | None = None,
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    effective_user_prompt: str | None = None,
    stage_budget: float = 25.0,
) -> str:
    try:
        target_model = model if model else "claude-sonnet-5"
        ensure_model_available(PROVIDER_CLAUDE, target_model, channel="vision")
        max_tokens = get_provider_vision_max_tokens(PROVIDER_CLAUDE)
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=stage_budget, max_retries=0)
        user_instruction = effective_user_prompt or _extract_vision_user_prompt(
            request_layout=request_layout,
            prompt=prompt,
        )
        layout = _coerce_request_layout(
            request_layout,
            history=history,
            system_prompt=prompt,
            runtime_context=request_context,
            current_user_content=None,
        )
        vision_content = [
            {"type": "text", "text": user_instruction},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _guess_image_media_type(image_bytes),
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                },
            },
        ]
        layout = layout.with_current_user_content(vision_content)
        claude_history = [
            {"role": message.role, "content": message.content}
            for message in layout.history
        ]
        claude_history.append({"role": "user", "content": vision_content})
        payload: dict[str, Any] = {
            "model": target_model,
            "max_tokens": max_tokens,
            "system": build_anthropic_system(layout),
            "messages": claude_history,
        }
        if not should_omit_claude_sampling(target_model):
            payload["temperature"] = temperature

        _capture_ai_request(request_capture, provider="Claude", endpoint="https://api.anthropic.com/v1/messages", payload=payload)
        if activity_tracker is not None:
            try:
                await activity_tracker.mark_outbound_attempt_once()
            except Exception as act_err:
                logging.warning("Failed to mark activity before outbound Claude vision call: %s", act_err)
        message = await client.messages.create(**payload)

        stop_reason = getattr(message, "stop_reason", None)
        usage_dict = None
        usage_obj = getattr(message, "usage", None)
        if usage_obj is not None:
            try:
                usage_dict = usage_obj.model_dump()
            except Exception:
                usage_dict = getattr(usage_obj, "__dict__", str(usage_obj))

        text_parts = []
        if getattr(message, "content", None):
            for item in message.content:
                if getattr(item, "type", None) == "text" and getattr(item, "text", None):
                    text_parts.append(item.text)
        result = "\n".join(text_parts).strip()

        safe_resp_payload = None
        try:
            import json
            safe_resp = {
                "id": getattr(message, "id", None),
                "model": getattr(message, "model", None),
                "stop_reason": stop_reason,
                "usage": usage_dict,
                "content": [{"type": "text", "text": result}] if result else [],
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

        if stop_reason in {"refusal", "safety"}:
            raise attach_error_metadata(
                AIServiceError(f"Claude Vision rejection: {stop_reason}"),
                classification="provider_rejection",
                finish_reason=stop_reason,
                http_status=200,
                diagnostics={"usage": usage_dict} if usage_dict else None,
                provider_response_payload=safe_resp_payload,
            )

        if stop_reason == "max_tokens":
            raise attach_error_metadata(
                AIResponseError("Claude Vision output budget exhausted"),
                classification="output_budget_exhausted",
                finish_reason="max_tokens",
                http_status=200,
                diagnostics={"usage": usage_dict} if usage_dict else None,
                provider_response_payload=safe_resp_payload,
            )

        if not result:
            raise attach_error_metadata(
                AIResponseError("Claude vision returned empty content"),
                classification="empty_response",
                finish_reason=stop_reason,
                http_status=200,
                diagnostics={"usage": usage_dict} if usage_dict else None,
                provider_response_payload=safe_resp_payload,
            )

        return result
    except anthropic.AuthenticationError as e:
        raise InsufficientBalanceError(f"Claude Vision API Error: {exception_summary(e)}") from e
    except (AIServiceError, AIResponseError):
        raise
    except Exception as e:
        error_text = str(e).lower()
        if any(marker in error_text for marker in ["credit balance", "billing", "quota", "purchase credits", "insufficient"]):
            raise InsufficientBalanceError(f"Claude Vision API Error: {exception_summary(e)}") from e
        logging.error("Claude vision error", exc_info=e)
        raise AIServiceError(f"Ошибка анализа изображения (Claude): {exception_summary(e)}") from e



async def _call_deepseek_api(
    api_key: str,
    model: str,
    history: list,
    context: str,
    system_prompt: str,
    temperature: float = 0.7,
    use_proxy: bool = True,
    timeout: float = 60.0,
    request_capture: dict | None = None,
    runtime_context: str = "",
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    max_output_tokens: int | None = None,
    thinking_enabled: bool | None = None,
):
    client = None
    try:
        if not api_key:
            raise AIServiceError("API key for DeepSeek is not configured.")
        normalized_model = normalize_deepseek_model(model)
        ensure_model_available(PROVIDER_DEEPSEEK, normalized_model)

        if use_proxy:
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
            transport = _build_async_transport_from_env("DEEPSEEK_PROXY", use_proxy=True)
        else:
            base_url = "https://api.deepseek.com"
            transport = None
        import httpx
        timeout_sec = timeout
        http_client = httpx.AsyncClient(transport=transport, trust_env=False, timeout=timeout_sec)

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

        layout = _coerce_request_layout(
            request_layout,
            history=history,
            system_prompt=system_prompt,
            context=context,
            runtime_context=runtime_context,
        )

        payload = {
            "model": normalized_model,
            "messages": build_openai_chat_messages(layout),
            "max_tokens": max_output_tokens or DEEPSEEK_CHAT_MAX_TOKENS,
            "temperature": temperature,
        }
        if thinking_enabled is True:
            payload["extra_body"] = {"thinking": {"type": "enabled"}}
        elif thinking_enabled is False:
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
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
                logging.warning("Failed to mark activity before outbound call: %s", act_err)
        chat_completion = await client.chat.completions.create(
            **payload,
        )
        raw_payload_str = None
        try:
            if hasattr(chat_completion, "model_dump_json"):
                raw_payload_str = chat_completion.model_dump_json()
            elif hasattr(chat_completion, "to_dict"):
                raw_payload_str = json.dumps(chat_completion.to_dict(), default=str)
            elif isinstance(chat_completion, dict):
                raw_payload_str = json.dumps(chat_completion, default=str)
            else:
                raw_payload_str = str(chat_completion)
        except Exception:
            raw_payload_str = str(chat_completion)

        visible_content, diagnostics = inspect_deepseek_response(
            chat_completion,
            model=normalized_model,
            platform="telegram",
        )
        if diagnostics.output_budget_exhausted:
            logging.warning(
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
                f"Deepseek вернул пустой content (output budget exhausted: finish_reason={diagnostics.finish_reason}, reasoning_len={diagnostics.reasoning_content_length})"
            )
            err.http_status = 200
            err.finish_reason = diagnostics.finish_reason
            err.diagnostics = diagnostics
            err.provider_response_payload = raw_payload_str
            err.classification = "output_budget_exhausted"
            raise err

        err = AIResponseError("Deepseek вернул пустой content")
        err.http_status = 200
        err.finish_reason = diagnostics.finish_reason
        err.diagnostics = diagnostics
        err.provider_response_payload = raw_payload_str
        err.classification = "empty_response"
        raise err
    except Exception as e:
        if isinstance(e, AIServiceError):
            raise
        if hasattr(e, 'code') and e.code == 'insufficient_quota':
            raise InsufficientBalanceError(f"Deepseek API Error: {exception_summary(e)}") from e
        logging.error(f"Deepseek API error: {e}")
        service_err = AIServiceError(f"Ошибка при обращении к Deepseek API: {exception_summary(e)}")
        if hasattr(e, 'status_code'):
            service_err.http_status = e.status_code
        elif hasattr(e, 'status'):
            service_err.http_status = e.status
        raise service_err from e
    finally:
        if client is not None:
            await client.close()




async def _call_openai_transcribe(api_key: str, file_bytes: bytes, filename: str) -> str:
    try:
        ensure_model_available(PROVIDER_OPENAI, "whisper-1", channel="transcription")
        client = AsyncOpenAI(api_key=api_key)

        transcription = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, file_bytes)
        )
        return transcription.text
    except AuthenticationError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Invalid API Key. {exception_summary(e)}") from e
    except RateLimitError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Rate limit or quota exceeded. {exception_summary(e)}") from e
    except BadRequestError as e:
        if "billing" in str(e) or "quota" in str(e).lower():
            raise InsufficientBalanceError(f"OpenAI API Error: Billing issue or insufficient quota. {exception_summary(e)}") from e
        logging.error(f"OpenAI API error: {e}")
        raise AIServiceError(f"Ошибка при транскрибации (OpenAI API): {exception_summary(e)}") from e
    except Exception as e:
        logging.error(f"OpenAI API transcription error: {e}")
        raise AIServiceError(f"Ошибка при транскрибации: {exception_summary(e)}") from e


async def transcribe_voice_message(file_bytes: bytes, filename: str) -> str:
    async with async_session_maker() as session:
        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            raise AIServiceError("Конфигурация ИИ не найдена.")

        provider = ai_config.transcription_provider

        if provider == "OpenAI":
            api_key = ai_config.openai_api_key
            if not api_key:
                raise AIServiceError(f"API ключ для {provider} (для транскрибации) не установлен администратором.")
            response_text = await _call_openai_transcribe(api_key, file_bytes, filename)

        elif provider == "Gemini":
            api_key = ai_config.gemini_api_key
            model = get_default_model(PROVIDER_GEMINI, channel="transcription")
            if not api_key:
                raise AIServiceError(f"API ключ для {provider} (для транскрибации) не установлен администратором.")
            response_text = await _call_gemini_transcribe(api_key, model, file_bytes, filename)
        elif provider == "KIE":
            api_key = ai_config.kie_api_key
            model = getattr(ai_config, "kie_transcription_model", None) or getattr(ai_config, "kie_model", None) or DEFAULT_KIE_TRANSCRIPTION_MODEL
            if not api_key:
                raise AIServiceError(f"API ключ для {provider} (для транскрибации) не установлен администратором.")
            try:
                response_text = await _call_kie_transcribe(
                    api_key,
                    _get_kie_base_url(ai_config),
                    _get_kie_upload_base_url(ai_config),
                    model,
                    file_bytes,
                    filename,
                )
            except (InsufficientBalanceError, AIServiceError) as kie_exc:
                openai_api_key = ai_config.openai_api_key
                if not openai_api_key:
                    raise

                logging.warning(
                    "KIE transcription failed, falling back to OpenAI Whisper: model=%s error=%s",
                    model,
                    kie_exc,
                )
                try:
                    response_text = await _call_openai_transcribe(openai_api_key, file_bytes, filename)
                except (InsufficientBalanceError, AIServiceError) as fallback_exc:
                    fallback_exc.provider_attempts = (
                        {
                            "provider": "KIE",
                            "model": model,
                            "error": exception_summary(kie_exc),
                        },
                        {
                            "provider": "OpenAI",
                            "model": "whisper-1",
                            # The active KIE exception is an implicit context here;
                            # this attempt must describe OpenAI itself.
                            "error": exception_summary(fallback_exc, include_context=False),
                        },
                    )
                    raise fallback_exc from kie_exc

        else:
            raise AIServiceError(f"Неизвестный провайдер транскрибации: {provider}")

        return response_text


async def _call_openai_api(
    api_key: str,
    model: str,
    history: list,
    context: str,
    system_prompt: str,
    temperature: float = 0.7,
    timeout: float = 60.0,
    request_capture: dict | None = None,
    runtime_context: str = "",
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    max_completion_tokens: int = 4096,
):
    try:
        if not api_key:
            raise AIServiceError("API key for OpenAI is not configured.")
        target_model = model if model else "gpt-5.6-terra"
        ensure_model_available(PROVIDER_OPENAI, target_model)
        client = AsyncOpenAI(api_key=api_key, timeout=timeout)

        layout = _coerce_request_layout(
            request_layout,
            history=history,
            system_prompt=system_prompt,
            context=context,
            runtime_context=runtime_context,
        )

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": build_openai_chat_messages(layout),
            "max_completion_tokens": max_completion_tokens,
        }
        if not target_model.startswith("gpt-5.6"):
            payload["temperature"] = temperature

        _capture_ai_request(
            request_capture,
            provider="OpenAI",
            endpoint="https://api.openai.com/v1/chat/completions",
            payload=payload,
        )
        if activity_tracker is not None:
            try:
                await activity_tracker.mark_outbound_attempt_once()
            except Exception as act_err:
                logging.warning("Failed to mark activity before outbound call: %s", act_err)
        chat_completion = await client.chat.completions.create(
            **payload,
        )
        return _extract_openai_chat_text(chat_completion, provider="OpenAI")
    except AuthenticationError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Invalid API Key. {exception_summary(e)}") from e
    except RateLimitError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Rate limit or quota exceeded. {exception_summary(e)}") from e
    except BadRequestError as e:
        if "billing" in str(e) or "quota" in str(e).lower():
            raise InsufficientBalanceError(f"OpenAI API Error: Billing issue or insufficient quota. {exception_summary(e)}") from e
        logging.error(f"OpenAI API error: {e}")
        raise AIServiceError(f"Ошибка при обращении к OpenAI API: {exception_summary(e)}") from e
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        raise AIServiceError(f"Ошибка при обращении к OpenAI API: {exception_summary(e)}") from e


async def get_ai_response(
    user_id: int,
    user_prompt: str,
    user_name: str,
    user_gender: str,
    bot=None,
    *,
    topic_id_override: int | None | object = _CURRENT_AI_CONTEXT,
    dialogue_id_override: int | None = None,
    include_test_context: bool = True,
    persist_service_data: bool = True,
    request_type: str = "chat",
    response_capture: dict | None = None,
    exclude_message_id: int | None = None,
    track_user_activity: bool = True,
    activity_tracker: ActivityTracker | None = None,
    minutes_since_last_visit: int | None = None,
    minutes_since_last_message: int | None = None,
    preferred_response_locale: str | None = None,
) -> str:
    async with async_session_maker() as session:
        user_result = await session.execute(
            select(User).options(
                selectinload(User.current_topic).selectinload(Topic.knowledge_base_files),
                selectinload(User.subscription),
            ).where(
                User.id == user_id)
        )
        user = user_result.scalar_one_or_none()

        if not user:
            return "❌ Ошибка: Пользователь не найден."

        active_topic_id = (
            user.current_topic_id
            if topic_id_override is _CURRENT_AI_CONTEXT
            else topic_id_override
        )
        if active_topic_id == 0:
            active_topic_id = None
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
            return "❌ Ошибка: Конфигурация ИИ не найдена."

        temperature = _resolve_temperature(ai_config)

        _, media_files = await load_available_media(session, active_topic_id)
        available_media_text, media_instruction_block = format_available_media_text(media_files, active_topic_id)

        provider = ai_config.provider
        provider_key = provider.strip().lower() if provider else ""

        api_key = _normalize_config_value(getattr(ai_config, f"{provider_key}_api_key", None))
        if provider_key in ['anthropic', 'claude'] and not api_key:
            api_key = _normalize_config_value(ai_config.claude_api_key)

        primary_config_error = None
        if not api_key:
            primary_config_error = AIServiceError(
                f"API key for primary AI provider '{provider}' is not configured"
            )

        model = _normalize_config_value(getattr(ai_config, f"{provider_key}_model", None))
        if provider_key in ['anthropic', 'claude'] and not model:
            model = _normalize_config_value(ai_config.claude_model)

        system_prompt_text = _load_configured_system_prompt(
            ai_config,
            active_topic.system_prompt if active_topic else None
        )

        subscription_config = await session.get(SubscriptionConfig, 1)

        # Only configured/topic prompt text belongs in the stable cache prefix.
        formatted_body = neutralize_stable_prompt(system_prompt_text)

        shared_prompt_block = (getattr(ai_config, 'shared_prompt_block', "") or "").strip()
        short_response_instruction = ""
        if getattr(user, 'response_length', 'normal') == 'short':
            short_response_instruction = DEFAULT_SHORT_RESPONSE_INSTRUCTION

        relevant_chunks = []
        if active_topic:
            doc_ids = [f.id for f in active_topic.knowledge_base_files]
            if doc_ids:
                relevant_chunks = await search_relevant_chunks(user_prompt, n_results=3, document_ids=doc_ids)
        else:
            # Exclude prompt templates from general KB so they do not override the active system prompt.
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

        request_time = (
            activity_tracker.request_time
            if activity_tracker is not None
            else datetime.utcnow()
        )
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
            stable_system_prompt=formatted_body,
            minutes_since_last_visit=minutes_since_last_visit,
            minutes_since_last_message=minutes_since_last_message,
            test_context="",
            short_response_instruction=short_response_instruction,
            knowledge_context=context,
            available_media_text=available_media_text,
            media_instruction_block=media_instruction_block,
            subscription_config=subscription_config,
            memory_mode=get_memory_mode(ai_config),
            service_capabilities=TELEGRAM_CAPABILITIES,
            preferred_response_locale=(
                preferred_response_locale
                if preferred_response_locale is not None
                else await resolve_user_effective_locale(
                    session,
                    user,
                    platform="max" if user.id >= 100_000_000_000 else "telegram",
                )
            ),
        )

        request_group_id = uuid.uuid4().hex[:12]
        primary_capture: dict = {}
        fallback_capture: dict = {}

        async def _dispatch_call(p_key, p_api_key, p_model, capture_dict: dict):
            use_proxy = getattr(ai_config, 'use_proxy', True)
            timeout = float(getattr(ai_config, "fallback_timeout", 60))
            normalized_provider = {
                "openai": PROVIDER_OPENAI,
                "anthropic": PROVIDER_CLAUDE,
                "claude": PROVIDER_CLAUDE,
                "gemini": PROVIDER_GEMINI,
                "kie": PROVIDER_KIE,
                "deepseek": PROVIDER_DEEPSEEK,
                "xai": "xAI",
            }.get(str(p_key).lower(), p_key)
            output_tokens = effective_chat_output_tokens(
                normalized_provider,
                p_model,
                getattr(ai_config, "max_output_tokens", None),
            )
            thinking_enabled = (
                getattr(ai_config, "deepseek_thinking_enabled", None)
                if str(p_key).lower() == "deepseek"
                else None
            )

            if not str(p_model).startswith(("primary-", "fallback-", "mock-", "test-", "dummy-", "gpt-5.6-turbo")) and not str(p_model).endswith(("-telegram", "-max")):
                try:
                    if p_key == 'openai':
                        ensure_model_available(PROVIDER_OPENAI, p_model)
                    elif p_key in ['anthropic', 'claude']:
                        ensure_model_available(PROVIDER_CLAUDE, p_model)
                    elif p_key == 'gemini':
                        ensure_model_available(PROVIDER_GEMINI, p_model)
                    elif p_key == 'kie':
                        ensure_model_available(PROVIDER_KIE, p_model, channel="chat")
                    elif p_key == 'deepseek':
                        ensure_model_available(PROVIDER_DEEPSEEK, p_model)
                    elif p_key == 'xai':
                        ensure_model_available(PROVIDER_OPENAI, p_model)
                    else:
                        raise AIServiceError(f"Неизвестный провайдер ИИ: '{p_key}'")
                except (AIServiceError, Exception) as e:
                    raise AIServiceError(f"Ошибка проверки модели ИИ: {e}") from e
            elif p_key not in {'openai', 'anthropic', 'claude', 'gemini', 'kie', 'deepseek', 'xai'}:
                raise AIServiceError(f"Неизвестный провайдер ИИ: '{p_key}'")


            async def _invoke():
                if p_key == 'openai':
                    return await _call_openai_api(
                        p_api_key,
                        p_model,
                        list(request_layout.history),
                        "",
                        formatted_body,
                        temperature,
                        timeout=timeout,
                        request_capture=capture_dict,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        max_completion_tokens=output_tokens,
                    )
                elif p_key in ['anthropic', 'claude']:
                    return await _call_claude_api(p_api_key, p_model, list(request_layout.history), "", formatted_body, temperature, timeout=timeout, request_capture=capture_dict, request_layout=request_layout, activity_tracker=activity_tracker, max_output_tokens=output_tokens)
                elif p_key == 'gemini':
                    return await _call_gemini_api(p_api_key, p_model, list(request_layout.history), "", formatted_body, temperature, timeout=timeout, request_capture=capture_dict, request_layout=request_layout, activity_tracker=activity_tracker, max_output_tokens=output_tokens)
                elif p_key == 'kie':
                    return await _call_kie_chat(p_api_key, _get_kie_base_url(ai_config), p_model, list(request_layout.history), "", formatted_body, temperature, timeout=timeout, request_capture=capture_dict, request_layout=request_layout, activity_tracker=activity_tracker, max_output_tokens=output_tokens)
                elif p_key == 'deepseek':
                    return await _call_deepseek_api(p_api_key, p_model, list(request_layout.history), "", formatted_body, temperature, use_proxy=use_proxy, timeout=timeout, request_capture=capture_dict, request_layout=request_layout, activity_tracker=activity_tracker, max_output_tokens=output_tokens, thinking_enabled=thinking_enabled)
                elif p_key == 'xai':
                    return await _call_openai_api(
                        p_api_key,
                        p_model,
                        list(request_layout.history),
                        "",
                        formatted_body,
                        temperature,
                        timeout=timeout,
                        request_capture=capture_dict,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        max_completion_tokens=output_tokens,
                    )
                else:
                    raise AIServiceError(f"Неизвестный провайдер ИИ: '{p_key}'")

            try:
                response_text = await asyncio.wait_for(_invoke(), timeout=timeout)
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError) as timeout_exc:
                err = AIServiceError(f"AI provider {p_key} timed out after {timeout}s")
                err.classification = "timeout"
                raise err from timeout_exc

            return _validate_text_response(response_text, provider=p_key)

        successful_log_id: int | None = None
        actual_provider = provider
        actual_model = model
        latency_ms = 0

        primary_start = time.monotonic()
        primary_succeeded = False
        try:
            if primary_config_error is not None:
                raise primary_config_error
            response_text = await _dispatch_call(provider_key, api_key, model, primary_capture)
            primary_latency = int((time.monotonic() - primary_start) * 1000)
            primary_prov, primary_mod = extract_effective_provider_and_model(
                primary_capture,
                default_provider=provider,
                default_model=model,
            )
            visible_text, service_blocks, invalid_data_blocks = extract_service_data(response_text)
            if response_capture is not None:
                response_capture.clear()
                response_capture.update({
                    "raw_response": response_text,
                    "visible_text": visible_text,
                })
            if invalid_data_blocks:
                logging.warning("AI returned %s invalid DATA block(s) for user %s", invalid_data_blocks, user_id)

            primary_log_id = await record_ai_attempt_log(
                session,
                user_id=user_id,
                platform="telegram",
                dialogue_id=active_dialogue_id,
                topic_id=active_topic_id,
                topic_name=active_topic.name if active_topic else None,
                request_type=request_type or "chat",
                provider=primary_prov,
                model=primary_mod,
                prompt_summary=user_prompt if user_prompt else None,
                request_capture=primary_capture,
                raw_response=response_text,
                clean_text=visible_text,
                latency_ms=primary_latency,
                status="success",
                request_group_id=request_group_id,
                attempt_no=1,
                attempt_role="primary",
            )
            primary_succeeded = True
            successful_log_id = primary_log_id
            actual_provider = primary_prov
            actual_model = primary_mod
            latency_ms = primary_latency
        except (AIServiceError, Exception) as primary_err:
            primary_latency = int((time.monotonic() - primary_start) * 1000)
            primary_prov, primary_mod = extract_effective_provider_and_model(
                primary_capture,
                default_provider=provider,
                default_model=model,
            )
            err_meta = extract_error_metadata(primary_err, provider=primary_prov)
            primary_log_id = await record_ai_attempt_log(
                session,
                user_id=user_id,
                platform="telegram",
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
                    bot=bot,
                    platform="telegram",
                    user_id=user_id,
                    provider=primary_prov,
                    model=primary_mod,
                    finish_reason=getattr(diag, "finish_reason", "length") if diag else "length",
                    visible_content_length=getattr(diag, "visible_content_length", 0) if diag else 0,
                    reasoning_content_length=getattr(diag, "reasoning_content_length", 0) if diag else 0,
                    max_tokens=effective_chat_output_tokens(
                        primary_prov,
                        primary_mod,
                        getattr(ai_config, "max_output_tokens", None),
                    ),
                    ai_log_id=primary_log_id,
                )

            allow_fallback = getattr(ai_config, 'allow_fallback', False)
            fb_provider = getattr(ai_config, 'fallback_provider', None)
            fb_model = getattr(ai_config, 'fallback_model', None)
            if not allow_fallback or not fb_provider or not fb_model:
                await send_terminal_ai_failure_alert(
                    bot=bot,
                    platform="telegram",
                    user_id=user_id,
                    primary_provider=primary_prov,
                    primary_model=primary_mod,
                    exception=primary_err,
                    classification=err_meta["error_classification"],
                    ai_log_ids=[primary_log_id] if primary_log_id else None,
                )
                if allow_fallback:
                    fallback_config_error = AIServiceError(
                        "Fallback AI provider configuration is incomplete: provider and model are required"
                    )
                    service_err = AIServiceError(
                        "Основной AI-провайдер не сработал, резервный провайдер не настроен"
                    )
                    service_err.ai_outcome = "PRIMARY_FAILED + FALLBACK_CONFIGURATION_ERROR"
                    service_err.provider_attempts = (
                        _build_ai_attempt(primary_prov, primary_mod, primary_err),
                        _build_ai_attempt(
                            fb_provider,
                            fb_model,
                            fallback_config_error,
                            status="CONFIGURATION_ERROR",
                            classification="configuration",
                        ),
                    )
                    raise service_err from primary_err
                if isinstance(primary_err, AIServiceError):
                    raise
                raise AIServiceError(f"Ошибка при обращении к AI-провайдеру: {primary_err}") from primary_err

            fb_key = fb_provider.strip().lower()
            fb_api_key = _normalize_config_value(getattr(ai_config, f"{fb_key}_api_key", None))
            if fb_key in ['anthropic', 'claude'] and not fb_api_key:
                fb_api_key = _normalize_config_value(ai_config.claude_api_key)
            if not fb_api_key:
                await send_terminal_ai_failure_alert(
                    bot=bot,
                    platform="telegram",
                    user_id=user_id,
                    primary_provider=primary_prov,
                    primary_model=primary_mod,
                    fallback_provider=fb_provider,
                    fallback_model=fb_model,
                    exception=primary_err,
                    classification="configuration",
                    ai_log_ids=[primary_log_id] if primary_log_id else None,
                )
                fallback_config_error = AIServiceError(
                    f"API key for fallback AI provider '{fb_provider}' is not configured"
                )
                logging.error(
                    "Fallback provider '%s' has no API key configured",
                    fb_provider,
                )
                service_err = AIServiceError(
                    "Основной AI-провайдер не сработал, резервный провайдер не настроен"
                )
                service_err.ai_outcome = "PRIMARY_FAILED + FALLBACK_CONFIGURATION_ERROR"
                service_err.provider_attempts = (
                    _build_ai_attempt(primary_prov, primary_mod, primary_err),
                    _build_ai_attempt(
                        fb_provider,
                        fb_model,
                        fallback_config_error,
                        status="CONFIGURATION_ERROR",
                        classification="configuration",
                    ),
                )
                raise service_err from primary_err

            logging.warning(
                f"Primary provider '{primary_prov}' failed ({primary_err}), "
                f"falling back to '{fb_provider}' / '{fb_model}'"
            )
            fallback_start = time.monotonic()
            try:
                response_text = await _dispatch_call(fb_key, fb_api_key, fb_model, fallback_capture)
                fallback_latency = int((time.monotonic() - fallback_start) * 1000)
                fb_prov, fb_mod = extract_effective_provider_and_model(
                    fallback_capture,
                    default_provider=fb_provider,
                    default_model=fb_model,
                )
                visible_text, service_blocks, invalid_data_blocks = extract_service_data(response_text)
                if response_capture is not None:
                    response_capture.clear()
                    response_capture.update({
                        "raw_response": response_text,
                        "visible_text": visible_text,
                    })
                if invalid_data_blocks:
                    logging.warning("AI returned %s invalid DATA block(s) for user %s", invalid_data_blocks, user_id)

                fb_log_id = await record_ai_attempt_log(
                    session,
                    user_id=user_id,
                    platform="telegram",
                    dialogue_id=active_dialogue_id,
                    topic_id=active_topic_id,
                    topic_name=active_topic.name if active_topic else None,
                    request_type=request_type or "chat",
                    provider=fb_prov,
                    model=fb_mod,
                    prompt_summary=user_prompt if user_prompt else None,
                    request_capture=fallback_capture,
                    raw_response=response_text,
                    clean_text=visible_text,
                    latency_ms=fallback_latency,
                    status="success",
                    request_group_id=request_group_id,
                    attempt_no=2,
                    attempt_role="fallback",
                )
                successful_log_id = fb_log_id
                actual_provider = fb_prov
                actual_model = fb_mod
                latency_ms = fallback_latency

                await _notify_ai_fallback_used(
                    bot,
                    user=user,
                    primary_provider=primary_prov,
                    primary_model=primary_mod,
                    fallback_provider=fb_prov,
                    fallback_model=fb_mod,
                    error=primary_err,
                )
            except Exception as fb_err:
                fallback_latency = int((time.monotonic() - fallback_start) * 1000)
                fb_prov, fb_mod = extract_effective_provider_and_model(
                    fallback_capture,
                    default_provider=fb_provider,
                    default_model=fb_model,
                )
                fb_err_meta = extract_error_metadata(fb_err, provider=fb_prov)
                fb_log_id = await record_ai_attempt_log(
                    session,
                    user_id=user_id,
                    platform="telegram",
                    dialogue_id=active_dialogue_id,
                    topic_id=active_topic_id,
                    topic_name=active_topic.name if active_topic else None,
                    request_type=request_type or "chat",
                    provider=fb_prov,
                    model=fb_mod,
                    prompt_summary=user_prompt if user_prompt else None,
                    request_capture=fallback_capture,
                    raw_response="",
                    latency_ms=fallback_latency,
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

                ai_log_ids = [i for i in (primary_log_id, fb_log_id) if i is not None]
                await send_terminal_ai_failure_alert(
                    bot=bot,
                    platform="telegram",
                    user_id=user_id,
                    primary_provider=primary_prov,
                    primary_model=primary_mod,
                    fallback_provider=fb_prov,
                    fallback_model=fb_mod,
                    exception=fb_err,
                    classification=fb_err_meta["error_classification"],
                    ai_log_ids=ai_log_ids if ai_log_ids else None,
                )

                logging.error(f"Fallback provider '{fb_prov}' also failed: {fb_err}")
                attempts = (
                    _build_ai_attempt(primary_prov, primary_mod, primary_err),
                    _build_ai_attempt(
                        fb_prov,
                        fb_mod,
                        fb_err,
                        include_context=False,
                    ),
                )
                fallback_classification, _ = classify_external_error(
                    fb_err,
                    provider=fb_prov,
                    include_context=False,
                )
                service_err = AIServiceError(
                    f"Основной провайдер ({primary_prov}) и резервный ({fb_prov}) недоступны"
                )
                service_err.provider_attempts = attempts
                service_err.ai_outcome = (
                    "PRIMARY_FAILED + FALLBACK_CONFIGURATION_ERROR"
                    if fallback_classification == "configuration"
                    else "BOTH_FAILED"
                )
                service_err.classification = getattr(fb_err, "classification", fallback_classification)
                raise service_err from fb_err

        automation_result = None
        if service_blocks and persist_service_data:
            try:
                automation_result = await apply_service_data_blocks(
                    session,
                    user=user,
                    dialogue_id=active_dialogue_id,
                    topic_id=active_topic_id,
                    blocks=service_blocks,
                )
                await session.commit()
            except Exception as commit_err:
                if hasattr(session, "rollback"):
                    await session.rollback()
                logging.error("Failed to commit service data for user %s: %s", user_id, commit_err)
                raise AIServiceError(f"Ошибка сохранения служебных данных ИИ: {commit_err}") from commit_err


        if bot is not None and automation_result is not None and automation_result.event_names:
            from automation_events import process_pending_events
            try:
                await process_pending_events(bot, user_id=user.id)
            except Exception:
                logging.exception("Immediate automation event processing failed for user %s", user.id)

        if bot is not None and getattr(user, "ai_debug_enabled", False):
            try:
                log_ref = f"#{successful_log_id}" if successful_log_id else ""
                debug_msg = (
                    f"🐛 <b>[AI DEBUG LOG]</b> {log_ref}\n"
                    f"🤖 <b>Провайдер:</b> {html.escape(actual_provider)} | <b>Модель:</b> {html.escape(actual_model)}\n"
                    f"⏱ <b>Время ответа:</b> {latency_ms / 1000:.2f} сек\n"
                    f"👤 <b>Пользователь:</b> ID {user_id}\n\n"
                    f"📥 <b>Сырой ответ модели:</b>\n"
                    f"<code>{html.escape(response_text[:3500])}</code>"
                )
                await bot.send_message(chat_id=user_id, text=debug_msg, parse_mode="HTML")
            except Exception as exc:
                logging.warning("Could not send live AI debug message to user %s: %s", user_id, exc)

        return visible_text


async def _call_gemini_transcribe(api_key: str, model: str, file_bytes: bytes, filename: str) -> str:
    import httpx
    import base64

    try:
        transport = _build_async_transport_from_env("GEMINI_PROXY")

        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type or not mime_type.startswith('audio/'):
            mime_type = 'audio/ogg'

        b64_data = base64.b64encode(file_bytes).decode('utf-8')
        target_model = model if model else get_default_model(PROVIDER_GEMINI, channel="transcription")
        ensure_model_available(PROVIDER_GEMINI, target_model, channel="transcription")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"

        payload = {
            "contents": [{
                "parts": [
                    {"text": "Сделай транскрипцию этой речи. Язык речи: русский. Верни только текст."},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    }
                ]
            }]
        }

        headers = {'Content-Type': 'application/json'}

        async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                error_data = response.json()
                error_payload = error_data.get('error', {}) if isinstance(error_data, dict) else {}
                error_msg = _nonempty_provider_reason(
                    error_payload.get('message') if isinstance(error_payload, dict) else None,
                    _nonempty_provider_reason(
                        getattr(response, 'text', None),
                        f"HTTP {response.status_code} без описания",
                    ),
                )

                if "User location" in error_msg:
                    raise InsufficientBalanceError(f"Gemini Geo-Block (Transcription): {error_msg}")

                logging.error(f"Gemini Transcribe REST Error: {response.status_code} - {error_msg}")
                raise AIServiceError(f"Ошибка транскрибации Gemini: {error_msg}")

            data = response.json()
            try:
                candidates = data.get('candidates', [])
                if not candidates:
                    return "Не удалось извлечь текст (пустой ответ от Gemini)."

                return candidates[0]['content']['parts'][0]['text']
            except (KeyError, IndexError) as e:
                logging.error(f"Gemini transcribe parsing error: {e}. Data: {data}")
                return "Не удалось извлечь текст транскрипции."

    except Exception as e:
        logging.error(f"Gemini API transcription error: {e}")
        if "billing" in str(e).lower() or "geo-block" in str(e).lower():
            raise InsufficientBalanceError(f"Gemini Error: {exception_summary(e)}") from e
        raise AIServiceError(f"Ошибка при транскрибации (Gemini API): {exception_summary(e)}") from e


async def _call_kie_transcribe(api_key: str, base_url: str, upload_base_url: str, model: str, file_bytes: bytes, filename: str) -> str:
    ensure_model_available(PROVIDER_KIE, model, channel="transcription")
    try:
        upload_bytes, upload_filename = await _prepare_kie_transcription_audio(file_bytes, filename)
        file_url = await _retry_kie_transcription_step(
            "upload",
            lambda: _upload_file_to_kie(
                api_key,
                upload_base_url,
                upload_bytes,
                upload_filename,
                "audio",
            ),
        )
        if model == "elevenlabs/speech-to-text":
            task_id = await _retry_kie_transcription_step(
                "create_task",
                lambda: _create_kie_task(
                    api_key,
                    base_url,
                    model,
                    {
                        "audio_url": file_url,
                        "language_code": "ru",
                        "tag_audio_events": False,
                        "diarize": False,
                    },
                ),
            )
            task_payload = await _poll_kie_task(api_key, base_url, task_id)
            result = _extract_kie_task_result(task_payload)
            transcription = _find_first_string_value(
                result,
                ("text", "transcript", "transcription", "content", "result"),
            )
            if not transcription:
                raise AIResponseError(f"KIE STT returned no transcription text: task_id={task_id} payload={task_payload}")
            return transcription

        prompt = "Сделай точную транскрипцию аудио. Язык речи: русский. Верни только текст без пояснений."
        return await _call_kie_multimodal(
            api_key,
            base_url,
            model,
            "Ты — сервис точной транскрибации речи.",
            [
                {"type": "text", "text": prompt},
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


async def _call_gemini_image_generation(api_key: str, model: str, prompt: str) -> bytes:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="image_gen")
    try:
        return await gemini_image.generate_image(api_key, target_model, prompt)
    except gemini_image.GeminiImageResponseError as exc:
        raise AIResponseError(str(exc)) from exc
    except gemini_image.GeminiImageError as exc:
        raise AIServiceError(str(exc)) from exc


async def edit_image_gemini_v3(api_key: str, model: str, prompt: str, image_bytes: bytes) -> bytes:
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="image_edit")
    try:
        return await gemini_image.edit_image(api_key, target_model, prompt, image_bytes)
    except gemini_image.GeminiImageResponseError as exc:
        raise AIResponseError(str(exc)) from exc
    except gemini_image.GeminiImageError as exc:
        raise AIServiceError(str(exc)) from exc


def _build_kie_image_generation_input(model: str, prompt: str) -> dict:
    aspect_ratio, _ = _select_image_generation_shape(prompt)
    if model == "google/imagen4-fast":
        return {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": "1",
        }
    if model in {"google/imagen4-ultra", "google/imagen4"}:
        return {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
        }
    if model == "bytedance/seedream-v4-text-to-image":
        return {
            "prompt": prompt,
            "image_size": "square_hd",
            "image_resolution": "1K",
            "max_images": 1,
        }
    if model == "seedream/4.5-text-to-image":
        return {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "quality": "basic",
        }
    raise AIServiceError(f"Неподдерживаемая KIE image generation model: {model}")


def _select_image_generation_shape(prompt: str) -> tuple[str, str]:
    prompt_lc = (prompt or "").lower()

    portrait_markers = (
        "tarot",
        "card",
        "oracle",
        "poster",
        "cover",
        "vertical",
        "portrait orientation",
        "full body",
        "full-body",
        "phone wallpaper",
    )
    landscape_markers = (
        "landscape orientation",
        "horizontal",
        "wide shot",
        "widescreen",
        "panoramic",
        "banner",
        "cinematic wide",
    )

    if any(marker in prompt_lc for marker in portrait_markers):
        return "3:4", "1024x1536"
    if any(marker in prompt_lc for marker in landscape_markers):
        return "4:3", "1536x1024"
    return "1:1", "1024x1024"


def _aspect_ratio_to_seedream_size(aspect_ratio: str) -> str:
    if aspect_ratio == "3:4":
        return "portrait_3_4"
    if aspect_ratio == "4:3":
        return "landscape_4_3"
    return "square_hd"


def _build_kie_image_edit_input(model: str, prompt: str, source_url: str) -> dict:
    aspect_ratio, _ = _select_image_generation_shape(prompt)

    if model == "google/nano-banana-edit":
        return {
            "prompt": prompt,
            "image_urls": [source_url],
            "output_format": "png",
            "image_size": "1:1",
        }
    if model == "bytedance/seedream-v4-edit":
        return {
            "prompt": prompt,
            "image_urls": [source_url],
            "image_size": "square_hd",
            "image_resolution": "1K",
            "max_images": 1,
        }
    if model == "seedream/4.5-edit":
        return {
            "prompt": prompt,
            "image_urls": [source_url],
            "aspect_ratio": aspect_ratio,
            "quality": "basic",
        }
    raise AIServiceError(f"Неподдерживаемая KIE image edit model: {model}")


async def _call_kie_image_generation(api_key: str, base_url: str, model: str, prompt: str) -> bytes:
    ensure_model_available(PROVIDER_KIE, model, channel="image_gen")
    attempts = 2
    last_exc = None
    for _ in range(attempts):
        try:
            task_id = await _create_kie_task(
                api_key,
                base_url,
                model,
                _build_kie_image_generation_input(model, prompt),
            )
            task_payload = await _poll_kie_task(api_key, base_url, task_id)
            result = _extract_kie_task_result(task_payload)
            result_urls = result.get("resultUrls") or result.get("result_urls") or []
            if not result_urls:
                raise AIResponseError(f"KIE image generation returned no result URLs: task_id={task_id} payload={task_payload}")
            download_url = await _get_kie_download_url(api_key, base_url, result_urls[0])
            return await _download_binary_file(download_url)
        except AIServiceError as exc:
            last_exc = exc
            if "internal error" not in str(exc).lower():
                raise
            await asyncio.sleep(2)
    raise last_exc or AIServiceError("KIE image generation failed without detailed error")


async def _call_kie_image_edit(api_key: str, base_url: str, upload_base_url: str, model: str, prompt: str, image_bytes: bytes) -> bytes:
    ensure_model_available(PROVIDER_KIE, model, channel="image_edit")
    source_url = await _upload_file_to_kie(
        api_key,
        upload_base_url,
        image_bytes,
        _guess_filename(image_bytes, "image_edit_source", "jpg"),
        "images",
    )
    task_id = await _create_kie_task(
        api_key,
        base_url,
        model,
        _build_kie_image_edit_input(model, prompt, source_url),
    )
    task_payload = await _poll_kie_task(api_key, base_url, task_id)
    result = _extract_kie_task_result(task_payload)
    result_urls = result.get("resultUrls") or result.get("result_urls") or []
    if not result_urls:
        raise AIResponseError(f"KIE image edit returned no result URLs: task_id={task_id} payload={task_payload}")
    download_url = await _get_kie_download_url(api_key, base_url, result_urls[0])
    return await _download_binary_file(download_url)


async def generate_image(prompt: str) -> any:
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            raise AIServiceError("Конфигурация ИИ не найдена.")

        configured_provider = getattr(config, "image_generation_provider", None)
        if configured_provider is None:
            provider = config.vision_provider or PROVIDER_OPENAI
        elif not isinstance(configured_provider, str) or not configured_provider.strip():
            raise AIServiceError("Некорректно задан провайдер генерации изображений.")
        else:
            provider = configured_provider
        provider_key = _normalize_provider_name(provider)
        if provider_key not in {"gemini", "kie", "openai"}:
            raise AIServiceError(f"Генерация изображений не поддерживается для провайдера: {provider}")
        model = getattr(config, "image_generation_model", None) or get_default_model(provider, channel="image_gen")

    if provider_key == 'gemini':
        ensure_model_available(PROVIDER_GEMINI, model, channel="image_gen")
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для генерации не установлен.")
        return await _call_gemini_image_generation(api_key, model, prompt)
    if provider_key == 'kie':
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для генерации не установлен.")
        target_model = model or get_default_model(PROVIDER_KIE, channel="image_gen")
        ensure_model_available(PROVIDER_KIE, target_model, channel="image_gen")
        return await _call_kie_image_generation(api_key, _get_kie_base_url(config), target_model, prompt)
    if provider_key == 'openai':
        return await generate_openai_image(prompt)
    raise AIServiceError(f"Генерация изображений не поддерживается для провайдера: {provider}")


async def edit_image(prompt: str, image_bytes: bytes) -> bytes:
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            raise AIServiceError("Конфигурация ИИ не найдена.")

        provider = getattr(config, "image_edit_provider", None) or config.vision_provider or PROVIDER_KIE
        provider_key = _normalize_provider_name(provider)
        model = getattr(config, "image_edit_model", None) or get_default_model(provider, channel="image_edit")

    if provider_key == "gemini":
        ensure_model_available(PROVIDER_GEMINI, model, channel="image_edit")
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для редактирования не установлен.")
        return await edit_image_gemini_v3(api_key, model, prompt, image_bytes)
    if provider_key == "kie":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для редактирования не установлен.")
        target_model = model or get_default_model(PROVIDER_KIE, channel="image_edit")
        ensure_model_available(PROVIDER_KIE, target_model, channel="image_edit")
        return await _call_kie_image_edit(
            api_key,
            _get_kie_base_url(config),
            _get_kie_upload_base_url(config),
            target_model,
            prompt,
            image_bytes,
        )
    raise AIServiceError(f"Редактирование изображений не поддерживается для провайдера: {provider}")


async def generate_openai_image(prompt: str) -> str:
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        api_key = config.openai_api_key if config and hasattr(config, 'openai_api_key') else None

    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise AIServiceError("API ключ OpenAI не установлен.")

    base_url = os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1")
    timeout = 60.0
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    try:
        model = get_default_model(PROVIDER_OPENAI, channel="image_gen")
        ensure_model_available(PROVIDER_OPENAI, model, channel="image_gen")
        _, preferred_size = _select_image_generation_shape(prompt)

        logging.info(f"Generating image via {model} with prompt: {prompt}")
        requested_sizes = [preferred_size]
        if preferred_size != "1024x1024":
            requested_sizes.append("1024x1024")

        response = None
        last_error = None
        for size in requested_sizes:
            try:
                response = await client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size=size
                )
                logging.info("OpenAI image generation completed with size=%s", size)
                break
            except Exception as exc:
                last_error = exc
                logging.warning("OpenAI image generation failed with size=%s: %s", size, exc)

        if response is None:
            raise last_error or AIServiceError("OpenAI image generation failed without response")

        if not response.data:
            raise AIServiceError("API не вернул данных (empty data).")

        img_data = response.data[0]

        if img_data.url:
            return img_data.url
        elif img_data.b64_json:
            return base64.b64decode(img_data.b64_json)
        else:
            raise AIServiceError("API не вернул ни URL, ни B64.")

    except Exception as e:
        logging.error(f"OpenAI Image Error ({model}): {e}")
        raise AIServiceError(f"Ошибка генерации изображения: {exception_summary(e)}") from e


async def _call_openai_vision(
    api_key: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    history: list = None,
    temperature: float = 0.7,
    request_context: str = "",
    request_capture: dict | None = None,
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    effective_user_prompt: str | None = None,
    stage_budget: float = 25.0,
) -> str:
    target_v_model = model or "gpt-5.6-terra"
    ensure_model_available(PROVIDER_OPENAI, target_v_model, channel="vision")
    max_tokens = get_provider_vision_max_tokens(PROVIDER_OPENAI)

    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    user_text = effective_user_prompt or _extract_vision_user_prompt(
        request_layout=request_layout,
        prompt=prompt,
    )
    layout = _coerce_request_layout(
        request_layout,
        history=history,
        system_prompt=prompt,
        runtime_context=request_context,
        current_user_content=None,
    ).with_current_user_content([
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "high"}},
    ])

    base_url = os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=stage_budget, max_retries=0)

    try:
        payload = {
            "model": target_v_model,
            "messages": build_openai_chat_messages(layout),
            "max_completion_tokens": max_tokens,
        }
        if not target_v_model.startswith("gpt-5.6"):
            payload["temperature"] = temperature

        _capture_ai_request(request_capture, provider="OpenAI", endpoint=f"{base_url.rstrip('/')}/chat/completions", payload=payload)
        if activity_tracker is not None:
            try:
                await activity_tracker.mark_outbound_attempt_once()
            except Exception as act_err:
                logging.warning("Failed to mark activity before outbound OpenAI vision call: %s", act_err)
        response = await client.chat.completions.create(**payload)

        choices = getattr(response, "choices", None)
        first_choice = choices[0] if choices else None
        finish_reason = getattr(first_choice, "finish_reason", None) if first_choice else None
        msg = getattr(first_choice, "message", None) if first_choice else None
        refusal = getattr(msg, "refusal", None) if msg else None
        content = getattr(msg, "content", None) if msg else None

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

        if not choices:
            raise attach_error_metadata(
                AIResponseError("OpenAI Vision вернул ответ без choices"),
                classification="empty_response",
                http_status=200,
                diagnostics={"usage": usage_dict} if usage_dict else None,
                provider_response_payload=safe_resp_payload,
            )

        has_refusal = isinstance(refusal, str) and bool(refusal.strip())
        is_content_filter = (finish_reason == "content_filter") or (
            isinstance(finish_reason, str) and finish_reason.lower() == "content_filter"
        )
        if has_refusal or is_content_filter:
            rejection_detail = refusal if has_refusal else finish_reason
            raise attach_error_metadata(
                AIServiceError(f"OpenAI Vision rejection: {rejection_detail}"),
                classification="provider_rejection",
                finish_reason="content_filter" if is_content_filter else "refusal",
                http_status=200,
                diagnostics={"usage": usage_dict} if usage_dict else None,
                provider_response_payload=safe_resp_payload,
            )

        is_length = (finish_reason == "length") or (
            isinstance(finish_reason, str) and finish_reason.lower() == "length"
        )
        if is_length:
            raise attach_error_metadata(
                AIResponseError("OpenAI Vision output budget exhausted"),
                classification="output_budget_exhausted",
                finish_reason="length",
                http_status=200,
                diagnostics={"usage": usage_dict} if usage_dict else None,
                provider_response_payload=safe_resp_payload,
            )

        if not isinstance(content, str) or not content.strip():
            raise attach_error_metadata(
                AIResponseError("OpenAI Vision вернул пустой content"),
                classification="empty_response",
                finish_reason=str(finish_reason) if isinstance(finish_reason, str) else None,
                http_status=200,
                diagnostics={"usage": usage_dict} if usage_dict else None,
                provider_response_payload=safe_resp_payload,
            )

        return content.strip()
    except (AIServiceError, AIResponseError):
        raise
    except Exception as e:
        logging.error(f"OpenAI Vision Error: {e}")
        raise AIServiceError(f"Ошибка анализа изображения (OpenAI): {exception_summary(e)}") from e


async def _call_gemini_vision(
    api_key: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    history: list = None,
    temperature: float = 0.7,
    request_context: str = "",
    request_capture: dict | None = None,
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    effective_user_prompt: str | None = None,
    stage_budget: float = 25.0,
) -> str:
    raw_proxy = os.getenv("GEMINI_PROXY")
    transport = None
    if raw_proxy:
        gemini_proxy = raw_proxy.strip().strip('"').strip("'")
        transport = httpx.AsyncHTTPTransport(proxy=gemini_proxy)

    b64_data = base64.b64encode(image_bytes).decode('utf-8')
    target_model = model if model else "gemini-3.7-flash"
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="vision")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"

    user_instruction = effective_user_prompt or _extract_vision_user_prompt(
        request_layout=request_layout,
        prompt=prompt,
    )
    layout = _coerce_request_layout(
        request_layout,
        history=history,
        system_prompt=prompt,
        runtime_context=request_context,
        current_user_content=None,
    ).with_current_user_content([
        {"text": user_instruction},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}},
    ])

    max_tokens = get_provider_vision_max_tokens(PROVIDER_GEMINI)
    generation_config: dict[str, Any] = {
        "maxOutputTokens": max_tokens,
    }
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
    _capture_ai_request(request_capture, provider="Gemini", endpoint=endpoint, payload=payload)

    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            logging.warning("Failed to mark activity before outbound Gemini vision call: %s", act_err)

    timeout = build_vision_httpx_timeout(stage_budget)
    try:
        async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=timeout) as client:
            response = await client.post(url, json=payload, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            data = response.json()
            prompt_feedback = data.get("promptFeedback") or {}
            block_reason = prompt_feedback.get("blockReason")
            candidates = data.get('candidates') or []
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
                request_capture["http_status"] = 200
                request_capture["finish_reason"] = finish_reason
                if usage_meta:
                    request_capture["usage"] = usage_meta
                if safe_resp_payload is not None:
                    request_capture["provider_response_payload"] = safe_resp_payload

            if block_reason:
                raise attach_error_metadata(
                    AIServiceError(f"Gemini Vision prompt blocked: {block_reason}"),
                    classification="provider_rejection",
                    finish_reason=block_reason,
                    http_status=200,
                    diagnostics={"usage": usage_meta} if usage_meta else None,
                    provider_response_payload=safe_resp_payload,
                )

            if not candidates:
                raise attach_error_metadata(
                    AIResponseError("Gemini Vision returned no candidates"),
                    classification="empty_response",
                    http_status=200,
                    diagnostics={"usage": usage_meta} if usage_meta else None,
                    provider_response_payload=safe_resp_payload,
                )

            if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
                raise attach_error_metadata(
                    AIServiceError(f"Gemini Vision candidate blocked: {finish_reason}"),
                    classification="provider_rejection",
                    finish_reason=finish_reason,
                    http_status=200,
                    diagnostics={"usage": usage_meta} if usage_meta else None,
                    provider_response_payload=safe_resp_payload,
                )

            if finish_reason == "MAX_TOKENS":
                raise attach_error_metadata(
                    AIResponseError("Gemini Vision output budget exhausted"),
                    classification="output_budget_exhausted",
                    finish_reason="MAX_TOKENS",
                    http_status=200,
                    diagnostics={"usage": usage_meta} if usage_meta else None,
                    provider_response_payload=safe_resp_payload,
                )

            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            if not text:
                raise attach_error_metadata(
                    AIResponseError("Gemini Vision returned empty content"),
                    classification="empty_response",
                    finish_reason=finish_reason,
                    http_status=200,
                    diagnostics={"usage": usage_meta} if usage_meta else None,
                    provider_response_payload=safe_resp_payload,
                )

            return text

        error_detail = response.text
        logging.error(f"Gemini Vision API Error ({response.status_code}): {error_detail}")
        err = AIServiceError(f"Ошибка API Gemini Vision: {response.status_code}")
        err.http_status = response.status_code
        err.provider_response_payload = error_detail
        raise err
    except (AIServiceError, InsufficientBalanceError, AIResponseError):
        raise
    except Exception as e:
        logging.error(f"Ошибка вызова Gemini Vision: {e}")
        raise AIServiceError(f"Ошибка анализа изображения (Gemini): {exception_summary(e)}") from e


async def _call_kie_vision_inference(
    api_key: str,
    base_url: str,
    model: str,
    file_url: str,
    prompt: str,
    history: list = None,
    temperature: float = 0.7,
    request_context: str = "",
    request_capture: dict | None = None,
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    effective_user_prompt: str | None = None,
    stage_budget: float = 25.0,
) -> str:
    ensure_model_available(PROVIDER_KIE, model, channel="vision")
    user_instruction = effective_user_prompt or _extract_vision_user_prompt(
        request_layout=request_layout,
        prompt=prompt,
    )
    user_content = [
        {"type": "text", "text": user_instruction},
        {"type": "image_url", "image_url": {"url": file_url}},
    ]
    layout = _coerce_request_layout(
        request_layout,
        history=history,
        system_prompt=prompt,
        runtime_context=request_context,
        current_user_content=user_content,
    ).with_current_user_content(user_content)

    payload = {
        "model": model,
        "messages": build_openai_chat_messages(layout),
        "max_tokens": KIE_VISION_INITIAL_MAX_TOKENS,
        "temperature": temperature,
        "stream": False,
    }
    endpoint = f"{_kie_model_base_url(base_url, model)}/chat/completions"
    _capture_ai_request(request_capture, provider="KIE", endpoint=endpoint, payload=payload)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if activity_tracker is not None:
        try:
            await activity_tracker.mark_outbound_attempt_once()
        except Exception as act_err:
            logging.warning("Failed to mark activity before outbound KIE vision inference: %s", act_err)

    timeout = build_vision_httpx_timeout(stage_budget)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(endpoint, headers=headers, json=payload)

        try:
            raw_payload = response.json()
        except Exception:
            raw_payload = {}

        if response.status_code >= 400:
            err = AIServiceError(f"KIE multimodal API error (HTTP {response.status_code}): {response.text}")
            err.http_status = response.status_code
            if isinstance(raw_payload, dict) and "code" in raw_payload:
                err.provider_code = raw_payload.get("code")
            err.provider_response_payload = response.text
            raise err

        response_payload = _validate_kie_json_response(
            response.status_code,
            raw_payload,
            context="Ошибка обращения к KIE multimodal API",
        )

        choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
        first_choice = choices[0] if (choices and isinstance(choices, list)) else None
        finish_reason = first_choice.get("finish_reason") if first_choice else None
        usage_data = response_payload.get("usage") if isinstance(response_payload, dict) else None
        provider_code_val = raw_payload.get("code") if isinstance(raw_payload, dict) else None

        safe_resp_payload = None
        try:
            import json
            safe_resp_payload = json.dumps(response_payload, ensure_ascii=False) if isinstance(response_payload, dict) else str(response_payload)
        except Exception:
            safe_resp_payload = None

        if request_capture is not None:
            request_capture["http_status"] = 200
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
                http_status=200,
                diagnostics=diag_meta,
                provider_response_payload=safe_resp_payload,
            )

        if finish_reason == "content_filter":
            raise attach_error_metadata(
                AIServiceError("KIE Vision content filter rejection"),
                classification="provider_rejection",
                finish_reason=finish_reason,
                http_status=200,
                diagnostics=diag_meta,
                provider_response_payload=safe_resp_payload,
            )

        if finish_reason == "length":
            raise attach_error_metadata(
                AIResponseError("KIE Vision output budget exhausted"),
                classification="output_budget_exhausted",
                finish_reason="length",
                http_status=200,
                diagnostics=diag_meta,
                provider_response_payload=safe_resp_payload,
            )

        text = _extract_kie_chat_text(response_payload)
        if not text:
            raise attach_error_metadata(
                AIResponseError("KIE multimodal request returned empty content"),
                classification="empty_response",
                finish_reason=finish_reason,
                http_status=200,
                diagnostics=diag_meta,
                provider_response_payload=safe_resp_payload,
            )

        return text
    except (InsufficientBalanceError, AIServiceError, AIResponseError):
        raise
    except Exception as e:
        logging.error("KIE vision inference error", exc_info=e)
        raise AIServiceError(f"Ошибка обращения к KIE multimodal API: {exception_summary(e)}") from e


async def _call_kie_vision(
    api_key: str,
    base_url: str,
    upload_base_url: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    history: list = None,
    temperature: float = 0.7,
    request_context: str = "",
    request_capture: dict | None = None,
    *,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    effective_user_prompt: str | None = None,
    stage_budget: float = 25.0,
) -> str:
    file_url = await _upload_file_to_kie(
        api_key,
        upload_base_url,
        image_bytes,
        _guess_filename(image_bytes, "vision_input", "jpg"),
        "images",
        timeout=build_vision_httpx_timeout(min(stage_budget, 20.0)),
        activity_tracker=activity_tracker,
    )
    return await _call_kie_vision_inference(
        api_key,
        base_url,
        model,
        file_url,
        prompt,
        history=history,
        temperature=temperature,
        request_context=request_context,
        request_capture=request_capture,
        request_layout=request_layout,
        activity_tracker=activity_tracker,
        effective_user_prompt=effective_user_prompt,
        stage_budget=stage_budget,
    )


async def analyze_image_content(
    image_bytes: bytes,
    prompt: str = "",
    history: list = None,
    *,
    user_prompt: str | None = None,
    request_context: str = "",
    request_capture: dict | None = None,
    request_layout: AIRequestLayout | None = None,
    activity_tracker: ActivityTracker | None = None,
    execution_context: VisionExecutionContext | None = None,
) -> str:
    effective_user_prompt = _extract_vision_user_prompt(
        user_prompt=user_prompt,
        request_layout=request_layout,
        prompt=prompt,
    )

    tracker = VisionDeadlineTracker(total_deadline_seconds=85.0)
    min_useful_budget = 3.0

    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            raise AIServiceError("Конфигурация ИИ не найдена.")

        primary_provider = getattr(config, "vision_provider", None) or "OpenAI"
        primary_model = getattr(config, "vision_model", None)
        allow_fallback = getattr(config, "allow_vision_fallback", False)
        fallback_provider = getattr(config, "vision_fallback_provider", None)
        fallback_model = getattr(config, "vision_fallback_model", None)
        temperature = _resolve_temperature(config)

        if primary_provider == "None":
            raise AIServiceError("Обработка изображений отключена администратором.")

    eff_allow_fallback, eff_fallback_provider, eff_fallback_model = resolve_effective_vision_fallback(
        primary_provider, allow_fallback, fallback_provider, fallback_model
    )

    # Pre-flight API key check for primary before any outbound HTTP
    if primary_provider not in (PROVIDER_OPENAI, PROVIDER_CLAUDE, PROVIDER_GEMINI, PROVIDER_KIE):
        err = AIServiceError(f"Неподдерживаемый провайдер для vision: {primary_provider}")
        err.classification = "configuration"
        raise err

    primary_target_model = primary_model or get_default_model(primary_provider, channel="vision")
    try:
        ensure_model_available(primary_provider, primary_target_model, channel="vision")
    except Exception as exc:
        err = AIServiceError(f"Недопустимая модель vision {primary_target_model} для {primary_provider}: {exc}")
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

    # Pre-flight check fallback configuration validity
    if eff_allow_fallback and eff_fallback_provider:
        fb_key_valid = True
        if eff_fallback_provider == "OpenAI" and not (config.openai_api_key or os.getenv('OPENAI_API_KEY')):
            fb_key_valid = False
        elif eff_fallback_provider == "Claude" and not config.claude_api_key:
            fb_key_valid = False
        elif eff_fallback_provider == "Gemini" and not config.gemini_api_key:
            fb_key_valid = False
        elif eff_fallback_provider == "KIE" and not getattr(config, "kie_api_key", None):
            fb_key_valid = False
        if not fb_key_valid:
            eff_allow_fallback = False

    request_group_id = f"vision-{uuid.uuid4().hex[:12]}"
    attempt_no = 0
    provider_attempts_summary = []

    async def _audit_attempt(
        p_name: str,
        m_name: str,
        role: str,
        status: str,
        latency_ms: int,
        attempt_capture: dict | None,
        raw_resp: str | None,
        clean_txt: str | None,
        err: Exception | None = None,
        classification: str | None = None,
        http_status: int | None = None,
        finish_reason: str | None = None,
        diagnostics: Any = None,
        provider_resp_payload: str | None = None,
    ):
        nonlocal attempt_no
        attempt_no += 1
        finish_reason = finish_reason or getattr(err, "finish_reason", None) or (attempt_capture.get("finish_reason") if attempt_capture else None)
        http_status = http_status or getattr(err, "http_status", None) or (attempt_capture.get("http_status") if attempt_capture else None)
        provider_resp_payload = provider_resp_payload or getattr(err, "provider_response_payload", None) or (attempt_capture.get("provider_response_payload") if attempt_capture else None)
        if diagnostics is None:
            err_diag = getattr(err, "diagnostics", None)
            if err_diag:
                diagnostics = err_diag
            elif attempt_capture:
                diag_dict = {k: v for k, v in attempt_capture.items() if k in ("usage", "finish_reason", "provider_code", "stage", "fallback_kind")}
                if diag_dict:
                    diagnostics = diag_dict
        elif isinstance(diagnostics, dict) and attempt_capture:
            for k in ("usage", "finish_reason", "provider_code"):
                if k in attempt_capture and k not in diagnostics:
                    diagnostics[k] = attempt_capture[k]

        fb_kind = getattr(diagnostics, "fallback_kind", None) if hasattr(diagnostics, "fallback_kind") else (diagnostics.get("fallback_kind") if isinstance(diagnostics, dict) else None)
        stg = getattr(diagnostics, "stage", None) if hasattr(diagnostics, "stage") else (diagnostics.get("stage") if isinstance(diagnostics, dict) else None)

        provider_attempts_summary.append({
            "provider": p_name,
            "model": m_name,
            "status": status,
            "classification": classification,
            "error": sanitize_vision_text(str(err)) if err else None,
            "attempt_no": attempt_no,
            "attempt_role": role,
            "fallback_kind": fb_kind,
            "stage": stg,
            "attempt": attempt_no,
            "role": role,
        })
        if execution_context is None:
            return

        sanitized_capture = sanitize_vision_request_payload(attempt_capture)
        sanitized_raw = sanitize_vision_text(raw_resp) if raw_resp else None
        sanitized_clean = sanitize_vision_text(clean_txt) if clean_txt else None
        sanitized_err_msg = sanitize_vision_text(str(err)) if err else None
        sanitized_resp_payload = sanitize_vision_text(provider_resp_payload) if provider_resp_payload else None
        sanitized_diag = sanitize_vision_request_payload(diagnostics) if diagnostics else None
        eff_classification = (
            str(classification[0])
            if isinstance(classification, (tuple, list))
            else (str(classification) if classification else None)
        )
        eff_finish_reason = str(finish_reason) if isinstance(finish_reason, str) else None

        async with async_session_maker() as audit_session:
            await record_ai_attempt_log(
                audit_session,
                user_id=execution_context.user_id,
                platform=execution_context.platform,
                dialogue_id=execution_context.dialogue_id,
                topic_id=execution_context.topic_id,
                topic_name=execution_context.topic_name,
                request_type="vision",
                provider=p_name,
                model=m_name,
                prompt_summary=effective_user_prompt or prompt[:100],
                request_capture=sanitized_capture,
                raw_response=sanitized_raw,
                clean_text=sanitized_clean,
                latency_ms=latency_ms,
                status=status,
                request_group_id=request_group_id,
                attempt_no=attempt_no,
                attempt_role=role,
                error_type=type(err).__name__ if err else None,
                error_message=sanitized_err_msg,
                error_classification=eff_classification,
                http_status=http_status,
                finish_reason=eff_finish_reason,
                diagnostics=sanitized_diag,
                provider_response_payload=sanitized_resp_payload,
            )

    last_error: Exception | None = None
    last_classification: str | None = None
    last_failed_provider: str = primary_provider
    last_failed_model: str = primary_target_model
    last_failed_budget: int = 4096 if primary_provider == "KIE" else get_provider_vision_max_tokens(primary_provider)
    success_result: str | None = None

    if primary_provider == "KIE":
        uploaded_file_url = None
        upload_budget = tracker.stage_budget("kie_upload", aggregate_cap=20.0)
        if upload_budget < min_useful_budget:
            raise AIServiceError("Таймаут до начала загрузки изображения в KIE.")

        upload_start = time.monotonic()
        upload_stage_cap = 20.0
        upload_err = None
        upload_capture = {}
        for upload_attempt in range(2):
            rem_upload = upload_stage_cap - (time.monotonic() - upload_start)
            if rem_upload < min_useful_budget:
                break
            cur_upload_budget = tracker.stage_budget("kie_upload", aggregate_cap=rem_upload, min_required=min_useful_budget)
            if cur_upload_budget < min_useful_budget:
                break
            try:
                uploaded_file_url = await run_coro_with_timeout(
                    _upload_file_to_kie(
                        primary_api_key,
                        _get_kie_upload_base_url(config),
                        image_bytes,
                        _guess_filename(image_bytes, "vision_input", "jpg"),
                        "images",
                        timeout=build_vision_httpx_timeout(cur_upload_budget),
                        activity_tracker=activity_tracker,
                    ),
                    cur_upload_budget,
                )
                upload_err = None
                break
            except Exception as u_exc:
                upload_err = u_exc
                u_cls, _ = classify_external_error(u_exc)
                if upload_attempt == 0 and should_retry_kie_vision_upload(u_cls):
                    continue
                break

        upload_latency = int((time.monotonic() - upload_start) * 1000)
        if not uploaded_file_url:
            last_error = upload_err or AIServiceError("Ошибка загрузки файла в KIE")
            last_classification, _ = classify_external_error(last_error)
            last_failed_provider = "KIE"
            last_failed_model = primary_target_model
            upload_diag = SimpleNamespace(stage="kie_upload", inference_http_started=False)
            await _audit_attempt(
                "KIE",
                primary_target_model,
                "primary",
                "error",
                upload_latency,
                upload_capture,
                None,
                None,
                err=last_error,
                classification=last_classification,
                http_status=getattr(last_error, "http_status", None),
                diagnostics=upload_diag,
                provider_resp_payload=getattr(last_error, "provider_response_payload", None),
            )
        else:
            kie_candidates = order_kie_vision_candidates(
                primary_target_model,
                selectable_models=get_selectable_models(PROVIDER_KIE, "vision"),
            )
            for idx, cand_model in enumerate(kie_candidates):
                ensure_model_available(PROVIDER_KIE, cand_model, channel="vision")
                stage_name = "primary_inference" if idx == 0 else "kie_alternate_model"
                stage_cap = 25.0 if idx == 0 else 20.0
                cand_budget = tracker.stage_budget(stage_name, aggregate_cap=stage_cap, min_required=min_useful_budget)
                if cand_budget < min_useful_budget:
                    break

                cand_start = time.monotonic()
                cand_capture = {}
                cand_diag = {"stage": stage_name}
                if idx > 0:
                    cand_diag["fallback_kind"] = "model"
                try:
                    res = await run_coro_with_timeout(
                        _call_kie_vision_inference(
                            primary_api_key,
                            _get_kie_base_url(config),
                            cand_model,
                            uploaded_file_url,
                            prompt,
                            history=history,
                            temperature=temperature,
                            request_context=request_context,
                            request_capture=cand_capture,
                            request_layout=request_layout,
                            activity_tracker=activity_tracker,
                            effective_user_prompt=effective_user_prompt,
                            stage_budget=cand_budget,
                        ),
                        cand_budget,
                    )
                    cand_latency = int((time.monotonic() - cand_start) * 1000)
                    if request_capture is not None:
                        request_capture.update(cand_capture)
                    clean_res, _, _ = extract_service_data(res)
                    await _audit_attempt(
                        "KIE",
                        cand_model,
                        "primary" if idx == 0 else "fallback",
                        "success",
                        cand_latency,
                        cand_capture,
                        res,
                        clean_res,
                        diagnostics=cand_diag,
                    )
                    success_result = res
                    break
                except Exception as cand_exc:
                    cand_latency = int((time.monotonic() - cand_start) * 1000)
                    cand_cls, _ = classify_external_error(cand_exc)
                    last_error = cand_exc
                    last_classification = cand_cls
                    last_failed_provider = "KIE"
                    last_failed_model = cand_model
                    await _audit_attempt(
                        "KIE",
                        cand_model,
                        "primary" if idx == 0 else "fallback",
                        "error",
                        cand_latency,
                        cand_capture,
                        None,
                        None,
                        err=cand_exc,
                        classification=cand_cls,
                        http_status=getattr(cand_exc, "http_status", None),
                        finish_reason=getattr(cand_exc, "finish_reason", None),
                        diagnostics=cand_diag,
                        provider_resp_payload=getattr(cand_exc, "provider_response_payload", None),
                    )
                    if idx == 0 and should_retry_kie_vision_model(cand_cls):
                        continue
                    break
    else:
        prim_start = time.monotonic()
        prim_budget = tracker.stage_budget("primary_inference", aggregate_cap=25.0)
        if prim_budget < min_useful_budget:
            raise AIServiceError("Таймаут до начала обращения к Vision провайдеру.")
        prim_capture = {}
        try:
            if primary_provider == "Gemini":
                res = await run_coro_with_timeout(
                    _call_gemini_vision(
                        primary_api_key,
                        primary_target_model,
                        image_bytes,
                        prompt,
                        history=history,
                        temperature=temperature,
                        request_context=request_context,
                        request_capture=prim_capture,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        effective_user_prompt=effective_user_prompt,
                        stage_budget=prim_budget,
                    ),
                    prim_budget,
                )
            elif primary_provider == "Claude":
                res = await run_coro_with_timeout(
                    _call_claude_vision(
                        primary_api_key,
                        primary_target_model,
                        image_bytes,
                        prompt,
                        history=history,
                        temperature=temperature,
                        request_context=request_context,
                        request_capture=prim_capture,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        effective_user_prompt=effective_user_prompt,
                        stage_budget=prim_budget,
                    ),
                    prim_budget,
                )
            else:
                res = await run_coro_with_timeout(
                    _call_openai_vision(
                        primary_api_key,
                        primary_target_model,
                        image_bytes,
                        prompt,
                        history=history,
                        temperature=temperature,
                        request_context=request_context,
                        request_capture=prim_capture,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        effective_user_prompt=effective_user_prompt,
                        stage_budget=prim_budget,
                    ),
                    prim_budget,
                )
            prim_latency = int((time.monotonic() - prim_start) * 1000)
            if request_capture is not None:
                request_capture.update(prim_capture)
            clean_res, _, _ = extract_service_data(res)
            await _audit_attempt(
                primary_provider,
                primary_target_model,
                "primary",
                "success",
                prim_latency,
                prim_capture,
                res,
                clean_res,
            )
            success_result = res
        except Exception as prim_exc:
            prim_latency = int((time.monotonic() - prim_start) * 1000)
            prim_cls, _ = classify_external_error(prim_exc)
            last_error = prim_exc
            last_classification = prim_cls
            last_failed_provider = primary_provider
            last_failed_model = primary_target_model
            await _audit_attempt(
                primary_provider,
                primary_target_model,
                "primary",
                "error",
                prim_latency,
                prim_capture,
                None,
                None,
                err=prim_exc,
                classification=prim_cls,
                http_status=getattr(prim_exc, "http_status", None),
                finish_reason=getattr(prim_exc, "finish_reason", None),
                provider_resp_payload=getattr(prim_exc, "provider_response_payload", None),
            )

    if success_result is not None:
        return success_result

    # Check provider fallback eligibility with strict pre-flight validation
    fallback_attempted = False
    fb_eligible = False
    fb_stage_budget = 0.0
    if eff_allow_fallback and bool(eff_fallback_provider):
        is_diff_provider = str(eff_fallback_provider).strip().lower() != str(primary_provider).strip().lower()
        is_supported_provider = str(eff_fallback_provider).strip() in (PROVIDER_OPENAI, PROVIDER_CLAUDE, PROVIDER_GEMINI, PROVIDER_KIE)
        is_valid_model = bool(eff_fallback_model)
        try:
            if is_valid_model and is_supported_provider:
                validate_model_selection(eff_fallback_provider, eff_fallback_model, channel="vision_fallback")
            else:
                is_valid_model = False
        except Exception:
            is_valid_model = False

        has_api_key = False
        if eff_fallback_provider == PROVIDER_KIE:
            has_api_key = bool(getattr(config, "kie_api_key", None))
        elif eff_fallback_provider == PROVIDER_GEMINI:
            has_api_key = bool(getattr(config, "gemini_api_key", None))
        elif eff_fallback_provider == PROVIDER_CLAUDE:
            has_api_key = bool(getattr(config, "claude_api_key", None))
        elif eff_fallback_provider == PROVIDER_OPENAI:
            has_api_key = bool(getattr(config, "openai_api_key", None) or os.getenv("OPENAI_API_KEY"))

        fb_budget_val = get_provider_vision_max_tokens(eff_fallback_provider) if is_supported_provider else 0
        should_fb = should_use_vision_provider_fallback(
            last_classification or "unknown",
            failed_budget=last_failed_budget,
            fallback_budget=fb_budget_val,
        )

        if is_diff_provider and is_supported_provider and is_valid_model and has_api_key and should_fb:
            fb_stage_budget = tracker.stage_budget("provider_fallback", aggregate_cap=15.0, min_required=min_useful_budget)
            if fb_stage_budget >= min_useful_budget:
                fb_eligible = True

    if fb_eligible:
        fallback_attempted = True
        fb_start = time.monotonic()
        fb_stage_cap = 15.0
        fb_capture = {}
        try:
            if eff_fallback_provider == "KIE":
                fb_file_url = None
                fb_upload_err = None
                for fb_up_attempt in range(2):
                    elapsed = time.monotonic() - fb_start
                    rem_stage = fb_stage_cap - elapsed
                    if rem_stage < min_useful_budget:
                        if fb_upload_err is not None:
                            raise fb_upload_err
                        raise AIServiceError("Недостаточно времени для загрузки в KIE (fallback)")
                    fb_upload_budget = tracker.stage_budget("provider_fallback_upload", aggregate_cap=rem_stage, min_required=min_useful_budget)
                    if fb_upload_budget < min_useful_budget:
                        if fb_upload_err is not None:
                            raise fb_upload_err
                        raise AIServiceError("Недостаточно времени для загрузки в KIE (fallback)")
                    try:
                        fb_file_url = await run_coro_with_timeout(
                            _upload_file_to_kie(
                                getattr(config, "kie_api_key", None),
                                _get_kie_upload_base_url(config),
                                image_bytes,
                                _guess_filename(image_bytes, "vision_input", "jpg"),
                                "images",
                                timeout=build_vision_httpx_timeout(fb_upload_budget),
                                activity_tracker=activity_tracker,
                            ),
                            fb_upload_budget,
                        )
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as up_exc:
                        fb_upload_err = up_exc
                        if fb_up_attempt == 0:
                            up_cls, _ = classify_external_error(up_exc, provider=PROVIDER_KIE)
                            if should_retry_kie_vision_upload(up_cls):
                                rem_after_err = fb_stage_cap - (time.monotonic() - fb_start)
                                rem_tracker = tracker.stage_budget("provider_fallback_upload", aggregate_cap=rem_after_err, min_required=min_useful_budget)
                                if rem_after_err >= min_useful_budget and rem_tracker >= min_useful_budget:
                                    logging.info("Retrying KIE provider fallback upload after transient error: %s", up_exc)
                                    continue
                        raise up_exc

                if not fb_file_url:
                    if fb_upload_err:
                        raise fb_upload_err
                    raise AIServiceError("Не удалось получить URL файла KIE для fallback")

                fb_spent_after_up = time.monotonic() - fb_start
                fb_inf_rem = fb_stage_cap - fb_spent_after_up
                if fb_inf_rem < min_useful_budget:
                    raise AIServiceError("Недостаточно времени для инференса KIE (fallback)")
                fb_inf_budget = tracker.stage_budget("provider_fallback_inference", aggregate_cap=fb_inf_rem, min_required=min_useful_budget)
                if fb_inf_budget < min_useful_budget:
                    raise AIServiceError("Недостаточно времени для инференса KIE (fallback)")
                fb_res = await run_coro_with_timeout(
                    _call_kie_vision_inference(
                        getattr(config, "kie_api_key", None),
                        _get_kie_base_url(config),
                        eff_fallback_model or "gemini-3-flash",
                        fb_file_url,
                        prompt,
                        history=history,
                        temperature=temperature,
                        request_context=request_context,
                        request_capture=fb_capture,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        effective_user_prompt=effective_user_prompt,
                        stage_budget=fb_inf_budget,
                    ),
                    fb_inf_budget,
                )
            elif eff_fallback_provider == "Gemini":
                fb_res = await run_coro_with_timeout(
                    _call_gemini_vision(
                        config.gemini_api_key,
                        eff_fallback_model or "gemini-3.7-flash",
                        image_bytes,
                        prompt,
                        history=history,
                        temperature=temperature,
                        request_context=request_context,
                        request_capture=fb_capture,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        effective_user_prompt=effective_user_prompt,
                        stage_budget=fb_stage_budget,
                    ),
                    fb_stage_budget,
                )
            elif eff_fallback_provider == "Claude":
                fb_res = await run_coro_with_timeout(
                    _call_claude_vision(
                        config.claude_api_key,
                        eff_fallback_model or "claude-sonnet-5",
                        image_bytes,
                        prompt,
                        history=history,
                        temperature=temperature,
                        request_context=request_context,
                        request_capture=fb_capture,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        effective_user_prompt=effective_user_prompt,
                        stage_budget=fb_stage_budget,
                    ),
                    fb_stage_budget,
                )
            else:
                fb_res = await run_coro_with_timeout(
                    _call_openai_vision(
                        config.openai_api_key or os.getenv("OPENAI_API_KEY"),
                        eff_fallback_model or "gpt-5.6-terra",
                        image_bytes,
                        prompt,
                        history=history,
                        temperature=temperature,
                        request_context=request_context,
                        request_capture=fb_capture,
                        request_layout=request_layout,
                        activity_tracker=activity_tracker,
                        effective_user_prompt=effective_user_prompt,
                        stage_budget=fb_stage_budget,
                    ),
                    fb_stage_budget,
                )

            fb_latency = int((time.monotonic() - fb_start) * 1000)
            if request_capture is not None:
                request_capture.update(fb_capture)
            clean_fb_res, _, _ = extract_service_data(fb_res)
            fb_diag = {"stage": "provider_fallback", "fallback_kind": "provider"}
            await _audit_attempt(
                eff_fallback_provider,
                eff_fallback_model or "",
                "fallback",
                "success",
                fb_latency,
                fb_capture,
                fb_res,
                clean_fb_res,
                diagnostics=fb_diag,
            )

            if execution_context is not None and execution_context.bot is not None:
                try:
                    await send_ai_fallback_used_alert(
                        bot=execution_context.bot,
                        platform=execution_context.platform,
                        user_id=execution_context.user_id,
                        chat_id=execution_context.chat_id,
                        bot_name=execution_context.bot_name,
                        primary_provider=primary_provider,
                        primary_model=primary_target_model,
                        fallback_provider=eff_fallback_provider,
                        fallback_model=eff_fallback_model or "",
                        primary_error=sanitize_vision_text(str(last_error)),
                        provider_attempts=provider_attempts_summary,
                        request_type="vision",
                    )
                except Exception as al_err:
                    logging.warning("Failed to send vision fallback alert: %s", al_err)

            return fb_res
        except Exception as fb_exc:
            fb_latency = int((time.monotonic() - fb_start) * 1000)
            fb_cls, _ = classify_external_error(fb_exc)
            last_error = fb_exc
            last_classification = fb_cls
            last_failed_provider = eff_fallback_provider
            last_failed_model = eff_fallback_model or ""
            fb_diag = {"stage": "provider_fallback", "fallback_kind": "provider"}
            await _audit_attempt(
                eff_fallback_provider,
                eff_fallback_model or "",
                "fallback",
                "error",
                fb_latency,
                fb_capture,
                None,
                None,
                err=fb_exc,
                classification=fb_cls,
                http_status=getattr(fb_exc, "http_status", None),
                finish_reason=getattr(fb_exc, "finish_reason", None),
                diagnostics=fb_diag,
                provider_resp_payload=getattr(fb_exc, "provider_response_payload", None),
            )

    if isinstance(last_error, AIServiceError):
        final_exc = last_error
    else:
        final_exc = AIServiceError(f"Не удалось выполнить анализ изображения: {last_error or 'Unknown error'}")
        if last_error is not None:
            final_exc.__cause__ = last_error
    attach_error_metadata(
        final_exc,
        classification=last_classification,
        provider=last_failed_provider,
        model=last_failed_model,
    )

    if execution_context is not None and execution_context.bot is not None:
        try:
            if last_classification == "output_budget_exhausted" and not fallback_attempted:
                await send_output_budget_exhausted_alert(
                    bot=execution_context.bot,
                    provider=last_failed_provider,
                    model=last_failed_model,
                    user_id=execution_context.user_id,
                    chat_id=execution_context.chat_id,
                    username=execution_context.username,
                    full_name=execution_context.full_name,
                    platform=execution_context.platform,
                    request_type="vision",
                )
            else:
                await send_terminal_ai_failure_alert(
                    bot=execution_context.bot,
                    platform=execution_context.platform,
                    user_id=execution_context.user_id,
                    chat_id=execution_context.chat_id,
                    username=execution_context.username,
                    full_name=execution_context.full_name,
                    provider=last_failed_provider,
                    model=last_failed_model,
                    stage="vision_orchestrator",
                    classification=last_classification or "provider_failure",
                    error_message=sanitize_vision_text(str(final_exc)),
                    request_type="vision",
                    attempts=provider_attempts_summary,
                )
            final_exc.admin_alert_handled = True
        except Exception as alert_err:
            logging.warning("Failed to send vision terminal alert: %s", alert_err)

    raise final_exc
