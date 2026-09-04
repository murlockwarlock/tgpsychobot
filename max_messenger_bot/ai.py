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
import uuid

import anthropic
import httpx
import google.generativeai as genai  # noqa: F401
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import gemini_image

from ai_log_context import apply_ai_log_context
from .legacy import AIConfig, KnowledgeBase, Message as DBMessage, Topic, User, async_session_maker
from .legacy import AILog
from .logging_utils import configure_logging, get_ai_logger
from automation_engine import apply_service_data_blocks, build_runtime_automation_context
from user_metadata import extract_service_data
from memory_mode import MEMORY_MODE_TOPIC, build_history_scope, normalize_memory_mode
from result_history import ai_history_role_filter, select_ai_history_messages
from error_reporting import classify_ai_error, exception_summary
from vector_store import search_relevant_chunks
from provider_models import (
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    PROVIDER_CLAUDE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    ensure_model_available,
    get_default_model,
    is_retired_model,
    normalize_deepseek_model,
    should_omit_claude_sampling,
)
from ai_request_context import (
    AIRequestLayout,
    build_anthropic_system,
    build_gemini_contents,
    build_gemini_system_parts,
    build_openai_chat_messages,
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
        raise AIResponseError(f"{provider} returned an empty or invalid text response")
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
    if model:
        return str(model)
    try:
        model = get_default_model(provider_key, channel="chat")
    except Exception:
        model = None
    return str(model or "—")


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
) -> str:
    target_model = model or "gpt-5.6-terra"
    ensure_model_available(PROVIDER_OPENAI, target_model)
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
    payload: dict = {
        "model": target_model,
        "messages": build_openai_chat_messages(request_layout or _legacy_layout(messages)),
        "max_completion_tokens": 4096,
    }
    if not target_model.startswith("gpt-5.6"):
        payload["temperature"] = temperature
    response = await client.chat.completions.create(**payload)
    return response.choices[0].message.content or ""


async def _call_deepseek(
    api_key: str,
    model: str,
    messages: list[dict] | None,
    temperature: float,
    *,
    request_layout: AIRequestLayout | None = None,
) -> str:
    normalized_model = normalize_deepseek_model(model)
    ensure_model_available(PROVIDER_DEEPSEEK, normalized_model)
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    response = await client.chat.completions.create(
        model=normalized_model,
        messages=build_openai_chat_messages(request_layout or _legacy_layout(messages)),
        max_tokens=4096,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def _call_claude(
    api_key: str,
    model: str,
    messages: list[dict] | None,
    system_prompt: str,
    temperature: float,
    *,
    request_layout: AIRequestLayout | None = None,
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
        "max_tokens": 4096,
        "system": build_anthropic_system(layout),
        "messages": anthropic_messages,
    }
    if not should_omit_claude_sampling(target_model):
        payload["temperature"] = temperature
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
) -> str:
    import httpx

    target_model = model or "gemini-3.7-flash"
    ensure_model_available(PROVIDER_GEMINI, target_model)
    layout = request_layout or _legacy_layout(messages, system_prompt)
    generation_config: dict = {"maxOutputTokens": 4096}
    if not (target_model.startswith("gemini-3.7") or target_model.startswith("gemini-3.6")):
        generation_config["temperature"] = temperature

    payload = {
        "contents": build_gemini_contents(layout),
        "systemInstruction": {
            "parts": build_gemini_system_parts(layout) or [{"text": ""}],
        },
        "generationConfig": generation_config,
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    transport = _build_gemini_proxy_transport()
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
    if status_code != 200:
        if is_kie_insufficient_balance(status_code, payload):
            raise InsufficientBalanceError(f"KIE API Error: {detail}")
        raise AIServiceError(f"{context}: status={status_code} message={detail}")
    code = payload.get("code")
    if code not in (None, 200, "200"):
        if is_kie_insufficient_balance(status_code, payload):
            raise InsufficientBalanceError(f"KIE API Error: {detail}")
        raise AIServiceError(f"{context}: {detail}")
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


async def _upload_file_to_kie(api_key: str, upload_base_url: str, file_bytes: bytes, filename: str, upload_path: str) -> str:
    url = f"{upload_base_url}/api/file-stream-upload"
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    form_data = {"uploadPath": upload_path, "fileName": filename}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(url, headers=headers, data=form_data, files=files)
        payload = response.json()
        data_payload = _validate_kie_json_response(response.status_code, payload, context="KIE upload failed")
        file_url = data_payload.get("downloadUrl") or data_payload.get("fileUrl")
        if not file_url:
            raise AIServiceError(f"KIE upload returned no file URL: {payload}")
        return file_url
    except (AIServiceError, InsufficientBalanceError):
        raise
    except Exception as e:
        logging.error("KIE upload error", exc_info=e)
        raise AIServiceError(f"Ошибка загрузки файла в KIE: {exception_summary(e)}") from e


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


async def _analyze_kie(api_key: str, base_url: str, upload_base_url: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float = 0.7, history: list = None, shared_instructions: tuple[str, ...] = (), request_layout: AIRequestLayout | None = None) -> str:
    ensure_model_available(PROVIDER_KIE, model, channel="vision")
    try:
        file_url = await _upload_file_to_kie(
            api_key, upload_base_url, image_bytes,
            _guess_filename(image_bytes, "vision_input", "jpg"), "images",
        )
        layout = request_layout or AIRequestLayout(
            stable_system_prompt=system_prompt,
            shared_instructions=shared_instructions,
            history=normalize_request_messages(history),
        )
        layout = layout.with_current_user_content([
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": file_url}},
            ])
        return await _call_kie_multimodal(
            api_key, base_url, model,
            system_prompt,
            layout.current_user_content,
            temperature=temperature,
            channel="vision",
            request_layout=layout,
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
    try:
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
) -> str:
    provider, temperature = _resolve_provider(ai_config)
    layout = (
        request_layout
        if isinstance(request_layout, AIRequestLayout)
        else _legacy_layout(messages, request_layout)
    )

    if provider == "openai":
        if not ai_config.openai_api_key:
            raise AIServiceError("OpenAI API key не задан")
        result = await _call_openai(
            ai_config.openai_api_key,
            ai_config.openai_model,
            [],
            temperature,
            request_layout=layout,
        )
    elif provider in {"claude", "anthropic"}:
        if not ai_config.claude_api_key:
            raise AIServiceError("Claude API key не задан")
        result = await _call_claude(
            ai_config.claude_api_key,
            ai_config.claude_model,
            [],
            layout.stable_system_prompt,
            temperature,
            request_layout=layout,
        )
    elif provider == "gemini":
        if not ai_config.gemini_api_key:
            raise AIServiceError("Gemini API key не задан")
        result = await _call_gemini(
            ai_config.gemini_api_key,
            ai_config.gemini_model,
            [],
            layout.stable_system_prompt,
            temperature,
            request_layout=layout,
        )
    elif provider == "deepseek":
        if not ai_config.deepseek_api_key:
            raise AIServiceError("DeepSeek API key не задан")
        result = await _call_deepseek(
            ai_config.deepseek_api_key,
            ai_config.deepseek_model,
            [],
            temperature,
            request_layout=layout,
        )
    elif provider == "kie":
        if not ai_config.kie_api_key:
            raise AIServiceError("KIE API key не задан")
        base_url = _get_kie_base_url(ai_config)
        result = await _call_kie_text_chat(
            ai_config.kie_api_key,
            base_url,
            ai_config.kie_model or "gemini-3-flash",
            [],
            layout.stable_system_prompt,
            temperature,
            request_layout=layout,
        )
    else:
        raise AIServiceError(f"Неподдерживаемый провайдер ИИ: {ai_config.provider}")
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
) -> str:
    async with async_session_maker() as session:
        user = await session.scalar(
            select(User)
            .options(selectinload(User.current_topic).selectinload(Topic.knowledge_base_files))
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
        shared_instructions = tuple(
            block
            for block in ((getattr(ai_config, "shared_prompt_block", None) or "").strip(),)
            if block
        )

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

        current_memory_mode = normalize_memory_mode(ai_config)
        history_scope = _build_max_history_scope(user, current_memory_mode, active_topic_id, active_dialogue_id)
        history_rows = (
            await session.execute(
                select(DBMessage)
                .options(selectinload(DBMessage.topic))
                .where(history_scope, ai_history_role_filter(DBMessage))
                .order_by(DBMessage.timestamp.asc())
            )
        ).scalars().all()

        limit_first = getattr(ai_config, "context_limit_first", 2) or 2
        limit_recent = getattr(ai_config, "context_limit_recent", 10) or 10

        history_rows = select_ai_history_messages(history_rows, limit_first, limit_recent)
        history_messages = [
            {"role": row.role, "content": row.content}
            for row in history_rows
            if row.content
        ]
        if history_messages and history_messages[-1]["role"] == "user" and history_messages[-1]["content"] == user_prompt:
            history_messages.pop()

        runtime_parts = [_build_client_runtime_context(user)]
        if getattr(user, "response_length", "normal") == "short":
            runtime_parts.append("Отвечай кратко, по делу, без длинных вступлений.")
        scenario_context = await build_runtime_automation_context(
            session,
            user_id=user.id,
            dialogue_id=active_dialogue_id,
            topic_id=active_topic_id,
        )
        request_context = (context,) if context else ()
        request_layout = AIRequestLayout(
            stable_system_prompt=stable_system_prompt,
            shared_instructions=shared_instructions,
            runtime_context=tuple(runtime_parts),
            scenario_context=(scenario_context,) if scenario_context else (),
            request_context=request_context,
            history=normalize_request_messages(history_messages),
            current_user_content=user_prompt,
        )
        temperature = _resolve_temperature(ai_config)
        start_time = time.monotonic()
        try:
            result = await _dispatch_provider(ai_config, request_layout)
            log.info("AI response generated user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, active_topic_id)
        except (AIServiceError, Exception) as primary_err:
            # Try fallback provider if configured
            fb_provider = getattr(ai_config, "fallback_provider", None)
            fb_model = getattr(ai_config, "fallback_model", None)
            allow_fallback = getattr(ai_config, "allow_fallback", False)
            fallback_succeeded = False
            if allow_fallback and fb_provider and fb_model:
                fb_key = fb_provider.strip().lower()
                if fb_key in {"claude", "anthropic"}:
                    fb_api_key = ai_config.claude_api_key
                else:
                    fb_api_key = getattr(ai_config, f"{fb_key}_api_key", None)
                if fb_api_key:
                    log.warning("Primary provider '%s' failed (%s), falling back to '%s'", ai_config.provider, primary_err, fb_provider)
                    try:
                        if fb_key == "openai":
                            result = await _call_openai(
                                fb_api_key, fb_model, [], temperature,
                                request_layout=request_layout,
                            )
                        elif fb_key in {"claude", "anthropic"}:
                            result = await _call_claude(
                                fb_api_key, fb_model, [], stable_system_prompt, temperature,
                                request_layout=request_layout,
                            )
                        elif fb_key == "gemini":
                            result = await _call_gemini(
                                fb_api_key, fb_model, [], stable_system_prompt, temperature,
                                request_layout=request_layout,
                            )
                        elif fb_key == "deepseek":
                            result = await _call_deepseek(
                                fb_api_key, fb_model, [], temperature,
                                request_layout=request_layout,
                            )
                        elif fb_key == "kie":
                            result = await _call_kie_text_chat(
                                fb_api_key, _get_kie_base_url(ai_config), fb_model, [],
                                stable_system_prompt, temperature,
                                request_layout=request_layout,
                            )
                        else:
                            raise AIServiceError(f"Неизвестный фолбэк провайдер: {fb_provider}")
                        result = _validate_text_response(result, provider=fb_key)
                        actual_provider = str(fb_provider)
                        actual_model = str(fb_model)
                        fallback_succeeded = True
                        log.info("Fallback response generated user_id=%s provider=%s", user_id, fb_provider)
                    except Exception as fb_err:
                        log.error("Fallback provider '%s' also failed: %s", fb_provider, fb_err)
                        raise AIServiceError(
                            f"Основной провайдер ({ai_config.provider}) и резервный ({fb_provider}) недоступны"
                        ) from fb_err
            if not fallback_succeeded:
                if isinstance(primary_err, AIServiceError):
                    log.exception("AI request failed user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, user.current_topic_id)
                    raise
                log.exception("Unexpected AI request failure user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, user.current_topic_id)
                raise AIServiceError(f"Ошибка при обращении к AI-провайдеру: {primary_err}") from primary_err

        latency_ms = int((time.monotonic() - start_time) * 1000)
        visible_text, _, _ = extract_service_data(result)
        ai_log = AILog(
            user_id=user_id,
            request_type="chat",
            provider=actual_provider,
            model=actual_model,
            prompt_summary=user_prompt if user_prompt else None,
            raw_response=result,
            clean_text=visible_text,
            latency_ms=latency_ms,
        )
        apply_ai_log_context(
            ai_log,
            platform="max",
            topic_id=active_topic_id,
            topic_name=active_topic.name if active_topic else None,
        )
        try:
            session.add(ai_log)
            await session.commit()
        except Exception:
            log.exception("Could not save shared AI log for user %s", user_id)
        return result


async def get_ai_response_direct(
    user_id: int,
    system_prompt: str,
    user_prompt: str,
    *,
    dialogue_id: int | None = None,
    topic_id: int | None = None,
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
        runtime_context = await build_runtime_automation_context(
            session,
            user_id=user.id,
            dialogue_id=active_dialogue_id,
            topic_id=active_topic_id,
        )

    base_system = neutralize_stable_prompt(
        system_prompt or ai_config.system_prompt or "Ты полезный ИИ-помощник."
    )
    runtime_parts = [_build_client_runtime_context(user)]
    if getattr(user, "response_length", "normal") == "short":
        runtime_parts.append("Отвечай кратко, по делу, без длинных вступлений.")
    request_layout = AIRequestLayout(
        stable_system_prompt=base_system,
        shared_instructions=tuple(
            block
            for block in ((getattr(ai_config, "shared_prompt_block", None) or "").strip(),)
            if block
        ),
        runtime_context=tuple(runtime_parts),
        scenario_context=(runtime_context,) if runtime_context and runtime_context.strip() else (),
        current_user_content=user_prompt,
    )
    try:
        result = await _dispatch_provider(ai_config, request_layout)
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
        async with async_session_maker() as session:
            user = await session.get(User, user_id)
            if user:
                await apply_service_data_blocks(
                    session,
                    user=user,
                    dialogue_id=active_dialogue_id,
                    topic_id=active_topic_id,
                    blocks=service_blocks,
                )
                await session.commit()
    return visible_text or result


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

async def _analyze_gemini(api_key: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float, history: list = None, shared_instructions: tuple[str, ...] = (), request_layout: AIRequestLayout | None = None) -> str:
    import httpx

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
    generation_config: dict = {"maxOutputTokens": 4096}
    if not (target_model.startswith("gemini-3.7") or target_model.startswith("gemini-3.6")):
        generation_config["temperature"] = temperature

    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": build_gemini_system_parts(layout) or [{"text": ""}],
        },
        "generationConfig": generation_config,
    }
    async with httpx.AsyncClient(timeout=60.0, transport=_build_gemini_proxy_transport()) as http:
        resp = await http.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise AIServiceError("Gemini vision returned empty candidates")
    return candidates[0]["content"]["parts"][0]["text"]


async def _analyze_openai(api_key: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float, history: list = None, shared_instructions: tuple[str, ...] = (), request_layout: AIRequestLayout | None = None) -> str:
    target_model = model or "gpt-5.6-terra"
    ensure_model_available(PROVIDER_OPENAI, target_model, channel="vision")
    b64_data = base64.b64encode(image_bytes).decode()
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
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
        "max_completion_tokens": 4096,
    }
    if not target_model.startswith("gpt-5.6"):
        payload["temperature"] = temperature
    response = await client.chat.completions.create(**payload)
    return response.choices[0].message.content or ""


async def _analyze_claude(api_key: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float, history: list = None, shared_instructions: tuple[str, ...] = (), request_layout: AIRequestLayout | None = None) -> str:
    target_model = model or "claude-sonnet-5"
    ensure_model_available(PROVIDER_CLAUDE, target_model, channel="vision")
    b64_data = base64.b64encode(image_bytes).decode()
    client = anthropic.AsyncAnthropic(api_key=api_key)
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
        "max_tokens": 4096,
        "system": build_anthropic_system(layout),
        "messages": claude_messages,
    }
    if not should_omit_claude_sampling(target_model):
        payload["temperature"] = temperature
    response = await client.messages.create(**payload)
    return response.content[0].text


async def analyze_image(user_id: int, image_bytes: bytes, prompt: str) -> str:
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

        provider = (config.vision_provider or "Gemini").strip()
        temperature = _resolve_temperature(config)
        
        photo_instructions = (
            "\n\nИНСТРУКЦИЯ ПО АНАЛИЗУ ФОТО:\n"
            "1. Если пользователь просит ИЗМЕНИТЬ это фото или 'сделать так же', добавь в конце: EDIT_IMG: <prompt on english>.\n"
            "2. Если нужно создать НОВОЕ фото с нуля, добавь в конце: GEN_IMG: <prompt on english>.\n"
            "3. ВАЖНО: Диалог уже начат. НЕ здоровайся, не представляйся и не используй вежливые вступления. Сразу переходи к сути разбора изображения."
        )
        system_prompt = _build_user_system_prompt(user, config)
        shared_instructions = tuple(
            block
            for block in (
                (getattr(config, "shared_prompt_block", None) or "").strip(),
                photo_instructions.strip(),
            )
            if block
        )

        current_memory_mode = normalize_memory_mode(config)
        history_scope = _build_max_history_scope(user, current_memory_mode)
        history_rows = (
            await session.execute(
                select(DBMessage)
                .options(selectinload(DBMessage.topic))
                .where(history_scope, ai_history_role_filter(DBMessage))
                .order_by(DBMessage.timestamp.asc())
            )
        ).scalars().all()

        limit_first = getattr(config, "context_limit_first", 2) or 2
        limit_recent = getattr(config, "context_limit_recent", 10) or 10

        # Filter out the message we just saved before calling this function, which ends with role == 'user' and starts with "[Изображение]"
        if history_rows and history_rows[-1].role == "user" and history_rows[-1].content.startswith("[Изображение]"):
            history_rows = history_rows[:-1]

        history_rows = select_ai_history_messages(history_rows, limit_first, limit_recent)

        history_list = [{"role": row.role, "content": row.content} for row in history_rows if row.content]
        runtime_parts = [_build_client_runtime_context(user)]
        if getattr(user, "response_length", "normal") == "short":
            runtime_parts.append("Отвечай кратко, по делу, без длинных вступлений.")
        scenario_context = await build_runtime_automation_context(
            session,
            user_id=user.id,
            dialogue_id=user.current_dialogue_id,
            topic_id=user.current_topic_id,
        )
        request_layout = AIRequestLayout(
            stable_system_prompt=system_prompt,
            shared_instructions=shared_instructions,
            runtime_context=tuple(runtime_parts),
            scenario_context=(scenario_context,) if scenario_context else (),
            history=normalize_request_messages(history_list),
        )

    if provider == "Gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для vision не задан")
        return await _analyze_gemini(api_key, config.vision_model or "gemini-3.7-flash", image_bytes, system_prompt, prompt, temperature, history=history_list, shared_instructions=shared_instructions, request_layout=request_layout)
    if provider in {"Claude", "Anthropic"}:
        api_key = config.claude_api_key
        if not api_key:
            raise AIServiceError("API ключ Claude для vision не задан")
        return await _analyze_claude(api_key, config.vision_model or "claude-sonnet-5", image_bytes, system_prompt, prompt, temperature, history=history_list, shared_instructions=shared_instructions, request_layout=request_layout)
    if provider == "KIE":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для vision не задан")
        model = config.vision_model or "gemini-3-flash"
        return await _analyze_kie(api_key, _get_kie_base_url(config), _get_kie_upload_base_url(config), model, image_bytes, system_prompt, prompt, temperature, history=history_list, shared_instructions=shared_instructions, request_layout=request_layout)
    # Default: OpenAI
    api_key = config.openai_api_key
    if not api_key:
        raise AIServiceError("API ключ OpenAI для vision не задан")
    return await _analyze_openai(api_key, config.vision_model or "gpt-5.6-terra", image_bytes, system_prompt, prompt, temperature, history=history_list, shared_instructions=shared_instructions, request_layout=request_layout)


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
