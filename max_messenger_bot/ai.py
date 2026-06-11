from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
import uuid

import anthropic
import httpx
import google.generativeai as genai  # noqa: F401
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .legacy import AIConfig, Message as DBMessage, Topic, User, async_session_maker
from .logging_utils import configure_logging, get_ai_logger
from memory_mode import MEMORY_MODE_TOPIC, build_history_scope, normalize_memory_mode
from prompt_injection import apply_global_prompt_appendix

configure_logging()
log = get_ai_logger("service")


class AIServiceError(RuntimeError):
    pass


class InsufficientBalanceError(AIServiceError):
    pass


def _build_max_history_scope(user: User, memory_mode: str):
    if memory_mode == MEMORY_MODE_TOPIC and user.current_topic_id is None:
        return (
            (DBMessage.user_id == user.id)
            & (DBMessage.dialogue_id == (user.current_dialogue_id or 1))
            & (DBMessage.topic_id.is_(None))
        )
    return build_history_scope(
        DBMessage,
        user.id,
        user.current_dialogue_id,
        user.current_topic_id,
        memory_mode,
    )


def _build_user_system_prompt(user: User, ai_config: AIConfig) -> str:
    system_prompt = user.current_topic.system_prompt if user.current_topic and user.current_topic.system_prompt else ai_config.system_prompt
    if not system_prompt:
        system_prompt = "Ты полезный ИИ-помощник."
    system_prompt = apply_global_prompt_appendix(system_prompt, getattr(ai_config, 'shared_prompt_block', None))

    safe_user_name = (getattr(user, "name", None) or getattr(user, "first_name", None) or "Не указано")
    safe_user_gender = getattr(user, "gender", None) or "Не указан"
    forced_user_header = f"ДАННЫЕ КЛИЕНТА:\nИМЯ: {safe_user_name}\nПОЛ: {safe_user_gender}\n"
    if getattr(user, "age", None):
        forced_user_header += f"ВОЗРАСТ: {user.age}\n"
    forced_user_header += "\n"
    try:
        formatted_body = system_prompt.format(user_name=safe_user_name, user_gender=safe_user_gender)
    except Exception:
        formatted_body = system_prompt
    final_prompt = forced_user_header.strip() + "\n\n" + formatted_body.strip()

    if getattr(user, "response_length", "normal") == "short":
        final_prompt += "\n\nОтвечай кратко, по делу, без длинных вступлений."
    return final_prompt


async def _call_openai(api_key: str, model: str, messages: list[dict], temperature: float) -> str:
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def _call_deepseek(api_key: str, model: str, messages: list[dict], temperature: float) -> str:
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def _call_claude(api_key: str, model: str, messages: list[dict], system_prompt: str, temperature: float) -> str:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    )
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


async def _call_gemini(api_key: str, model: str, messages: list[dict], system_prompt: str, temperature: float) -> str:
    import httpx

    contents = []
    for item in messages:
        role = "user" if item["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": item["content"]}]})
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }
    target_model = model or "gemini-2.5-flash"
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
    if status_code != 200:
        detail = payload.get("msg") or payload.get("message") or str(payload)
        raise AIServiceError(f"{context}: status={status_code} message={detail}")
    code = payload.get("code")
    if code not in (None, 200, "200"):
        detail = payload.get("msg") or payload.get("message") or str(payload)
        if any(word in str(detail).lower() for word in ["billing", "quota", "balance", "credit"]):
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
        raise AIServiceError(f"Ошибка загрузки файла в KIE: {e}")


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
        raise AIServiceError(f"Ошибка создания задачи KIE: {e}")


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


async def _call_kie_multimodal(api_key: str, base_url: str, model: str, system_prompt: str, user_content: list, temperature: float = 0.7) -> str:
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(
                f"{_kie_model_base_url(base_url, model)}/chat/completions",
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
        raise AIServiceError(f"Ошибка обращения к KIE multimodal API: {e}")


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
        )
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE transcription error", exc_info=e)
        raise AIServiceError(f"Ошибка при транскрибации (KIE API): {e}")


async def _analyze_kie(api_key: str, base_url: str, upload_base_url: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float = 0.7) -> str:
    try:
        file_url = await _upload_file_to_kie(
            api_key, upload_base_url, image_bytes,
            _guess_filename(image_bytes, "vision_input", "jpg"), "images",
        )
        return await _call_kie_multimodal(
            api_key, base_url, model,
            system_prompt,
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": file_url}},
            ],
            temperature=temperature,
        )
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE vision error", exc_info=e)
        raise AIServiceError(f"Ошибка анализа изображения (KIE): {e}")


async def _generate_kie(api_key: str, base_url: str, model: str, prompt: str) -> bytes:
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


async def _call_kie_text_chat(api_key: str, base_url: str, model: str, messages: list[dict], system_prompt: str, temperature: float) -> str:
    """Call KIE API for text chat (OpenAI-compatible endpoint)."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "max_tokens": 4096,
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(
                f"{_kie_model_base_url(base_url, model)}/chat/completions",
                headers=headers,
                json=payload,
            )
        response_payload = _validate_kie_json_response(
            response.status_code,
            response.json(),
            context="Ошибка при обращении к KIE Chat API",
        )
        text = _extract_kie_chat_text(response_payload)
        if not text:
            raise AIServiceError("KIE chat returned empty content")
        return text
    except (AIServiceError, InsufficientBalanceError):
        raise
    except Exception as e:
        log.error("KIE chat error: %s", e, exc_info=True)
        raise AIServiceError(f"Ошибка при обращении к KIE Chat API: {e}") from e


def _resolve_provider(ai_config: AIConfig) -> tuple[str, float]:
    provider = (ai_config.provider or "").strip().lower()
    temperature = getattr(ai_config, "temperature", 0.7) or 0.7
    return provider, temperature


async def _dispatch_provider(ai_config: AIConfig, system_prompt: str, messages: list[dict]) -> str:
    provider, temperature = _resolve_provider(ai_config)

    if provider == "openai":
        if not ai_config.openai_api_key:
            raise AIServiceError("OpenAI API key не задан")
        return await _call_openai(ai_config.openai_api_key, ai_config.openai_model, [{"role": "system", "content": system_prompt}, *messages], temperature)
    if provider in {"claude", "anthropic"}:
        if not ai_config.claude_api_key:
            raise AIServiceError("Claude API key не задан")
        return await _call_claude(ai_config.claude_api_key, ai_config.claude_model, messages, system_prompt, temperature)
    if provider == "gemini":
        if not ai_config.gemini_api_key:
            raise AIServiceError("Gemini API key не задан")
        return await _call_gemini(ai_config.gemini_api_key, ai_config.gemini_model, messages, system_prompt, temperature)
    if provider == "deepseek":
        if not ai_config.deepseek_api_key:
            raise AIServiceError("DeepSeek API key не задан")
        return await _call_deepseek(ai_config.deepseek_api_key, ai_config.deepseek_model, [{"role": "system", "content": system_prompt}, *messages], temperature)
    if provider == "kie":
        if not ai_config.kie_api_key:
            raise AIServiceError("KIE API key не задан")
        base_url = _get_kie_base_url(ai_config)
        return await _call_kie_text_chat(ai_config.kie_api_key, base_url, ai_config.kie_model or "gemini-3-flash", messages, system_prompt, temperature)
    raise AIServiceError(f"Неподдерживаемый провайдер ИИ: {ai_config.provider}")


async def get_ai_response(user_id: int, user_prompt: str) -> str:
    async with async_session_maker() as session:
        user = await session.scalar(
            select(User)
            .options(selectinload(User.current_topic))
            .where(User.id == user_id)
        )
        if not user:
            raise AIServiceError("Пользователь не найден")

        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            raise AIServiceError("AIConfig не найден")

        system_prompt = _build_user_system_prompt(user, ai_config)

        current_memory_mode = normalize_memory_mode(ai_config)
        history_scope = _build_max_history_scope(user, current_memory_mode)
        history_rows = (
            await session.execute(
                select(DBMessage)
                .where(history_scope)
                .order_by(DBMessage.timestamp.asc())
            )
        ).scalars().all()

        limit_first = getattr(ai_config, "context_limit_first", 2) or 2
        limit_recent = getattr(ai_config, "context_limit_recent", 10) or 10
        if len(history_rows) > limit_first + limit_recent:
            history_rows = history_rows[:limit_first] + history_rows[-limit_recent:]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": row.role, "content": row.content} for row in history_rows if row.content)
        messages.append({"role": "user", "content": user_prompt})

        stripped = [item for item in messages if item["role"] != "system"]
        temperature = getattr(ai_config, "temperature", 0.7) or 0.7
        try:
            result = await _dispatch_provider(ai_config, system_prompt, stripped)
            log.info("AI response generated user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, user.current_topic_id)
            return result
        except InsufficientBalanceError:
            raise
        except (AIServiceError, Exception) as primary_err:
            # Try fallback provider if configured
            fb_provider = getattr(ai_config, "fallback_provider", None)
            fb_model = getattr(ai_config, "fallback_model", None)
            allow_fallback = getattr(ai_config, "allow_fallback", False)
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
                            result = await _call_openai(fb_api_key, fb_model, [{"role": "system", "content": system_prompt}, *stripped], temperature)
                        elif fb_key in {"claude", "anthropic"}:
                            result = await _call_claude(fb_api_key, fb_model, stripped, system_prompt, temperature)
                        elif fb_key == "gemini":
                            result = await _call_gemini(fb_api_key, fb_model, stripped, system_prompt, temperature)
                        elif fb_key == "deepseek":
                            result = await _call_deepseek(fb_api_key, fb_model, [{"role": "system", "content": system_prompt}, *stripped], temperature)
                        elif fb_key == "kie":
                            result = await _call_kie_text_chat(fb_api_key, _get_kie_base_url(ai_config), fb_model, stripped, system_prompt, temperature)
                        else:
                            raise AIServiceError(f"Неизвестный фолбэк провайдер: {fb_provider}")
                        log.info("Fallback response generated user_id=%s provider=%s", user_id, fb_provider)
                        return result
                    except Exception as fb_err:
                        log.error("Fallback provider '%s' also failed: %s", fb_provider, fb_err)
                        raise AIServiceError(
                            f"Основной провайдер ({ai_config.provider}) и резервный ({fb_provider}) недоступны"
                        ) from fb_err
            if isinstance(primary_err, AIServiceError):
                log.exception("AI request failed user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, user.current_topic_id)
                raise
            log.exception("Unexpected AI request failure user_id=%s provider=%s topic_id=%s", user_id, ai_config.provider, user.current_topic_id)
            raise AIServiceError(f"Ошибка при обращении к AI-провайдеру: {primary_err}") from primary_err


async def get_ai_response_direct(user_id: int, system_prompt: str, user_prompt: str) -> str:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            raise AIServiceError("Пользователь не найден")
        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            raise AIServiceError("AIConfig не найден")

    prompt = apply_global_prompt_appendix(system_prompt or ai_config.system_prompt or "Ты полезный ИИ-помощник.", getattr(ai_config, 'shared_prompt_block', None))
    if getattr(user, "response_length", "normal") == "short":
        prompt += "\n\nОтвечай кратко, по делу, без длинных вступлений."
    messages = [{"role": "user", "content": user_prompt}]
    try:
        result = await _dispatch_provider(ai_config, prompt, messages)
        log.info("AI direct response generated user_id=%s provider=%s", user_id, ai_config.provider)
        return result
    except AIServiceError:
        log.exception("AI direct request failed user_id=%s provider=%s", user_id, ai_config.provider)
        raise
    except Exception as exc:
        log.exception("Unexpected AI direct request failure user_id=%s provider=%s", user_id, ai_config.provider)
        raise AIServiceError(f"Ошибка при прямом обращении к AI-провайдеру: {exc}") from exc


# ---------------------------------------------------------------------------
# Voice Transcription
# ---------------------------------------------------------------------------

async def _transcribe_openai(api_key: str, file_bytes: bytes, filename: str) -> str:
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
    transcription = await client.audio.transcriptions.create(model="whisper-1", file=(filename, file_bytes))
    return transcription.text


async def _transcribe_gemini(api_key: str, model: str, file_bytes: bytes, filename: str) -> str:
    import httpx

    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type or not mime_type.startswith("audio/"):
        mime_type = "audio/ogg"
    b64_data = base64.b64encode(file_bytes).decode()
    target_model = model or "gemini-2.5-flash"
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
        return await _transcribe_gemini(api_key, config.gemini_model or "gemini-2.5-flash", file_bytes, filename)
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
            return await _transcribe_gemini(
                config.gemini_api_key,
                config.gemini_model or "gemini-2.5-flash",
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

async def _analyze_gemini(api_key: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float) -> str:
    import httpx

    b64_data = base64.b64encode(image_bytes).decode()
    target_model = model or "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}},
            ]
        }],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }
    async with httpx.AsyncClient(timeout=60.0, transport=_build_gemini_proxy_transport()) as http:
        resp = await http.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise AIServiceError("Gemini vision returned empty candidates")
    return candidates[0]["content"]["parts"][0]["text"]


async def _analyze_openai(api_key: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float) -> str:
    b64_data = base64.b64encode(image_bytes).decode()
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
    response = await client.chat.completions.create(
        model=model or "gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
                ],
            },
        ],
        max_tokens=4096,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def _analyze_claude(api_key: str, model: str, image_bytes: bytes, system_prompt: str, prompt: str, temperature: float) -> str:
    b64_data = base64.b64encode(image_bytes).decode()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model or "claude-sonnet-4-5-20250929",
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
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
    temperature = getattr(config, "temperature", 0.7) or 0.7
    system_prompt = _build_user_system_prompt(user, config)

    if provider == "Gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для vision не задан")
        return await _analyze_gemini(api_key, config.vision_model or "gemini-2.0-flash", image_bytes, system_prompt, prompt, temperature)
    if provider in {"Claude", "Anthropic"}:
        api_key = config.claude_api_key
        if not api_key:
            raise AIServiceError("API ключ Claude для vision не задан")
        return await _analyze_claude(api_key, config.vision_model or "claude-sonnet-4-5-20250929", image_bytes, system_prompt, prompt, temperature)
    if provider == "KIE":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для vision не задан")
        model = config.vision_model or "google/gemini-2.5-pro"
        return await _analyze_kie(api_key, _get_kie_base_url(config), _get_kie_upload_base_url(config), model, image_bytes, system_prompt, prompt, temperature)
    # Default: OpenAI
    api_key = config.openai_api_key
    if not api_key:
        raise AIServiceError("API ключ OpenAI для vision не задан")
    return await _analyze_openai(api_key, config.vision_model or "gpt-4o", image_bytes, system_prompt, prompt, temperature)


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------

async def _generate_gemini(api_key: str, model: str, prompt: str) -> bytes:
    import httpx

    target_model = model or "imagen-4.0-generate-001"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:predict?key={api_key}"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": "1:1"},
    }
    async with httpx.AsyncClient(timeout=90.0, transport=_build_gemini_proxy_transport()) as http:
        resp = await http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    predictions = data.get("predictions", [])
    if not predictions:
        raise AIServiceError("Imagen вернул пустой ответ")
    img_b64 = predictions[0].get("bytesBase64Encoded")
    if not img_b64:
        raise AIServiceError("Imagen не вернул данные изображения")
    return base64.b64decode(img_b64)


async def _generate_openai(api_key: str, prompt: str) -> bytes:
    import httpx

    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1"))
    response = await client.images.generate(model="gpt-image-1.5", prompt=prompt, n=1, size="1024x1024")
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

    provider = getattr(config, "image_generation_provider", None) or config.vision_provider or "Gemini"
    provider = provider.strip()

    if provider == "Gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для генерации не задан")
        model = getattr(config, "image_generation_model", None) or "imagen-4.0-generate-001"
        return await _generate_gemini(api_key, model, prompt)
    if provider == "KIE":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для генерации не задан")
        model = getattr(config, "image_generation_model", None) or "google/imagen4-fast"
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
    import httpx

    b64_data = base64.b64encode(image_bytes).decode()
    target_model = model or "gemini-3-pro-image-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise AIServiceError("Gemini image edit returned no candidates")
    for part in candidates[0]["content"]["parts"]:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            return base64.b64decode(inline["data"])
    raise AIServiceError("Gemini image edit returned no image in response")


async def edit_image(prompt: str, image_bytes: bytes) -> bytes:
    """Edit image using the configured provider."""
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
    if not config:
        raise AIServiceError("AIConfig не найден")

    provider = getattr(config, "image_edit_provider", None) or config.vision_provider or "Gemini"
    provider = provider.strip()

    if provider == "Gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise AIServiceError("API ключ Gemini для редактирования не задан")
        model = getattr(config, "image_edit_model", None) or "gemini-3-pro-image-preview"
        return await _edit_gemini(api_key, model, prompt, image_bytes)
    if provider == "KIE":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise AIServiceError("API ключ KIE для редактирования не задан")
        model = getattr(config, "image_edit_model", None) or "google/nano-banana-edit"
        return await _edit_kie(api_key, _get_kie_base_url(config), _get_kie_upload_base_url(config), model, prompt, image_bytes)
    raise AIServiceError(f"Редактирование изображений не поддерживается для провайдера: {provider}")
