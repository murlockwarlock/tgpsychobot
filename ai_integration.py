import logging
import google.generativeai as genai
import anthropic
import os
import io
import mimetypes
import base64
import asyncio
import json
import httpx
from sqlalchemy import select, or_
from sqlalchemy.orm import selectinload
from openai import AsyncOpenAI, AuthenticationError, RateLimitError, BadRequestError

from database import (async_session_maker, AIConfig, Message as DBMessage, User, Topic, TestSession,
                     MediaLibrary, TopicMediaDeck, MediaCollection, media_collection_items, topic_collection_association)
from memory_mode import get_memory_mode, is_global_memory_mode
from prompt_blocks import (
    DEFAULT_SERVICE_PROMPT_TEMPLATE,
    DEFAULT_SHORT_RESPONSE_INSTRUCTION,
    render_prompt_block,
)
from vector_store import search_relevant_chunks

class InsufficientBalanceError(Exception):
    pass


class AIServiceError(Exception):
    """Transient AI provider error (network, 5xx, etc.) — show friendly message to user."""
    pass


class AIResponseError(AIServiceError):
    """Provider returned an invalid or empty payload."""
    pass


def _build_async_transport_from_env(env_var_name: str):
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


def _get_kie_base_url(ai_config: AIConfig) -> str:
    return (getattr(ai_config, "kie_base_url", None) or "https://api.kie.ai").rstrip("/")


def _get_kie_upload_base_url(ai_config: AIConfig) -> str:
    return (getattr(ai_config, "kie_upload_base_url", None) or "https://kieai.redpandaai.co").rstrip("/")


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

    return f"{fallback_stem}.{ext}"


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


def _is_kie_transient_failure(error: Exception | str) -> bool:
    text = str(error).lower()
    markers = [
        "maintained",
        "maintenance",
        "internal error",
        "try again later",
        "server is currently being maintained",
    ]
    return any(marker in text for marker in markers)


def _validate_kie_json_response(status_code: int, payload: dict, *, context: str) -> dict:
    if status_code != 200:
        detail = payload.get("msg") or payload.get("message") or str(payload)
        raise AIServiceError(f"{context}: status={status_code} message={detail}")

    code = payload.get("code")
    if code not in (None, 200, "200"):
        detail = payload.get("msg") or payload.get("message") or str(payload)
        lowered = str(detail).lower()
        if any(word in lowered for word in ["billing", "quota", "balance", "credit"]):
            raise InsufficientBalanceError(f"KIE API Error: {detail}")
        raise AIServiceError(f"{context}: {detail}")

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


async def generate_response(user_id: int, user_prompt: str) -> str:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            return "Ошибка: Пользователь не найден."

        user_name = user.name if user.name else "Незнакомец"
        user_gender = user.gender if user.gender else "unknown"

    return await get_ai_response(user_id, user_prompt, user_name, user_gender)


async def _call_gemini_api(api_key: str, model: str, history: list, context: str, system_prompt: str, temperature: float = 0.7) -> str:
    import httpx
    try:
        transport = _build_async_transport_from_env("GEMINI_PROXY")
        contents = []
        for msg in history:
            if not msg.content: continue
            role = 'user' if msg.role == 'user' else 'model'
            contents.append({'role': role, 'parts': [{'text': msg.content}]})
        if not contents or contents[-1]['role'] != 'user':
            return "Ошибка: История диалога должна заканчиваться сообщением пользователя."
        full_system_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}"
        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": full_system_prompt}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 4096,
            }
        }
        target_model = model if model else "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=60.0) as client:
            response = await client.post(url, json=payload, headers={'Content-Type': 'application/json'})
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(response.text))
                if "location" in error_msg.lower():
                    raise InsufficientBalanceError(f"Geo-Block: {error_msg}")
                raise AIServiceError(f"Ошибка API Gemini: {error_msg}")
            data = response.json()
            candidates = data.get('candidates', [])
            if not candidates:
                return "Ошибка: Gemini вернул пустой ответ или контент заблокирован."
            return candidates[0]['content']['parts'][0]['text']
    except Exception as e:
        if any(word in str(e).lower() for word in ["billing", "quota", "location", "geo-block"]):
            raise InsufficientBalanceError(f"Gemini API Error: {e}")
        raise AIServiceError(f"Ошибка при обращении к Gemini: {e}")


async def _call_kie_chat(api_key: str, base_url: str, model: str, history: list, context: str, system_prompt: str, temperature: float = 0.7) -> str:
    try:
        kie_history = []
        for msg in history:
            if msg.content:
                kie_history.append({"role": msg.role, "content": msg.content})

        full_system_prompt = f"{system_prompt}\n\nИспользуй следующие данные из базы знаний для ответа:\n{context}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": full_system_prompt},
                *kie_history,
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
            response.status_code,
            response.json(),
            context="Ошибка при обращении к KIE Chat API",
        )
        text = _extract_kie_chat_text(response_payload)
        if not text:
            raise AIResponseError("KIE chat returned empty content")
        return text
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE chat error", exc_info=e)
        raise AIServiceError(f"Ошибка при обращении к KIE Chat API: {e}")


async def _upload_file_to_kie(api_key: str, upload_base_url: str, file_bytes: bytes, filename: str, upload_path: str) -> str:
    url = f"{upload_base_url}/api/file-stream-upload"
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    data = {"uploadPath": upload_path, "fileName": filename}
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
        payload = response.json()
        data_payload = _validate_kie_json_response(
            response.status_code,
            payload,
            context="KIE upload failed",
        )
        file_url = data_payload.get("downloadUrl") or data_payload.get("fileUrl")
        if not file_url:
            raise AIResponseError(f"KIE upload returned no file URL: {payload}")
        return file_url
    except (AIServiceError, AIResponseError):
        raise
    except Exception as e:
        logging.error("KIE upload error", exc_info=e)
        raise AIServiceError(f"Ошибка загрузки файла в KIE: {e}")


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
        raise AIServiceError(f"Ошибка обращения к KIE multimodal API: {e}")


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
        raise AIServiceError(f"Ошибка создания задачи KIE: {e}")


def _extract_kie_task_result(task_payload: dict) -> dict:
    response_payload = task_payload.get("response")
    if isinstance(response_payload, dict) and response_payload:
        return response_payload
    result_json = task_payload.get("resultJson")
    if isinstance(result_json, str) and result_json:
        try:
            return json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise AIResponseError(f"Cannot decode KIE resultJson: {exc}: {result_json}") from exc
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
    import httpx

    endpoint = f"{base_url}/api/v1/chat/credit"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.get(endpoint, headers=headers)
        if response.status_code != 200:
            raise AIServiceError(f"KIE credits check failed: status={response.status_code} body={response.text}")

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
    except (AIServiceError, AIResponseError):
        raise
    except Exception as e:
        logging.error("KIE credits check error", exc_info=e)
        raise AIServiceError(f"Ошибка проверки остатка кредитов KIE: {e}")


async def _call_claude_api(api_key: str, model: str, history: list, context: str, system_prompt: str, temperature: float = 0.7):
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)

        claude_history = []
        for msg in history:
            claude_history.append({'role': msg.role, 'content': msg.content})

        full_system_prompt = f"{system_prompt}\n\nИспользуй следующие данные из базы знаний для ответа:\n{context}"

        message = await client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=full_system_prompt,
            messages=claude_history
        )
        return message.content[0].text
    except anthropic.AuthenticationError as e:
        raise InsufficientBalanceError(f"Claude API Error: {e}")
    except Exception as e:
        logging.error(f"Claude API error: {e}")
        raise AIServiceError(f"Ошибка при обращении к Claude API: {e}")


async def _call_deepseek_api(api_key: str, model: str, history: list, context: str, system_prompt: str, temperature: float = 0.7):
    client = None
    try:
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
        transport = _build_async_transport_from_env("DEEPSEEK_PROXY")
        import httpx
        http_client = httpx.AsyncClient(transport=transport, trust_env=False, timeout=60.0)

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

        deepseek_history = []
        for msg in history:
            deepseek_history.append({'role': msg.role, 'content': msg.content})

        full_system_prompt = f"{system_prompt}\n\nИспользуй следующие данные из базы знаний для ответа:\n{context}"

        messages_with_system = [
            {"role": "system", "content": full_system_prompt},
            *deepseek_history
        ]

        chat_completion = await client.chat.completions.create(
            model=model,
            messages=messages_with_system,
            max_tokens=4096,
            temperature=temperature
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        if hasattr(e, 'code') and e.code == 'insufficient_quota':
            raise InsufficientBalanceError(f"Deepseek API Error: {e}")
        logging.error(f"Deepseek API error: {e}")
        raise AIServiceError(f"Ошибка при обращении к Deepseek API: {e}")
    finally:
        if client is not None:
            await client.close()




async def _call_openai_transcribe(api_key: str, file_bytes: bytes, filename: str) -> str:
    try:
        client = AsyncOpenAI(api_key=api_key)

        transcription = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, file_bytes)
        )
        return transcription.text
    except AuthenticationError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Invalid API Key. {e}")
    except RateLimitError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Rate limit or quota exceeded. {e}")
    except BadRequestError as e:
        if "billing" in str(e) or "quota" in str(e).lower():
            raise InsufficientBalanceError(f"OpenAI API Error: Billing issue or insufficient quota. {e}")
        logging.error(f"OpenAI API error: {e}")
        raise AIServiceError(f"Ошибка при транскрибации (OpenAI API): {e}")
    except Exception as e:
        logging.error(f"OpenAI API transcription error: {e}")
        raise AIServiceError(f"Ошибка при транскрибации: {e}")


async def transcribe_voice_message(file_bytes: bytes, filename: str) -> str:
    async with async_session_maker() as session:
        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            return "❌ Ошибка: Конфигурация ИИ не найдена."

        provider = ai_config.transcription_provider

        if provider == "OpenAI":
            api_key = ai_config.openai_api_key
            if not api_key:
                return f"❌ Ошибка: API ключ для {provider} (для транскрибации) не установлен администратором."
            response_text = await _call_openai_transcribe(api_key, file_bytes, filename)

        elif provider == "Gemini":
            api_key = ai_config.gemini_api_key
            model = ai_config.gemini_model
            if not api_key:
                return f"❌ Ошибка: API ключ для {provider} (для транскрибации) не установлен администратором."
            if not model:
                return f"❌ Ошибка: Модель для {provider} (для транскрибации) не выбрана администратором."
            response_text = await _call_gemini_transcribe(api_key, model, file_bytes, filename)
        elif provider == "KIE":
            api_key = ai_config.kie_api_key
            model = getattr(ai_config, "kie_transcription_model", None) or getattr(ai_config, "kie_model", None)
            if not api_key:
                return f"❌ Ошибка: API ключ для {provider} (для транскрибации) не установлен администратором."
            if not model:
                return f"❌ Ошибка: Модель для {provider} (для транскрибации) не выбрана администратором."
            response_text = await _call_kie_transcribe(
                api_key,
                _get_kie_base_url(ai_config),
                _get_kie_upload_base_url(ai_config),
                model,
                file_bytes,
                filename,
            )

        else:
            return f"❌ Ошибка: Неизвестный провайдер транскрибации: {provider}"

        return response_text


async def _call_openai_api(api_key: str, model: str, history: list, context: str, system_prompt: str, temperature: float = 0.7):
    try:
        client = AsyncOpenAI(api_key=api_key)

        openai_history = []
        for msg in history:
            openai_history.append({'role': msg.role, 'content': msg.content})

        full_system_prompt = f"{system_prompt}\n\nИспользуй следующие данные из базы знаний для ответа:\n{context}"

        messages_with_system = [
            {"role": "system", "content": full_system_prompt},
            *openai_history
        ]

        chat_completion = await client.chat.completions.create(
            model=model,
            messages=messages_with_system,
            max_tokens=4096,
            temperature=temperature
        )
        return chat_completion.choices[0].message.content
    except AuthenticationError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Invalid API Key. {e}")
    except RateLimitError as e:
        raise InsufficientBalanceError(f"OpenAI API Error: Rate limit or quota exceeded. {e}")
    except BadRequestError as e:
        if "billing" in str(e) or "quota" in str(e).lower():
            raise InsufficientBalanceError(f"OpenAI API Error: Billing issue or insufficient quota. {e}")
        logging.error(f"OpenAI API error: {e}")
        raise AIServiceError(f"Ошибка при обращении к OpenAI API: {e}")
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        raise AIServiceError(f"Ошибка при обращении к OpenAI API: {e}")


async def get_ai_response(user_id: int, user_prompt: str, user_name: str, user_gender: str) -> str:
    async with async_session_maker() as session:
        user_result = await session.execute(
            select(User).options(selectinload(User.current_topic).selectinload(Topic.knowledge_base_files)).where(
                User.id == user_id)
        )
        user = user_result.scalar_one_or_none()

        if not user:
            return "❌ Ошибка: Пользователь не найден."

        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            return "❌ Ошибка: Конфигурация ИИ не найдена."

        temperature = getattr(ai_config, 'temperature', 0.7) or 0.7

        available_media_text = ""
        if user.current_topic_id:
            # Получаем ID коллекций, привязанных к этому топику
            coll_stmt = select(topic_collection_association.c.collection_id).where(
                topic_collection_association.c.topic_id == user.current_topic_id
            )
            coll_res = await session.execute(coll_stmt)
            assigned_coll_ids = [r[0] for r in coll_res.all()]

            if assigned_coll_ids:
                # Медиа из привязанных коллекций + свои медиа по topic_id (для аудио и пр.)
                media_stmt = select(MediaLibrary).where(
                    or_(
                        MediaLibrary.id.in_(
                            select(media_collection_items.c.media_id).where(
                                media_collection_items.c.collection_id.in_(assigned_coll_ids)
                            )
                        ),
                        MediaLibrary.topic_id == user.current_topic_id
                    )
                )
            else:
                # Фоллбэк: старые колоды (topic_media_deck) или прямой topic_id
                deck_stmt = select(TopicMediaDeck.deck_name).where(
                    TopicMediaDeck.topic_id == user.current_topic_id
                )
                deck_res = await session.execute(deck_stmt)
                assigned_decks = [r[0] for r in deck_res.all()]
                if assigned_decks:
                    media_stmt = select(MediaLibrary).where(
                        or_(
                            MediaLibrary.category.in_(assigned_decks),
                            MediaLibrary.topic_id == user.current_topic_id
                        )
                    )
                else:
                    media_stmt = select(MediaLibrary).where(
                        MediaLibrary.topic_id == user.current_topic_id
                    )

            media_res = await session.execute(media_stmt)
            media_files = media_res.scalars().all()
            if media_files:
                categories = {}
                for m in media_files:
                    cat = m.category or ''
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(m)
                available_media_text = "Доступные медиа-файлы в этой теме:\n"
                for cat, files in categories.items():
                    if cat:
                        available_media_text += f"\nКатегория (для тегов RANDOM_IMG/CHOICE_IMG): \"{cat}\"\n"
                    for m in files:
                        desc_part = f" — {m.description}" if m.description else ""
                        available_media_text += f"  - [{m.media_type.upper()}] {m.file_name}{desc_part}\n"
            else:
                available_media_text = (
                    "Медиа-файлы (карты, аудио) в этой теме НЕ загружены.\n"
                    "НЕ используй теги RANDOM_IMG, CHOICE_IMG, CHOICE_IMG_HIDDEN, SHOW_IMG, SEND_AUDIO.\n"
                    "Для визуализации используй только GEN_IMG: [промпт на английском].\n"
                )

        provider = ai_config.provider
        provider_key = provider.strip().lower() if provider else ""

        api_key = getattr(ai_config, f"{provider_key}_api_key", None)
        if provider_key in ['anthropic', 'claude'] and not api_key:
            api_key = ai_config.claude_api_key

        if not api_key:
            return f"⚠️ Ошибка настройки: Не указан API ключ для провайдера '{provider}'. Пожалуйста, сообщите администратору."

        model = getattr(ai_config, f"{provider_key}_model", None)
        if provider_key in ['anthropic', 'claude'] and not model:
            model = ai_config.claude_model

        limit_first = ai_config.context_limit_first
        limit_recent = ai_config.context_limit_recent

        system_prompt_text = _load_configured_system_prompt(
            ai_config,
            user.current_topic.system_prompt if user.current_topic else None
        )

        test_results_txt = ""
        secret_answers_txt = ""

        test_session = await session.get(TestSession, user_id)
        if test_session and test_session.is_finished:
            if test_session.answers:
                test_results_txt = test_session.answers
            if test_session.secret_answers:
                secret_answers_txt = test_session.secret_answers

        test_context_injection = ""
        if user.current_topic_id is None and (test_results_txt or secret_answers_txt):
            context_parts = []
            status_instruction = ""
            if test_results_txt:
                context_parts.append(f"Результаты основного теста (пройден):\n{test_results_txt}")
            if secret_answers_txt:
                context_parts.append(f"Ответы пользователя на СЕКРЕТНЫЙ тест (УЖЕ ПРОЙДЕН):\n{secret_answers_txt}")
                status_instruction = "Пользователь УЖЕ прошел все тесты. Обсуждай результаты."
            elif test_results_txt:
                status_instruction = "Пользователь прошел основной тест. Предложи пройти секретный блок."

            if context_parts:
                joined_results = "\n\n".join(context_parts)
                test_context_injection = f"\n\n[КОНТЕКСТ ТЕСТА]\n{joined_results}\nИНСТРУКЦИЯ: {status_instruction}"

        safe_user_name = user_name if user_name else "Не указано"
        safe_user_gender = user_gender if user_gender else "Не указан"
        forced_user_header = f"ДАННЫЕ КЛИЕНТА:\nИМЯ: {safe_user_name}\nПОЛ: {safe_user_gender}\n"
        if user.age:
            forced_user_header += f"ВОЗРАСТ: {user.age}\n"
        forced_user_header += "\n"

        try:
            formatted_body = system_prompt_text.format(user_name=safe_user_name, user_gender=safe_user_gender, test_results=test_results_txt, secret_answers=secret_answers_txt)
        except Exception:
            formatted_body = system_prompt_text

        shared_prompt_block = (getattr(ai_config, 'shared_prompt_block', "") or "").strip()
        short_response_instruction = ""
        if getattr(user, 'response_length', 'normal') == 'short':
            short_response_instruction = DEFAULT_SHORT_RESPONSE_INSTRUCTION

        service_prompt_template = getattr(ai_config, 'service_prompt_block', None) or DEFAULT_SERVICE_PROMPT_TEMPLATE
        service_prompt_block = render_prompt_block(
            service_prompt_template,
            available_media_text=available_media_text,
            test_context_injection=test_context_injection,
            short_response_instruction=short_response_instruction,
        )

        prompt_parts = [forced_user_header.strip(), formatted_body.strip()]
        if shared_prompt_block:
            prompt_parts.append(shared_prompt_block)
        if service_prompt_block:
            prompt_parts.append(service_prompt_block)
        system_prompt = "\n\n".join(part for part in prompt_parts if part)

        relevant_chunks = []
        if user.current_topic:
            doc_ids = [f.id for f in user.current_topic.knowledge_base_files]
            if doc_ids:
                relevant_chunks = await search_relevant_chunks(user_prompt, n_results=3, document_ids=doc_ids)
        else:
            from database import KnowledgeBase
            gen_files_res = await session.execute(select(KnowledgeBase.id).where(KnowledgeBase.use_in_general_mode == True))
            gen_doc_ids = gen_files_res.scalars().all()
            if gen_doc_ids:
                relevant_chunks = await search_relevant_chunks(user_prompt, n_results=3, document_ids=gen_doc_ids)

        context = "\n\n".join(relevant_chunks)

        memory_mode = get_memory_mode(ai_config)
        stmt = select(DBMessage).where(
            DBMessage.user_id == user.id,
            DBMessage.dialogue_id == user.current_dialogue_id,
        )
        if not is_global_memory_mode(memory_mode):
            stmt = stmt.where(DBMessage.topic_id == user.current_topic_id)
        stmt = stmt.order_by(DBMessage.timestamp.asc())
        result = await session.execute(stmt)
        all_messages = result.scalars().all()

        if len(all_messages) <= limit_first + limit_recent:
            final_history = list(all_messages)
        else:
            final_history = all_messages[:limit_first] + all_messages[-limit_recent:]

        final_history.append(DBMessage(role='user', content=user_prompt))

        if provider_key == 'openai':
            response_text = await _call_openai_api(api_key, model, final_history, context, system_prompt, temperature)
        elif provider_key in ['anthropic', 'claude']:
            response_text = await _call_claude_api(api_key, model, final_history, context, system_prompt, temperature)
        elif provider_key == 'gemini':
            response_text = await _call_gemini_api(api_key, model, final_history, context, system_prompt, temperature)
        elif provider_key == 'kie':
            response_text = await _call_kie_chat(api_key, _get_kie_base_url(ai_config), model, final_history, context, system_prompt, temperature)
        elif provider_key == 'deepseek':
            response_text = await _call_deepseek_api(api_key, model, final_history, context, system_prompt, temperature)
        elif provider_key == 'xai':
            response_text = await _call_openai_api(api_key, model, final_history, context, system_prompt, temperature)
        else:
            response_text = f"❌ Ошибка: Неизвестный провайдер ИИ: '{provider}'"

        return response_text


async def _call_gemini_transcribe(api_key: str, model: str, file_bytes: bytes, filename: str) -> str:
    import httpx
    import base64

    try:
        raw_proxy = os.getenv("GEMINI_PROXY")

        transport = None

        if raw_proxy:
            gemini_proxy = raw_proxy.strip().strip('"').strip("'")
            transport = httpx.AsyncHTTPTransport(proxy=gemini_proxy)

        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type or not mime_type.startswith('audio/'):
            mime_type = 'audio/ogg'

        b64_data = base64.b64encode(file_bytes).decode('utf-8')
        target_model = model if model else "gemini-2.5-flash"

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
                error_msg = error_data.get('error', {}).get('message', str(response.text))

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
            raise InsufficientBalanceError(f"Gemini Error: {e}")
        raise AIServiceError(f"Ошибка при транскрибации (Gemini API): {e}")


async def _call_kie_transcribe(api_key: str, base_url: str, upload_base_url: str, model: str, file_bytes: bytes, filename: str) -> str:
    try:
        file_url = await _upload_file_to_kie(api_key, upload_base_url, file_bytes, filename, "audio")
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
        )
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE transcription error", exc_info=e)
        raise AIServiceError(f"Ошибка при транскрибации (KIE API): {e}")


async def _call_gemini_image_generation(api_key: str, model: str, prompt: str) -> bytes:
    import httpx
    import base64
    try:
        raw_proxy = os.getenv("GEMINI_PROXY")
        transport = None
        if raw_proxy:
            gemini_proxy = raw_proxy.strip().strip('"').strip("'")
            transport = httpx.AsyncHTTPTransport(proxy=gemini_proxy)

        target_model = model if model else "imagen-4.0-generate-001"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:predict?key={api_key}"

        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1",
                "personGeneration": "allow_adult"
            }
        }

        async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=60.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                raise Exception(f"Imagen Error: {response.text}")

            data = response.json()
            predictions = data.get('predictions', [])
            if not predictions:
                raise Exception("No images generated")

            img_b64 = predictions[0].get('bytesBase64Encoded')
            return base64.b64decode(img_b64)
    except Exception as e:
        logging.error(f"Imagen Generation Error: {e}")
        raise e


async def edit_image_gemini_v3(api_key: str, model: str, prompt: str, image_bytes: bytes) -> bytes:
    import httpx
    import base64
    try:
        raw_proxy = os.getenv("GEMINI_PROXY")
        transport = None
        if raw_proxy:
            gemini_proxy = raw_proxy.strip().strip('"').strip("'")
            transport = httpx.AsyncHTTPTransport(proxy=gemini_proxy)
        b64_data = base64.b64encode(image_bytes).decode('utf-8')
        target_model = model if model else "gemini-3-pro-image-preview"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"Generate an edited version of this image based on: {prompt}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}}
                ]
            }]
        }
        async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=60.0) as client:
            response = await client.post(url, json=payload, headers={'Content-Type': 'application/json'})
            if response.status_code != 200:
                logging.error(f"Gemini Edit Image HTTP Error: {response.status_code} - {response.text}")
                raise AIServiceError(f"Ошибка редактирования изображения Gemini: {response.status_code} {response.text}")
            data = response.json()
            parts = data.get('candidates', [{}])[0].get('content', {}).get('parts', [])
            for part in parts:
                if 'inline_data' in part:
                    return base64.b64decode(part['inline_data']['data'])
            raise AIResponseError("Gemini edit image returned no binary payload")
    except Exception as e:
        logging.error("Gemini edit_image_gemini_v3 Exception", exc_info=e)
        raise AIServiceError(f"Ошибка редактирования изображения Gemini: {e}")


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


async def _call_kie_image_generation(api_key: str, base_url: str, model: str, prompt: str) -> bytes:
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
        {
            "prompt": prompt,
            "image_urls": [source_url],
            "output_format": "png",
            "image_size": "1:1",
        },
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
            raise Exception("Конфигурация ИИ не найдена.")

        provider = getattr(config, "image_generation_provider", None) or config.vision_provider
        provider_key = _normalize_provider_name(provider)
        model = getattr(config, "image_generation_model", None) or "imagen-4.0-generate-001"

    if provider_key == 'gemini':
        api_key = config.gemini_api_key
        if not api_key:
            raise Exception("API ключ Gemini для генерации не установлен.")
        return await _call_gemini_image_generation(api_key, model, prompt)
    if provider_key == 'kie':
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise Exception("API ключ KIE для генерации не установлен.")
        try:
            return await _call_kie_image_generation(api_key, _get_kie_base_url(config), model or "google/imagen4-fast", prompt)
        except AIServiceError as exc:
            if _is_kie_transient_failure(exc):
                logging.warning("KIE image generation transient failure, falling back to OpenAI: %s", exc)
                return await generate_openai_image(prompt)
            raise
    else:
        return await generate_openai_image(prompt)


async def edit_image(prompt: str, image_bytes: bytes) -> bytes:
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            raise Exception("Конфигурация ИИ не найдена.")

        provider = getattr(config, "image_edit_provider", None) or config.vision_provider
        provider_key = _normalize_provider_name(provider)
        model = getattr(config, "image_edit_model", None) or "gemini-3-pro-image-preview"

    if provider_key == "gemini":
        api_key = config.gemini_api_key
        if not api_key:
            raise Exception("API ключ Gemini для редактирования не установлен.")
        return await edit_image_gemini_v3(api_key, model, prompt, image_bytes)
    if provider_key == "kie":
        api_key = getattr(config, "kie_api_key", None)
        if not api_key:
            raise Exception("API ключ KIE для редактирования не установлен.")
        return await _call_kie_image_edit(
            api_key,
            _get_kie_base_url(config),
            _get_kie_upload_base_url(config),
            model or "google/nano-banana-edit",
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
        raise Exception("API ключ OpenAI не установлен.")

    base_url = os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        model = "gpt-image-1.5"
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
            raise last_error or Exception("OpenAI image generation failed without response")

        if not response.data:
            raise Exception("API не вернул данных (empty data).")

        img_data = response.data[0]

        if img_data.url:
            return img_data.url
        elif img_data.b64_json:
            return base64.b64decode(img_data.b64_json)
        else:
            raise Exception("API не вернул ни URL, ни B64.")

    except Exception as e:
        logging.error(f"OpenAI Image Error ({model}): {e}")
        raise Exception(f"Ошибка генерации изображения: {e}")


async def analyze_image_content(image_bytes: bytes, prompt: str, history: list = None) -> str:
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            raise Exception("Конфигурация ИИ не найдена.")

        provider = config.vision_provider
        v_model = config.vision_model
        temperature = getattr(config, 'temperature', 0.7) or 0.7
        api_key = None

        if provider == "Gemini":
            api_key = config.gemini_api_key
            if not api_key:
                return "❌ Ошибка: API ключ для Gemini (Vision) не установлен."
            return await _call_gemini_vision(api_key, v_model, image_bytes, prompt, history=history, temperature=temperature)
        if provider == "KIE":
            api_key = getattr(config, "kie_api_key", None)
            if not api_key:
                return "❌ Ошибка: API ключ для KIE (Vision) не установлен."
            try:
                return await _call_kie_vision(
                    api_key,
                    _get_kie_base_url(config),
                    _get_kie_upload_base_url(config),
                    v_model or "gemini-3-flash",
                    image_bytes,
                    prompt,
                    history=history,
                    temperature=temperature,
                )
            except AIServiceError as exc:
                if not _is_kie_transient_failure(exc):
                    raise
                logging.warning("KIE vision transient failure, falling back to OpenAI: %s", exc)
                api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise
                v_model = "gpt-4o"
        else:
            api_key = config.openai_api_key

        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "❌ Ошибка: API ключ для OpenAI (Vision) не установлен."

        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        formatting_rules = (
            "\n\nТЕХНИЧЕСКИЕ ПРАВИЛА ФОРМАТИРОВАНИЯ:\n"
            "1. Markdown: Всегда используй стандартный Markdown. Никакого ручного HTML.\n"
            "2. ПРАВИЛО ВЫДЕЛЕНИЯ ТЕКСТА: Используй жирный шрифт (**текст**) только для заголовков. "
            "Никогда не выделяй жирным целые абзацы. Всегда закрывай теги **.\n"
            "3. Списки: Для маркированных списков используй исключительно дефис '-'.\n"
        )

        vision_instructions = (
            "You are a professional expert analyst. Analyze the provided image thoroughly. "
            "If visualization is needed, add at the very end: "
            "GEN_IMG: [Detailed English prompt].\n\n"
            f"Role and Context: {prompt}{formatting_rules}"
        )

        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": vision_instructions},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "high"}}
            ]
        })

        base_url = os.getenv("BASE_URL_OPENAI", "https://api.openai.com/v1")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        try:
            response = await client.chat.completions.create(
                model=v_model,
                messages=messages,
                max_tokens=4096,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI Vision Error: {e}")
            raise AIServiceError(f"Ошибка анализа изображения (OpenAI): {e}")


async def _call_gemini_vision(api_key: str, model: str, image_bytes: bytes, prompt: str, history: list = None, temperature: float = 0.7) -> str:
    import httpx
    import base64
    import asyncio

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            raw_proxy = os.getenv("GEMINI_PROXY")
            transport = None
            if raw_proxy:
                gemini_proxy = raw_proxy.strip().strip('"').strip("'")
                transport = httpx.AsyncHTTPTransport(proxy=gemini_proxy)

            b64_data = base64.b64encode(image_bytes).decode('utf-8')
            target_model = model if model else "gemini-1.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"

            contents = []
            if history:
                for msg in history:
                    if not msg.content: continue
                    role = 'user' if msg.role == 'user' else 'model'
                    contents.append({'role': role, 'parts': [{'text': msg.content}]})

            contents.append({
                "role": "user",
                "parts": [
                    {"text": "Проанализируй это изображение согласно системной инструкции выше."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}}
                ]
            })

            payload = {
                "contents": contents,
                "systemInstruction": {"parts": [{"text": prompt}]},
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096}
            }

            async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=60.0) as client:
                response = await client.post(url, json=payload, headers={'Content-Type': 'application/json'})

                if response.status_code == 200:
                    data = response.json()
                    if 'candidates' in data and data['candidates']:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    return "Ошибка: Не удалось получить текст из ответа Gemini Vision."

                if response.status_code in [503, 429]:
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Gemini 503/429 error, retry {attempt + 1}/{max_retries} after {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                error_detail = response.text
                logging.error(f"Gemini Vision API Error ({response.status_code}): {error_detail}")
                raise AIServiceError(f"Ошибка API Gemini Vision: {response.status_code}")

        except (AIServiceError, InsufficientBalanceError):
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue
            logging.error(f"Ошибка вызова Gemini Vision: {e}")
            raise AIServiceError(f"Ошибка анализа изображения (Gemini): {e}")

    raise AIServiceError("Сервис Gemini Vision временно перегружен.")


async def _call_kie_vision(
    api_key: str,
    base_url: str,
    upload_base_url: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    history: list = None,
    temperature: float = 0.7,
) -> str:
    try:
        file_url = await _upload_file_to_kie(
            api_key,
            upload_base_url,
            image_bytes,
            _guess_filename(image_bytes, "vision_input", "jpg"),
            "images",
        )

        history_text = []
        if history:
            for msg in history:
                if msg.content:
                    prefix = "Пользователь" if msg.role == "user" else "Ассистент"
                    history_text.append(f"{prefix}: {msg.content}")

        system_prompt = prompt
        if history_text:
            system_prompt = f"{prompt}\n\nКонтекст диалога:\n" + "\n".join(history_text[-12:])

        return await _call_kie_multimodal(
            api_key,
            base_url,
            model,
            system_prompt,
            [
                {"type": "text", "text": "Проанализируй это изображение согласно системной инструкции."},
                {"type": "image_url", "image_url": {"url": file_url}},
            ],
            temperature=temperature,
        )
    except (InsufficientBalanceError, AIServiceError):
        raise
    except Exception as e:
        logging.error("KIE vision error", exc_info=e)
        raise AIServiceError(f"Ошибка анализа изображения (KIE): {e}")
