"""Shared direct Gemini image generation and editing adapter.

This module owns the native Gemini Interactions REST protocol. Telegram and
MAX only select the configured provider/model and translate these provider
errors into their local AI error classes.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from typing import Any

import httpx

from provider_models import PROVIDER_GEMINI, ensure_model_available
from provider_failures import ProviderFailure, attach_failure, normalize_exception, parse_provider_failure, safe_message


GEMINI_INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
_REQUEST_TIMEOUT_SECONDS = 120.0
_MAX_ERROR_DETAIL_LENGTH = 240


class GeminiImageError(RuntimeError):
    """Safe provider/network failure while calling Gemini image generation."""


class GeminiImageResponseError(GeminiImageError):
    """Gemini returned an invalid, empty, or otherwise unusable image response."""


def _local_response_failure(
    operation: str,
    message: str,
    *,
    status: int = 200,
    headers: Any = None,
) -> ProviderFailure:
    header_values = {str(key).lower(): value for key, value in (headers or {}).items()}
    return ProviderFailure(
        provider="Gemini",
        operation=operation,
        classification="invalid_response",
        detail_code="invalid_response",
        provider_error_message=safe_message(message),
        http_status=status,
        request_id=safe_message(
            header_values.get("x-request-id")
            or header_values.get("x-goog-request-id")
            or header_values.get("request-id"),
            200,
        ) or None,
        retryable=False,
        transient=False,
        classification_source="message_fallback",
    )


def _error_with_failure(
    exception: Exception,
    *,
    operation: str,
    response: httpx.Response | None = None,
    payload: dict[str, Any] | None = None,
) -> GeminiImageError:
    headers = getattr(response, "headers", None) if response is not None else None
    status = getattr(response, "status_code", None) if response is not None else None
    failure = parse_provider_failure("Gemini", operation, status, payload, headers)
    if failure is None:
        failure = normalize_exception(exception, "Gemini", operation)
    wrapped = exception if isinstance(exception, GeminiImageError) else GeminiImageError(str(exception))
    attach_failure(wrapped, failure)
    if payload is not None:
        try:
            wrapped.provider_response_payload = safe_message(json.dumps(payload, ensure_ascii=False, default=str), 4000)
        except Exception:
            wrapped.provider_response_payload = safe_message(payload, 4000)
    return wrapped


def _build_gemini_proxy_transport() -> httpx.AsyncHTTPTransport | None:
    raw_proxy = os.getenv("GEMINI_PROXY")
    if not raw_proxy:
        return None
    proxy = raw_proxy.strip().strip('"').strip("'")
    if not proxy:
        return None
    return httpx.AsyncHTTPTransport(proxy=proxy)


def detect_image_mime_type(image_bytes: bytes) -> str:
    """Detect a supported image MIME type from its file signature."""
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or not image_bytes:
        raise GeminiImageResponseError("Gemini image edit source image is empty")

    data = bytes(image_bytes)
    header = data[:32]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx"}:
            return "image/heic"
        if brand in {b"heif", b"mif1", b"msf1"}:
            return "image/heif"

    raise GeminiImageResponseError("Gemini image edit source image format is unsupported")


def build_generation_payload(model: str, prompt: str) -> dict[str, Any]:
    """Build the native Interactions text-to-image request payload."""
    ensure_model_available(PROVIDER_GEMINI, model, channel="image_gen")
    return {
        "model": model,
        "input": [{"type": "text", "text": prompt}],
    }


def build_edit_payload(model: str, prompt: str, image_bytes: bytes) -> dict[str, Any]:
    """Build the native Interactions image-edit request payload."""
    ensure_model_available(PROVIDER_GEMINI, model, channel="image_edit")
    mime_type = detect_image_mime_type(image_bytes)
    return {
        "model": model,
        "input": [
            {
                "type": "image",
                "data": base64.b64encode(bytes(image_bytes)).decode("ascii"),
                "mime_type": mime_type,
            },
            {"type": "text", "text": prompt},
        ],
    }


def _safe_error_detail(payload: object) -> str:
    if not isinstance(payload, dict):
        return "provider returned an error"

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        status = error.get("status") or error.get("code")
        if isinstance(message, str) and message.strip():
            detail = message.strip()
            if status:
                detail = f"{status}: {detail}"
        elif status:
            detail = str(status)
        else:
            detail = "provider returned an error"
    elif isinstance(error, str) and error.strip():
        detail = error.strip()
    else:
        detail = "provider returned an error"

    return detail[:_MAX_ERROR_DETAIL_LENGTH]


def _decode_image_data(data: object) -> bytes:
    if not isinstance(data, str) or not data:
        raise GeminiImageResponseError("Gemini image response contained empty image data")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise GeminiImageResponseError("Gemini image response contained invalid base64") from exc
    if not decoded:
        raise GeminiImageResponseError("Gemini image response contained an empty image")
    return decoded


def extract_image_bytes(payload: object, *, operation: str = "image_generation") -> bytes:
    """Extract the first image block from a native Interactions response."""
    if not isinstance(payload, dict):
        raise GeminiImageResponseError("Gemini image response was not a JSON object")
    failure = parse_provider_failure("Gemini", operation, 200, payload)
    if failure is not None:
        error = attach_failure(GeminiImageError(f"Gemini image provider error: {_safe_error_detail(payload)}"), failure)
        try:
            error.provider_response_payload = safe_message(json.dumps(payload, ensure_ascii=False, default=str), 4000)
        except Exception:
            pass
        raise error

    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise GeminiImageResponseError("Gemini image response contained no model_output steps")

    found_model_output = False
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        found_model_output = True
        content = step.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            return _decode_image_data(block.get("data"))

    if not found_model_output:
        error = GeminiImageResponseError("Gemini image response contained no model_output steps")
        raise attach_failure(error, _local_response_failure(operation, str(error)))
    error = GeminiImageResponseError("Gemini image response contained no image block")
    raise attach_failure(error, _local_response_failure(operation, str(error)))


def _parse_json_response(response: httpx.Response, *, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        error = GeminiImageResponseError(
            f"Gemini image provider returned invalid JSON (HTTP {response.status_code})"
        )
        failure = parse_provider_failure(
            "Gemini",
            operation,
            response.status_code,
            {"error": {"message": getattr(response, "text", "")[:4000]}},
            getattr(response, "headers", None) or {},
        ) or _local_response_failure(
            operation,
            str(error),
            status=response.status_code,
            headers=getattr(response, "headers", None),
        )
        raise attach_failure(
            error,
            failure,
        ) from exc

    if not isinstance(payload, dict):
        error = GeminiImageResponseError("Gemini image provider returned a non-object JSON response")
        failure = parse_provider_failure(
            "Gemini",
            operation,
            response.status_code,
            {"error": {"message": str(payload)[:4000]}},
            getattr(response, "headers", None) or {},
        ) or _local_response_failure(
            operation,
            str(error),
            status=response.status_code,
            headers=getattr(response, "headers", None),
        )
        raise attach_failure(
            error,
            failure,
        )
    return payload


async def _post_interaction(api_key: str, payload: dict[str, Any], *, operation: str) -> dict[str, Any]:
    if not isinstance(api_key, str) or not api_key.strip():
        error = GeminiImageError("Gemini image API key is not configured")
        raise attach_failure(error, normalize_exception(error, "Gemini", operation))

    headers = {
        "x-goog-api-key": api_key.strip(),
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(
            transport=_build_gemini_proxy_transport(),
            trust_env=False,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(
                GEMINI_INTERACTIONS_ENDPOINT,
                headers=headers,
                json=payload,
            )
    except httpx.TimeoutException as exc:
        error = _error_with_failure(exc, operation=operation)
        error.args = ("Gemini image request timed out",)
        raise error from exc
    except httpx.RequestError as exc:
        error = _error_with_failure(exc, operation=operation)
        error.args = ("Gemini image request failed due to a network error",)
        raise error from exc
    except Exception as exc:
        logging.warning("Gemini image request failed: %s", type(exc).__name__)
        error = _error_with_failure(exc, operation=operation)
        raise error from exc

    try:
        response_payload = _parse_json_response(response, operation=operation)
    except GeminiImageError:
        raise

    failure = parse_provider_failure(
        "Gemini",
        operation,
        response.status_code,
        response_payload,
        getattr(response, "headers", None) or {},
    )
    if failure is not None:
        error = attach_failure(
            GeminiImageError(
                f"Gemini image provider request failed (HTTP {response.status_code}): "
                f"{failure.provider_error_message or failure.detail_code}"
            ),
            failure,
        )
        try:
            error.provider_response_payload = safe_message(json.dumps(response_payload, ensure_ascii=False, default=str), 4000)
        except Exception:
            pass
        raise error

    if not 200 <= response.status_code < 300:
        error = GeminiImageError(
            f"Gemini image provider request failed (HTTP {response.status_code}): "
            f"{_safe_error_detail(response_payload)}"
        )
        raise attach_failure(error, _local_response_failure(operation, str(error), status=response.status_code))

    return response_payload


async def generate_image(api_key: str, model: str, prompt: str) -> bytes:
    """Generate an image with a direct Gemini image model."""
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="image_gen")
    payload = build_generation_payload(target_model, prompt)
    return extract_image_bytes(
        await _post_interaction(api_key, payload, operation="image_generation"),
        operation="image_generation",
    )


async def edit_image(api_key: str, model: str, prompt: str, image_bytes: bytes) -> bytes:
    """Edit an image with a direct Gemini image model."""
    target_model = (model or "").strip()
    ensure_model_available(PROVIDER_GEMINI, target_model, channel="image_edit")
    payload = build_edit_payload(target_model, prompt, image_bytes)
    return extract_image_bytes(
        await _post_interaction(api_key, payload, operation="image_edit"),
        operation="image_edit",
    )
