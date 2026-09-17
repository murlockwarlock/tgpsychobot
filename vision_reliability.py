"""Shared platform-neutral reliability helpers and policies for AI Vision."""
from __future__ import annotations

import asyncio
import copy
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Coroutine, Sequence, TypeVar

import httpx

T = TypeVar("T")


@dataclass(frozen=True)
class VisionExecutionContext:
    """Scalar execution context for vision analysis requests.

    Does NOT contain SQLAlchemy User/Topic ORM objects to avoid session-attachment
    or lifecycle-transaction contamination.
    """
    user_id: int | None = None
    username: str | None = None
    full_name: str | None = None
    dialogue_id: int | None = None
    topic_id: int | None = None
    topic_name: str | None = None
    platform: str = "telegram"
    chat_id: int | None = None
    bot_name: str | None = None
    bot: Any | None = None  # Telegram Bot instance for delivery; None for MAX


class VisionDeadlineTracker:
    """Tracks global wall-clock deadline and per-stage budgets for vision analysis."""

    def __init__(
        self,
        total_timeout_seconds: float = 85.0,
        total_deadline_seconds: float | None = None,
        total_deadline_sec: float | None = None,
    ):
        if total_deadline_sec is not None:
            self.total_timeout_seconds = total_deadline_sec
        elif total_deadline_seconds is not None:
            self.total_timeout_seconds = total_deadline_seconds
        else:
            self.total_timeout_seconds = total_timeout_seconds
        self.start_time = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def remaining(self) -> float:
        rem = self.total_timeout_seconds - self.elapsed
        return max(0.0, rem)

    def remaining_time(self) -> float:
        return self.remaining

    def remaining_seconds(self) -> float:
        return self.remaining

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0.0

    def stage_budget(
        self,
        stage_name_or_budget: Any = None,
        *,
        aggregate_cap: float | None = None,
        max_stage_budget: float | None = None,
        reserve_sec: float = 5.0,
        min_required: float = 3.0,
    ) -> float:
        """Calculate stage budget bounded by remaining global deadline."""
        cap = 25.0
        if aggregate_cap is not None:
            cap = aggregate_cap
        elif max_stage_budget is not None:
            cap = max_stage_budget
        elif isinstance(stage_name_or_budget, (int, float)):
            cap = float(stage_name_or_budget)

        rem = self.remaining
        if rem <= reserve_sec:
            return 0.0
        usable = rem - reserve_sec
        effective = min(cap, usable)
        if effective < min_required:
            return 0.0
        return max(0.0, effective)

    def get_stage_budget(self, stage_cap: float, min_required: float = 5.0) -> float:
        """Calculate stage budget bounded by remaining global deadline."""
        return self.stage_budget(stage_cap, min_required=min_required)


async def run_coro_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float | None = None,
    *,
    timeout_sec: float | None = None,
) -> T:
    """Execute coroutine bounded by timeout with active cancellation across Python versions."""
    eff_timeout = timeout_sec if timeout_sec is not None else timeout_seconds
    if eff_timeout is None or eff_timeout <= 0:
        raise asyncio.TimeoutError("Stage timeout budget exhausted before start")
    if sys.version_info >= (3, 11):
        async with asyncio.timeout(eff_timeout):
            return await coro
    else:
        return await asyncio.wait_for(coro, timeout=eff_timeout)


def build_vision_httpx_timeout(stage_budget: float) -> httpx.Timeout:
    """Create phase-aware HTTPX timeout derived from stage budget."""
    return httpx.Timeout(
        connect=min(10.0, stage_budget),
        read=stage_budget,
        write=min(30.0, stage_budget),
        pool=5.0,
    )


def attach_error_metadata(
    exc: Exception,
    *,
    classification: str | None = None,
    http_status: int | None = None,
    provider_code: int | str | None = None,
    finish_reason: str | None = None,
    diagnostics: Any = None,
    provider_response_payload: str | None = None,
    admin_alert_handled: bool | None = None,
    stage: str | None = None,
    **kwargs: Any,
) -> Exception:
    """Attach structured diagnostic metadata to an exception without subclass mutation."""
    if classification is not None:
        exc.classification = classification
    if http_status is not None:
        exc.http_status = http_status
    if provider_code is not None:
        exc.provider_code = provider_code
    if finish_reason is not None:
        exc.finish_reason = finish_reason
    if diagnostics is not None:
        exc.diagnostics = diagnostics
    if provider_response_payload is not None:
        exc.provider_response_payload = provider_response_payload
    if admin_alert_handled is not None:
        exc.admin_alert_handled = admin_alert_handled
    if stage is not None:
        exc.stage = stage
    for k, v in kwargs.items():
        if v is not None:
            setattr(exc, k, v)
    return exc


# Regex patterns for media redaction
_BASE64_DATA_URI_PATTERN = re.compile(r"data:image\/[a-zA-Z0-9.+_-]+;base64,[A-Za-z0-9+/=_-]+", re.IGNORECASE)
_RAW_BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/_-]{4}){16,}={0,2}")
_KIE_MEDIA_URL_PATTERN = re.compile(r"https?:\/\/[^\s\"'<>]+(?:\/upload\/|\/files\/|\/temp\/|\/download\/|\/file-stream-upload|\/images\/)[^\s\"'<>]+", re.IGNORECASE)
_SIGNED_MEDIA_URL_PATTERN = re.compile(
    r"https?:\/\/[^\s\"'<>]+\.(?:png|jpe?g|webp|gif|bmp)(?:\?[^\s\"'<>]*)?|https?:\/\/[^\s\"'<>]*?(?:X-Amz-Signature|signature=|sig=|token=)[^\s\"'<>]*", re.IGNORECASE
)


def sanitize_vision_request_payload(payload: Any) -> Any:
    """Sanitize images, base64 strings, and media URLs from payloads for audit/logs/alerts.

    Returns a clean, decoupled deep-copy. Ordinary endpoint URLs are preserved.
    """
    if payload is None:
        return None

    if isinstance(payload, bytes):
        return f"<redacted_image_bytes length={len(payload)}>"

    if isinstance(payload, str):
        # 1. Check data URI
        if "data:image/" in payload:
            payload = _BASE64_DATA_URI_PATTERN.sub("<redacted_data_image_uri>", payload)
        # 2. Check media URLs
        if _KIE_MEDIA_URL_PATTERN.search(payload):
            payload = _KIE_MEDIA_URL_PATTERN.sub("<redacted_kie_media_url>", payload)
        if _SIGNED_MEDIA_URL_PATTERN.search(payload):
            payload = _SIGNED_MEDIA_URL_PATTERN.sub("<redacted_media_url>", payload)
        # 3. Check long standard or URL-safe base64 chunks
        if _RAW_BASE64_PATTERN.search(payload):
            payload = _RAW_BASE64_PATTERN.sub("<redacted_base64_data>", payload)
        return payload

    if isinstance(payload, dict):
        cleaned_dict = {}
        for k, v in payload.items():
            k_lower = str(k).lower()
            if k_lower in {"image_bytes", "file_bytes", "bytes"}:
                byte_len = len(v) if isinstance(v, (bytes, bytearray)) else "?"
                cleaned_dict[k] = f"<redacted_image_bytes length={byte_len}>"
            elif k_lower in {"data", "b64_json", "base64"} and isinstance(v, str) and len(v) > 50:
                cleaned_dict[k] = f"<redacted_base64 length={len(v)}>"
            elif k_lower in {"file_url", "image_url"}:
                cleaned_dict[k] = "<redacted_media_url>"
            elif k_lower == "url" and isinstance(v, str):
                if v.startswith("data:image/") or _KIE_MEDIA_URL_PATTERN.search(v) or _SIGNED_MEDIA_URL_PATTERN.search(v):
                    cleaned_dict[k] = "<redacted_media_url>"
                else:
                    cleaned_dict[k] = v
            elif k_lower in {"temp_url", "download_url", "downloadurl", "fileurl"}:
                cleaned_dict[k] = "<redacted_media_url>"
            elif k_lower == "inline_data" and isinstance(v, dict):
                c_inline = copy.deepcopy(v)
                if "data" in c_inline:
                    c_inline["data"] = f"<redacted_base64 length={len(str(c_inline['data']))}>"
                cleaned_dict[k] = c_inline
            else:
                cleaned_dict[k] = sanitize_vision_request_payload(v)
        return cleaned_dict

    if isinstance(payload, (list, tuple, set)):
        cleaned_list = [sanitize_vision_request_payload(item) for item in payload]
        return type(payload)(cleaned_list)

    return payload


def sanitize_vision_text(text: str | None) -> str:
    """Sanitize strings containing base64 data, data URIs, or media URLs."""
    if not text:
        return ""
    res = sanitize_vision_request_payload(text)
    return str(res)



def _norm_cls(classification: Any) -> str:
    if isinstance(classification, (tuple, list)):
        return str(classification[0]) if classification else ""
    return str(classification or "")


def should_retry_kie_vision_upload(classification: Any) -> bool:
    """KIE upload retry eligibility: max 1 retry for transient transport/server issues."""
    cls = _norm_cls(classification)
    return cls in {"timeout", "network_ssl", "network_connection", "provider_5xx"}


def should_retry_kie_vision_model(classification: Any) -> bool:
    """Same-KIE alternate model retry eligibility."""
    # 429 rate_limit does NOT retry same-KIE model
    cls = _norm_cls(classification)
    return cls in {
        "timeout",
        "network_ssl",
        "network_connection",
        "provider_5xx",
        "invalid_response",
        "empty_response",
    }


def should_use_vision_provider_fallback(
    classification: Any,
    primary_budget: int = 4096,
    fallback_budget: int = 16384,
    *,
    failed_budget: int | None = None,
) -> bool:
    """Inter-provider fallback eligibility."""
    cls = _norm_cls(classification)
    eff_primary = failed_budget if failed_budget is not None else primary_budget
    if cls == "output_budget_exhausted":
        return fallback_budget > eff_primary
    return cls in {
        "timeout",
        "network_ssl",
        "network_connection",
        "provider_5xx",
        "invalid_response",
        "empty_response",
        "rate_limit",
        "auth",
        "insufficient_balance_quota",
        "forbidden_geo",
    }


def order_kie_vision_candidates(
    primary_model: str,
    selectable_models: Sequence[str] | None = None,
) -> list[str]:
    """Order KIE multimodal candidates based on authoritative catalog.

    Preferred valid model first, followed by at most one distinct valid alternate.
    """
    if selectable_models is None:
        try:
            from provider_models import PROVIDER_KIE, get_selectable_models
            catalog = list(get_selectable_models(PROVIDER_KIE, "vision"))
        except Exception:
            catalog = []
    else:
        catalog = [str(m).strip() for m in selectable_models if str(m).strip()]

    p = (primary_model or "").strip()
    candidates: list[str] = []

    if p and (not catalog or p in catalog):
        candidates.append(p)
    elif catalog:
        candidates.append(catalog[0])

    for m in catalog:
        if m not in candidates:
            candidates.append(m)
            break

    return candidates


def resolve_effective_vision_fallback(
    primary_provider_or_config: Any,
    allow_fallback: bool | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> tuple[bool, str | None, str | None]:
    """Resolve effective fallback configuration without mutating database state.

    Accepts either an AIConfig object or scalar fields.
    Returns (is_active, fallback_provider, fallback_model).
    """
    if allow_fallback is None and hasattr(primary_provider_or_config, "vision_provider"):
        config = primary_provider_or_config
        primary_provider = getattr(config, "vision_provider", None)
        allow_fallback = bool(getattr(config, "allow_vision_fallback", False))
        fallback_provider = getattr(config, "vision_fallback_provider", None)
        fallback_model = getattr(config, "vision_fallback_model", None)
    else:
        primary_provider = primary_provider_or_config

    if not allow_fallback:
        return False, fallback_provider, fallback_model

    if not fallback_provider or not fallback_model:
        return False, fallback_provider, fallback_model

    if str(fallback_provider).strip().lower() == str(primary_provider or "").strip().lower():
        return False, fallback_provider, fallback_model

    return True, fallback_provider, fallback_model
