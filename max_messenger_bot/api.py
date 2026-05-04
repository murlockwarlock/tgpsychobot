from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp

from .logging_utils import configure_logging, get_max_logger

configure_logging()
log = get_max_logger("api")


class MaxApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error_code: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code
        self.payload = payload or {}

    @property
    def is_attachment_not_ready(self) -> bool:
        if self.error_code == "attachment.not.ready":
            return True
        text = str(self).lower()
        return "attachment.not.ready" in text


class MaxApiClient:
    def __init__(self, token: str, base_url: str) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "MaxApiClient":
        self._session = aiohttp.ClientSession(
            headers={"Authorization": self.token},
            timeout=aiohttp.ClientTimeout(total=120),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("MAX API session is not initialized")
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        # yarl rejects bool query param values — convert them to lowercase strings
        if params:
            params = {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}
        try:
            async with self.session.request(method, url, params=params, json=json_data) as response:
                text = await response.text()
                if response.status != expected_status:
                    error_payload: dict[str, Any] | None = None
                    error_code: str | None = None
                    try:
                        error_payload = json.loads(text) if text else None
                    except json.JSONDecodeError:
                        error_payload = None
                    if isinstance(error_payload, dict):
                        error_code = error_payload.get("code") or error_payload.get("error_code")
                    log.error(
                        "MAX API request failed method=%s path=%s status=%s params=%s body=%s response=%s",
                        method,
                        path,
                        response.status,
                        params,
                        json_data,
                        text[:2000],
                    )
                    raise MaxApiError(
                        f"{method} {path} failed: HTTP {response.status}: {text}",
                        status=response.status,
                        error_code=error_code,
                        payload=error_payload,
                    )
                if not text:
                    return {}
                if response.content_type != "application/json":
                    return {"raw": text}
                return await response.json()
        except aiohttp.ClientError as exc:
            log.exception("MAX transport error method=%s path=%s params=%s body=%s", method, path, params, json_data)
            raise MaxApiError(f"{method} {path} transport error: {exc}") from exc
        except asyncio.TimeoutError as exc:
            log.exception("MAX timeout method=%s path=%s params=%s body=%s", method, path, params, json_data)
            raise MaxApiError(f"{method} {path} timeout") from exc

    async def get_me(self) -> dict[str, Any]:
        return await self._request("GET", "/me")

    async def set_webhook(self, url: str, secret: str | None, update_types: tuple[str, ...]) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": url, "update_types": list(update_types)}
        if secret:
            payload["secret"] = secret
        result = await self._request("POST", "/subscriptions", json_data=payload)
        log.info("MAX webhook subscription updated url=%s update_types=%s", url, list(update_types))
        return result

    async def get_updates(self, marker: int | None, timeout: int, limit: int, update_types: tuple[str, ...]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "timeout": timeout,
            "limit": limit,
        }
        if marker is not None:
            params["marker"] = marker
        if update_types:
            params["types"] = ",".join(update_types)
        result = await self._request("GET", "/updates", params=params)
        updates = result.get("updates", []) if isinstance(result, dict) else []
        log.info("MAX updates fetched marker=%s count=%s", marker, len(updates))
        return result

    async def send_message(
        self,
        *,
        user_id: int | None = None,
        chat_id: int | None = None,
        text: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        format_: str = "html",
        disable_link_preview: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"disable_link_preview": disable_link_preview}
        if user_id is not None:
            params["user_id"] = user_id
        if chat_id is not None:
            params["chat_id"] = chat_id
        # In MAX HTML mode, \n is ignored — normalize to <br/> for consistent rendering
        normalized_text = (text or "").replace("\n", "<br/>") if format_ == "html" else (text or "")
        body: dict[str, Any] = {"text": normalized_text}
        if attachments is not None:
            body["attachments"] = attachments
        if format_:
            body["format"] = format_
        result = await self._request("POST", "/messages", params=params, json_data=body)
        log.info(
            "MAX message sent user_id=%s chat_id=%s text_len=%s attachments=%s",
            user_id,
            chat_id,
            len(normalized_text),
            len(attachments or []),
        )
        return result

    async def edit_message(
        self,
        message_id: str,
        *,
        text: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        format_: str = "html",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if text is not None:
            body["text"] = text
        if attachments is not None:
            body["attachments"] = attachments
        if format_:
            body["format"] = format_
        result = await self._request("PUT", "/messages", params={"message_id": message_id}, json_data=body)
        log.info("MAX message edited message_id=%s text_len=%s attachments=%s", message_id, len(text or ""), len(attachments or []))
        return result

    async def delete_message(self, message_id: str) -> dict[str, Any]:
        return await self._request("DELETE", "/messages", params={"message_id": message_id}, expected_status=200)

    async def answer_callback(
        self,
        callback_id: str,
        *,
        text: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        notification: str | None = None,
        format_: str = "html",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if text is not None or attachments is not None:
            payload["message"] = {"text": text or "", "format": format_}
            if attachments is not None:
                payload["message"]["attachments"] = attachments
        if notification:
            payload["notification"] = notification
        result = await self._request("POST", "/answers", params={"callback_id": callback_id}, json_data=payload)
        log.info("MAX callback answered callback_id=%s has_message=%s has_notification=%s", callback_id, "message" in payload, bool(notification))
        return result

    async def create_upload(self, media_type: str) -> dict[str, Any]:
        return await self._request("POST", "/uploads", params={"type": media_type})

    async def upload_file(self, media_type: str, file_path: str | Path) -> dict[str, Any]:
        create_result = await self.create_upload(media_type)
        upload_url = create_result["url"]
        try:
            with Path(file_path).open("rb") as file_handle:
                data = aiohttp.FormData()
                data.add_field("data", file_handle)
                async with self.session.post(upload_url, data=data) as response:
                    text = await response.text()
                    if response.status >= 400:
                        log.error(
                            "MAX upload failed media_type=%s file=%s status=%s response=%s",
                            media_type,
                            file_path,
                            response.status,
                            text[:2000],
                        )
                        raise MaxApiError(f"Upload failed: HTTP {response.status}: {text}")
                    result = await response.json()
        except aiohttp.ClientError as exc:
            log.exception("MAX upload transport error media_type=%s file=%s", media_type, file_path)
            raise MaxApiError(f"Upload transport error: {exc}") from exc
        if media_type in {"audio", "video"}:
            result.setdefault("token", create_result.get("token"))
        return result

    async def send_media_attachment(
        self,
        *,
        chat_id: int,
        media_type: str,
        token: str,
        caption: str | None = None,
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        attachments = [{"type": media_type, "payload": {"token": token}}]
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.send_message(chat_id=chat_id, text=caption or "", attachments=attachments)
            except MaxApiError as exc:
                if not exc.is_attachment_not_ready or attempt >= max_attempts:
                    raise
                delay = min(5.0, 0.5 * attempt)
                log.warning(
                    "MAX attachment not ready media_type=%s token=%s chat_id=%s attempt=%s/%s delay=%.1fs",
                    media_type,
                    token,
                    chat_id,
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
        raise MaxApiError("MAX media send failed after retries")

    async def download_attachment(self, token: str, url: str | None = None) -> bytes:
        """Download a user-sent attachment by URL (preferred) or via API token."""
        import aiohttp as _aiohttp

        if url:
            async with _aiohttp.ClientSession(timeout=_aiohttp.ClientTimeout(total=60)) as http:
                async with http.get(url) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    log.warning("MAX attachment direct download failed status=%s url=%s", resp.status, url)

        result = await self._request("GET", f"/uploads/{token}")
        download_url = result.get("url") or result.get("download_url")
        if download_url:
            async with _aiohttp.ClientSession(timeout=_aiohttp.ClientTimeout(total=60)) as http:
                async with http.get(download_url) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    raise MaxApiError(f"MAX attachment download via resolved URL failed: HTTP {resp.status}")

        raise MaxApiError(f"Cannot resolve download URL for token={token}")
