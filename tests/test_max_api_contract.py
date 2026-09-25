from __future__ import annotations

import json

import pytest

from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.settings import clear_settings_cache, get_settings


class _Response:
    def __init__(self, *, status: int = 200, body: str = "", content_type: str = ""):
        self.status = status
        self._body = body
        self.content_type = content_type

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return self._body

    async def json(self):
        return json.loads(self._body)


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return self.responses.pop(0)

    def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_answer_callback_does_not_create_implicit_notification():
    client = MaxApiClient("test-token", "https://max.test")
    session = _Session([_Response(body='{"success":true}', content_type="application/json")])
    client._session = session

    await client.answer_callback("callback-id")

    method, url, kwargs = session.requests[0]
    assert method == "POST"
    assert url == "https://max.test/answers"
    assert kwargs["json"] is None


@pytest.mark.asyncio
async def test_answer_callback_keeps_explicit_notification():
    client = MaxApiClient("test-token", "https://max.test")
    session = _Session([_Response(body='{"success":true}', content_type="application/json")])
    client._session = session

    await client.answer_callback("callback-id", notification="Сохранено")

    assert session.requests[0][2]["json"] == {"notification": "Сохранено"}


@pytest.mark.asyncio
async def test_upload_file_reuses_create_token_when_upload_response_is_empty():
    client = MaxApiClient("test-token", "https://max.test")
    session = _Session(
        [
            _Response(body='{"url":"https://upload.test/file","token":"max-token"}', content_type="application/json"),
            _Response(status=200, body="", content_type=""),
        ]
    )
    client._session = session

    result = await client.upload_file("video", __file__)

    assert result["token"] == "max-token"
    assert session.requests[0][1] == "https://max.test/uploads"
    assert session.requests[1][1] == "https://upload.test/file"


@pytest.mark.asyncio
async def test_upload_file_extracts_image_token_from_photos_response():
    client = MaxApiClient("test-token", "https://max.test")
    session = _Session(
        [
            _Response(body='{"url":"https://upload.test/file"}', content_type="application/json"),
            _Response(
                body='{"photos":{"photo-id":{"token":"image-token"}}}',
                content_type="application/json",
            ),
        ]
    )
    client._session = session

    result = await client.upload_file("image", __file__)

    assert result["token"] == "image-token"
    assert result["photos"]["photo-id"]["token"] == "image-token"


def test_max_api_default_uses_current_platform_endpoint(monkeypatch):
    monkeypatch.delenv("MAX_API_BASE", raising=False)
    clear_settings_cache()

    assert get_settings().max_api_base == "https://platform-api2.max.ru"

    clear_settings_cache()
