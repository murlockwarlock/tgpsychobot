from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from aiogram.methods import SendMessage, SendPhoto

from database import FollowupStep, User
from followups import FollowupStepSendResult, FollowupTransportRegistry, emit_followup_step
from canonical_content import canonical_markup_to_html
from mailing_utils import mailing_text_to_html, parse_mailing_text, render_mailing_text, send_mailing_content
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.identity import MAX_ID_OFFSET
from response_buttons import extract_response_buttons


class TelegramBoundary:
    def __init__(self):
        self.calls = []

    async def send_message(self, chat_id, text, **kwargs):
        kwargs = {"chat_id": chat_id, "text": text, **kwargs}
        SendMessage(**kwargs)
        self.calls.append(("message", kwargs))
        return SimpleNamespace(message_id=1)

    async def send_photo(self, chat_id, photo, **kwargs):
        kwargs = {"chat_id": chat_id, "photo": photo, **kwargs}
        SendPhoto(**kwargs)
        self.calls.append(("photo", kwargs))
        return SimpleNamespace(message_id=1)


class MaxResponse:
    status = 200
    content_type = "application/json"

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return json.dumps(self.payload)

    async def json(self):
        return self.payload


class MaxBoundary:
    def __init__(self):
        self.requests = []

    def request(self, method, url, *, params=None, json=None):
        self.requests.append({"method": method, "url": url, "params": params or {}, "body": json})
        return MaxResponse({"message": {"body": {"mid": "max-boundary-1"}}})


def _formatted_source() -> str:
    return (
        '<b>Жирный</b> <i>курсив</i> <u>подчеркнутый</u> '
        '<s>зачеркнутый</s> <a href="https://example.com">ссылка</a> '
        '<blockquote>цитата</blockquote>\n\n'
        '[Продолжить](btn:continue)\n'
        '[Сайт](https://example.com)'
    )


def test_canonical_markup_renders_formatted_html_once_and_keeps_mixed_markdown():
    source = '<b>Привет</b> [сайт](https://example.com)'

    assert canonical_markup_to_html(source) == '<b>Привет</b> <a href="https://example.com">сайт</a>'
    assert mailing_text_to_html(source) == '<b>Привет</b> <a href="https://example.com">сайт</a>'


def test_canonical_markup_keeps_unsupported_wrapped_button_visible():
    source = '<b>[Начать](btn:start)</b>'

    clean, rows = extract_response_buttons(source)

    assert rows == []
    assert mailing_text_to_html(clean) == source


def test_malformed_button_declaration_remains_visible():
    source = "<b>Привет</b>\n[Сломанная](btn:)"

    clean, rows = parse_mailing_text(source)

    assert rows == []
    assert mailing_text_to_html(clean) == source


def test_mailing_personalization_cannot_create_or_corrupt_button_declarations():
    source = '[{name}](btn:start)\n<b>Здравствуйте, {name}</b>'
    user = SimpleNamespace(
        name='A]\n[Run](btn:unexpected)',
        username=None,
        birth_day=None,
        birth_month=None,
        birth_year=None,
    )

    rendered = render_mailing_text(source, user)
    clean, rows = extract_response_buttons(rendered.visible_text)

    assert rendered.visible_text == '<b>Здравствуйте, A] [Run](btn:unexpected)</b>'
    assert rows == []
    assert [(button.text, button.kind, button.value) for row in rendered.response_button_rows for button in row] == [
        ("{name}", "action", "start"),
    ]
    assert clean == rendered.visible_text


@pytest.mark.parametrize(
    "recipient_value",
    [
        "[Run](btn:unexpected)",
        "[Site](https://example.com)",
        "] [ ( )\n< > & btn: https://",
    ],
)
def test_mailing_personalization_only_changes_visible_body(recipient_value):
    rendered = render_mailing_text("Привет, {name}", SimpleNamespace(name=recipient_value))

    assert rendered.response_button_rows == []
    clean, rows = extract_response_buttons(rendered.visible_text)
    assert clean == rendered.visible_text
    assert rows == []


def test_mailing_personalization_preserves_authored_button_structure():
    source = "Привет, {name}\n\n[Продолжить](btn:continue)"
    user = SimpleNamespace(name="[Run](btn:unexpected)")

    rendered = render_mailing_text(source, user)

    assert rendered.visible_text == "Привет, [Run](btn:unexpected)"
    assert [
        [(button.text, button.kind, button.value) for button in row]
        for row in rendered.response_button_rows
    ] == [[("Продолжить", "action", "continue")]]


@pytest.mark.asyncio
async def test_formatted_telegram_mailing_uses_validated_transport_and_buttons():
    source = _formatted_source()
    bot = TelegramBoundary()
    mailing = SimpleNamespace(
        text=source,
        media_file_id=None,
        media_file_type=None,
        media_position="media_top",
    )

    await send_mailing_content(bot, 42, mailing)

    kind, payload = bot.calls[-1]
    assert kind == "message"
    assert payload["text"] == (
        '<b>Жирный</b> <i>курсив</i> <u>подчеркнутый</u> '
        '<s>зачеркнутый</s> <a href="https://example.com">ссылка</a> '
        '<blockquote>цитата</blockquote>'
    )
    buttons = [button for row in payload["reply_markup"].inline_keyboard for button in row]
    assert [(button.text, button.callback_data, button.url) for button in buttons] == [
        ("Продолжить", "ai_btn:continue", None),
        ("Сайт", None, "https://example.com"),
    ]


@pytest.mark.asyncio
async def test_formatted_telegram_mailing_media_caption_uses_validated_transport():
    bot = TelegramBoundary()
    mailing = SimpleNamespace(
        text='<i>Подпись</i>\n\n[Продолжить](btn:continue)',
        media_file_id="photo-token",
        media_file_type="photo",
        media_position="media_top",
    )

    await send_mailing_content(bot, 42, mailing)

    kind, payload = bot.calls[-1]
    assert kind == "photo"
    assert payload["caption"] == "<i>Подпись</i>"
    assert payload["reply_markup"].inline_keyboard[0][0].callback_data == "ai_btn:continue"


@pytest.mark.asyncio
async def test_formatted_static_followup_uses_validated_telegram_and_max_payloads():
    source = _formatted_source()
    clean, rows = extract_response_buttons(source)
    step = FollowupStep(message_type="static", message_text=source, delay_minutes=1)
    result = FollowupStepSendResult(source, clean, None, rows)

    telegram = TelegramBoundary()
    await emit_followup_step(
        FollowupTransportRegistry(telegram=telegram),
        user=User(id=42),
        step=step,
        send_result=result,
    )
    _, telegram_payload = telegram.calls[-1]
    assert telegram_payload["text"] == mailing_text_to_html(clean)
    assert "&lt;b&gt;" not in telegram_payload["text"]
    assert telegram_payload["reply_markup"].inline_keyboard[0][0].callback_data == "ai_btn:continue"

    max_transport = MaxBoundary()
    max_client = MaxApiClient("token", "https://max.test")
    max_client._session = max_transport
    await emit_followup_step(
        FollowupTransportRegistry(max_client=max_client),
        user=User(id=MAX_ID_OFFSET + 55),
        step=step,
        send_result=result,
    )
    max_payload = max_transport.requests[-1]["body"]
    assert max_payload["text"] == mailing_text_to_html(clean)
    assert "&lt;b&gt;" not in max_payload["text"]
    assert max_payload["attachments"][0]["payload"]["buttons"][0][0] == {
        "type": "callback",
        "text": "Продолжить",
        "payload": "ai_btn:continue",
    }
