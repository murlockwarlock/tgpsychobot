"""
Comprehensive unit and integration tests for Event Content Validator, safe limits,
custom emoji processing, media delivery modes, and admin boundaries.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramBadRequest

from database import EventScenario, EventStep, User
from event_content_validator import (
    ALLOWED_EVENT_HTML_TAGS,
    MAX_ATTACHMENTS_WITH_KEYBOARD_LIMIT,
    MAX_TEXT_LIMIT_CONSERVATIVE,
    MAX_TEXT_LIMIT_TECHNICAL,
    MAX_TOTAL_ATTACHMENTS_LIMIT,
    MEDIA_FORMAT_CAPTION,
    MEDIA_FORMAT_LEGACY_AUTO,
    MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
    MEDIA_FORMAT_SEPARATE_TEXT_TOP,
    TELEGRAM_CAPTION_LIMIT_CONSERVATIVE,
    TELEGRAM_CAPTION_LIMIT_TECHNICAL,
    TELEGRAM_TEXT_LIMIT_CONSERVATIVE,
    TELEGRAM_TEXT_LIMIT_TECHNICAL,
    EventContentValidationError,
    TelegramHtmlValidator,
    calculate_available_description_chars,
    get_media_format,
    parse_visible_telegram_text,
    render_plain_text_for_max,
    resolve_media_mode_and_order,
    strip_telegram_custom_emoji,
    utf16_code_units_len,
    validate_scenario_content,
)
from event_admin_service import (
    create_scenario,
    get_scenario_summary,
    replace_scenario_media,
    update_scenario_fields,
)
from event_delivery_worker import _build_delivery_plan, _send_delivery_part


def test_format_mappings_and_resolutions():
    assert resolve_media_mode_and_order(MEDIA_FORMAT_SEPARATE_MEDIA_TOP) == ("separate", "media_top")
    assert resolve_media_mode_and_order(MEDIA_FORMAT_SEPARATE_TEXT_TOP) == ("separate", "text_top")
    assert resolve_media_mode_and_order(MEDIA_FORMAT_CAPTION) == ("caption", "media_top")
    assert resolve_media_mode_and_order(MEDIA_FORMAT_LEGACY_AUTO) == ("legacy_auto", "media_top")

    assert get_media_format("caption", "media_top") == MEDIA_FORMAT_CAPTION
    assert get_media_format("separate", "text_top") == MEDIA_FORMAT_SEPARATE_TEXT_TOP
    assert get_media_format("separate", "media_top") == MEDIA_FORMAT_SEPARATE_MEDIA_TOP
    assert get_media_format("legacy_auto", "media_top") == MEDIA_FORMAT_LEGACY_AUTO


def test_utf16_code_units_calculation():
    # ASCII chars: 1 code unit
    assert utf16_code_units_len("Hello") == 5
    # Cyrillic chars: 1 code unit in BMP
    assert utf16_code_units_len("Привет") == 6
    # 4-byte astral plane emoji: 2 UTF-16 code units (surrogate pair)
    assert utf16_code_units_len("👍") == 2
    assert utf16_code_units_len("🌟") == 2
    assert utf16_code_units_len("Привет 👍") == 7 + 2


def test_telegram_html_validator_custom_emoji():
    # Valid tg-emoji
    raw = 'Привет <tg-emoji emoji-id="5368324170671202286">👍</tg-emoji> мир!'
    vis, length, errors = parse_visible_telegram_text(raw)
    assert not errors
    assert vis == "Привет 👍 мир!"
    assert length == utf16_code_units_len("Привет 👍 мир!")

    # Unbalanced tags
    unbalanced = '<b>Тест <tg-emoji emoji-id="123">😊</b>'
    vis, length, errors = parse_visible_telegram_text(unbalanced)
    assert errors
    assert any("Несбалансированный" in err or "unbalanced" in err for err in errors)

    # Disallowed tags
    disallowed = '<script>alert("hack")</script>'
    vis, length, errors = parse_visible_telegram_text(disallowed)
    assert errors
    assert any("Недопустимый HTML-тег" in err for err in errors)


def test_strip_custom_emoji_and_render_for_max():
    tg_text = 'Приглашаем на <tg-emoji emoji-id="123456">🔥</tg-emoji> вебинар <b>"AI 2026"</b>!'
    # strip_telegram_custom_emoji leaves HTML but replaces tg-emoji with fallback
    stripped = strip_telegram_custom_emoji(tg_text)
    assert stripped == 'Приглашаем на 🔥 вебинар <b>"AI 2026"</b>!'

    # render_plain_text_for_max strips HTML tags and produces plain text
    plain = render_plain_text_for_max(tg_text)
    assert plain == 'Приглашаем на 🔥 вебинар "AI 2026"!'


def test_boundary_checks_telegram_caption():
    # Telegram caption conservative limit: 900 chars
    scen = SimpleNamespace(
        id=1,
        code="caption_probe",
        event_type="webinar",
        title="Short Title",
        starts_at=datetime.utcnow() + timedelta(days=2),
        duration_minutes=90,
        place=None,
        link="https://zoom.us/test",
        description="",
        media_mode="caption",
        media_order="media_top",
    )
    media = [SimpleNamespace(file_type="photo", file_id="p1")]

    # Caption with 50 chars description -> valid
    scen.description = "D" * 50
    res_valid = validate_scenario_content(scen, media=media, media_format=MEDIA_FORMAT_CAPTION)
    assert res_valid.is_valid

    # Caption with description making total message exceed 900 -> invalid
    scen.description = "D" * 850
    res_invalid = validate_scenario_content(scen, media=media, media_format=MEDIA_FORMAT_CAPTION)
    assert not res_invalid.is_valid
    assert any("подпись" in v.error_message for v in res_invalid.violations)


def test_boundary_checks_telegram_text_and_max():
    scen = SimpleNamespace(
        id=1,
        code="text_probe",
        event_type="webinar",
        title="Title",
        starts_at=datetime.utcnow() + timedelta(days=2),
        duration_minutes=90,
        place=None,
        link="https://zoom.us/test",
        description="",
        media_mode="separate",
        media_order="media_top",
    )
    media = [SimpleNamespace(file_type="photo", file_id="p1")]

    # Conservative limit is 4000 TG / 3900 MAX
    # 3500 chars -> valid
    scen.description = "A" * 3500
    res = validate_scenario_content(scen, media=media, media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP)
    assert res.is_valid

    # 3950 chars -> exceeds MAX conservative (3900)
    scen.description = "A" * 3950
    res_max_excess = validate_scenario_content(scen, media=media, media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP)
    assert not res_max_excess.is_valid
    assert any("MAX" in v.error_message for v in res_max_excess.violations)


def test_mode_c_requires_exactly_one_media():
    scen = SimpleNamespace(
        id=1,
        code="mode_c_count",
        event_type="webinar",
        title="Webinar",
        starts_at=datetime.utcnow() + timedelta(days=2),
        duration_minutes=90,
        place=None,
        link="https://zoom.us/test",
        description="Brief text",
        media_mode="caption",
        media_order="media_top",
    )

    # 0 media -> invalid for Mode C
    res_0 = validate_scenario_content(scen, media=[], media_format=MEDIA_FORMAT_CAPTION)
    assert not res_0.is_valid
    assert any("требует ровно 1 медиа-вложение" in v.error_message for v in res_0.violations)

    # 2 media -> invalid for Mode C
    media_2 = [
        SimpleNamespace(file_type="photo", file_id="p1"),
        SimpleNamespace(file_type="photo", file_id="p2"),
    ]
    res_2 = validate_scenario_content(scen, media=media_2, media_format=MEDIA_FORMAT_CAPTION)
    assert not res_2.is_valid
    assert any("требует ровно 1 медиа-вложение" in v.error_message for v in res_2.violations)

    # 1 media -> valid
    media_1 = [SimpleNamespace(file_type="photo", file_id="p1")]
    res_1 = validate_scenario_content(scen, media=media_1, media_format=MEDIA_FORMAT_CAPTION)
    assert res_1.is_valid


def test_max_total_attachments_boundary():
    scen = SimpleNamespace(
        id=1,
        code="max_attach",
        event_type="webinar",
        title="Title",
        starts_at=datetime.utcnow() + timedelta(days=2),
        duration_minutes=90,
        place=None,
        link="https://zoom.us/test",
        description="Text",
        media_mode="separate",
        media_order="media_top",
    )
    # 11 media + 1 control = 12 total attachments in MAX -> valid
    media_11 = [SimpleNamespace(file_type="photo", file_id=f"p{i}") for i in range(11)]
    res_11 = validate_scenario_content(scen, media=media_11, media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP)
    assert res_11.is_valid


def test_calculate_available_description_chars():
    scen = SimpleNamespace(
        id=1,
        code="budget_calc",
        event_type="webinar",
        title="A" * 100,
        starts_at=datetime.utcnow() + timedelta(days=2),
        duration_minutes=90,
        place=None,
        link="https://zoom.us/j/1234567890",
        description="",
        media_mode="separate",
        media_order="media_top",
    )
    budgets = calculate_available_description_chars(
        scen,
        media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
        steps=[],
    )
    assert "telegram" in budgets
    assert "max" in budgets
    assert "conservative" in budgets
    assert budgets["conservative"] > 2000
    assert budgets["conservative"] <= 3900


@pytest.mark.asyncio
async def test_auto_transition_from_caption_to_separate_on_media_deletion(test_session_maker):
    scen = await create_scenario(
        test_session_maker,
        code="auto_trans_event",
        event_type="webinar",
        title="Auto transition test",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        duration_minutes=60,
        link="https://zoom.us/test",
        media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
    )
    # Add 1 media
    await replace_scenario_media(
        test_session_maker,
        scen.id,
        [{"file_type": "photo", "file_id": "photo_1", "source_platform": "telegram"}],
    )
    # Switch to caption format now that 1 media is attached
    await update_scenario_fields(test_session_maker, scen.id, media_format=MEDIA_FORMAT_CAPTION)
    summary_before = await get_scenario_summary(test_session_maker, scen.id)
    assert summary_before.media_mode == "caption"

    # Now remove the media (empty list) -> should auto-transition to separate to prevent deadlocks
    await replace_scenario_media(test_session_maker, scen.id, [])
    summary_after = await get_scenario_summary(test_session_maker, scen.id)
    assert summary_after.media_mode == "separate"
    assert summary_after.media_count == 0


@pytest.mark.asyncio
async def test_update_scenario_fields_atomic_validation_rollback(test_session_maker):
    scen = await create_scenario(
        test_session_maker,
        code="atomic_rollback",
        event_type="webinar",
        title="Initial Valid Title",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        duration_minutes=60,
        link="https://zoom.us/test",
        media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
    )

    # Attempt to update with invalid HTML in description
    with pytest.raises(EventContentValidationError) as exc_info:
        await update_scenario_fields(
            test_session_maker,
            scen.id,
            description="<b>Unclosed bold tag",
        )
    assert "HTML" in str(exc_info.value)

    # Verify that the DB was not modified
    summary = await get_scenario_summary(test_session_maker, scen.id)
    assert summary.description is None or summary.description == ""


@pytest.mark.asyncio
async def test_max_save_rejects_html_transport_overflow_even_when_visible_text_fits(test_session_maker):
    scen = await create_scenario(
        test_session_maker,
        code="max_html_transport_boundary",
        event_type="webinar",
        title="HTML boundary",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/html-boundary",
        media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
    )
    raw_html = "<b>x</b>" * 520
    assert len(raw_html) > MAX_TEXT_LIMIT_TECHNICAL
    assert len("x" * 520) < MAX_TEXT_LIMIT_TECHNICAL

    with pytest.raises(EventContentValidationError) as exc_info:
        await update_scenario_fields(test_session_maker, scen.id, description=raw_html)

    assert "MAX" in str(exc_info.value)
    assert "разметкой" in str(exc_info.value)
    summary = await get_scenario_summary(test_session_maker, scen.id)
    assert summary.description is None


def test_delivery_plan_modes_a_b_c():
    scen_a = SimpleNamespace(
        id=1,
        code="mode_a",
        media_mode="separate",
        media_order="media_top",
        media=[SimpleNamespace(file_type="photo", file_id="p1", max_token="t1")],
    )
    # Mode A: separate media top -> media parts first, then text part
    plan_a = _build_delivery_plan(scen_a, 1, "Hello text", include_media=True)
    assert len(plan_a) == 2
    assert plan_a[0]["kind"] == "media"
    assert plan_a[1]["kind"] == "text"

    # Mode B: separate text top -> text first, then media parts
    scen_b = SimpleNamespace(
        id=2,
        code="mode_b",
        media_mode="separate",
        media_order="text_top",
        media=[SimpleNamespace(file_type="photo", file_id="p1", max_token="t1")],
    )
    plan_b = _build_delivery_plan(scen_b, 2, "Hello text", include_media=True)
    assert len(plan_b) >= 2
    assert plan_b[0]["kind"] == "text"
    assert plan_b[1]["kind"] == "media"

    # Mode C: caption mode with 1 media -> exactly 1 media_caption message
    scen_c = SimpleNamespace(
        id=3,
        code="mode_c",
        media_mode="caption",
        media_order="media_top",
        media=[SimpleNamespace(file_type="photo", file_id="p1", max_token="t1")],
    )
    plan_c = _build_delivery_plan(scen_c, 3, "Hello caption", include_media=True)
    assert len(plan_c) == 1
    assert plan_c[0]["kind"] == "media_caption"
    assert plan_c[0]["caption"] == "Hello caption"


@pytest.mark.asyncio
async def test_telegram_custom_emoji_fallback_on_unauthorized_error():
    # When bot lacks permission to send custom emoji, Telegram returns 400 Bad Request
    # delivery worker must catch this, strip <tg-emoji> tags to Unicode fallbacks, and retry.
    bot = AsyncMock()
    method_mock = MagicMock()
    forbidden_emoji_error = TelegramBadRequest(
        method=method_mock,
        message="Bad Request: can't use custom emoji",
    )

    # 1. Text message fallback test
    text_payload = {
        "kind": "text",
        "text": 'Привет! <tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji> Встречаемся онлайн!',
    }
    second_success = MagicMock(message_id=777)
    bot.send_message = AsyncMock(side_effect=[forbidden_emoji_error, second_success])

    msg_id = await _send_delivery_part(
        text_payload,
        platform="telegram",
        user_id=123456,
        event_id=1,
        is_last=True,
        bot=bot,
        max_client=None,
    )
    assert msg_id == "777"
    assert bot.send_message.call_count == 2
    # Verify second call had custom emoji stripped to Unicode fallback
    retry_call_args = bot.send_message.call_args_list[1]
    assert retry_call_args.kwargs["text"] == "Привет! 🔥 Встречаемся онлайн!"

    # 2. Caption mode fallback test
    caption_payload = {
        "kind": "media_caption",
        "items": [{"file_type": "photo", "file_id": "test_photo_id", "source_platform": "telegram"}],
        "caption": 'Фото с <tg-emoji emoji-id="9999999999999999999">🚀</tg-emoji> запуска!',
    }
    bot.send_photo = AsyncMock(side_effect=[forbidden_emoji_error, MagicMock(message_id=888)])
    cap_msg_id = await _send_delivery_part(
        caption_payload,
        platform="telegram",
        user_id=123456,
        event_id=1,
        is_last=True,
        bot=bot,
        max_client=None,
    )
    assert cap_msg_id == "888"
    assert bot.send_photo.call_count == 2
    retry_photo_args = bot.send_photo.call_args_list[1]
    assert retry_photo_args.kwargs["caption"] == "Фото с 🚀 запуска!"


def test_custom_emoji_html_roundtrip_preservation():
    # Verify tg-emoji tags with IDs are parsed without losing attributes or double-escaping
    raw = '<b>Мероприятие:</b> <tg-emoji emoji-id="12345">💎</tg-emoji> &lt;Код&gt; <i>курса</i>'
    vis, utf16_len, errors = parse_visible_telegram_text(raw)
    assert not errors
    assert vis == 'Мероприятие: 💎 <Код> курса'

    stripped = strip_telegram_custom_emoji(raw)
    assert stripped == '<b>Мероприятие:</b> 💎 &lt;Код&gt; <i>курса</i>'
    assert "<tg-emoji" not in stripped
    assert "💎" in stripped

    # Plain text for MAX
    max_text = render_plain_text_for_max(raw)
    assert max_text == 'Мероприятие: 💎 <Код> курса'


def test_title_validation_visible_length_vs_storage_length():
    from event_admin_service import validate_scenario_input

    future = datetime.utcnow() + timedelta(days=2)
    common_kwargs = {
        "code": "valid_code",
        "event_type": "webinar",
        "starts_at_utc": future,
        "duration_minutes": 60,
        "place": None,
        "link": "https://example.test",
    }

    # 1. Valid title under 1000 chars
    valid_title = "Тестовое мероприятие " + ("A" * 500)
    validate_scenario_input(title=valid_title, **common_kwargs)

    # 2. Visible length > 1000 chars
    long_visible = "A" * 1001
    with pytest.raises(ValueError) as exc:
        validate_scenario_input(title=long_visible, **common_kwargs)
    assert "должно содержать от 1 до 1000 символов" in str(exc.value)

    # 3. Visible length <= 1000 chars, but raw HTML with tg-emoji tags > 1000 chars (storage limit)
    # Each tg-emoji wrapper adds ~40 chars for 2 chars of visible emoji
    emojis = '<tg-emoji emoji-id="1234567890123456789">🔥</tg-emoji>' * 25
    title_raw = ("A" * 50) + emojis
    vis, utf16_len, _ = parse_visible_telegram_text(title_raw)
    assert len(vis) < 1000  # Visible length is only ~75 characters
    assert len(title_raw) > 1000  # Raw storage length is over 1300 characters

    with pytest.raises(ValueError) as exc:
        validate_scenario_input(title=title_raw, **common_kwargs)
    assert "предел хранения (1000 символов)" in str(exc.value)

