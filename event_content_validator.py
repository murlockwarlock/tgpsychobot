"""
Event Content Validator for Telegram and MAX bot messaging.

Enforces safe limits for event announcements, confirmations, reminders, and live notices:
- Telegram Bot API:
  - Technical limit: 4096 chars (text), 1024 chars (caption) measured in UTF-16 code units.
  - Project conservative working boundary: 4000 chars (text), 900 chars (caption).
  - Preserves Telegram custom emoji (<tg-emoji emoji-id="...">fallback</tg-emoji>).
  - Validates HTML entity syntax and structure.
- MAX Bot API:
  - Technical limit: 4000 chars.
  - Project conservative working boundary: 3900 chars.
  - Attachments limit: at most 12 total attachments per message (inline keyboard counts as 1 attachment).
  - Safe Unicode fallback for Telegram custom emoji (extracting inner fallback emoji).
  - Rate limit consideration: 2 messages/second per dialog.
- Media formats:
  - Mode A: separate_media_top (text separate, media above)
  - Mode B: separate_text_top (text separate, media below)
  - Mode C: caption (single photo/video together with text)
  - legacy_auto: historical compatibility mode for pre-existing records.
"""

from __future__ import annotations

import html
from html.parser import HTMLParser
import re
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

# Canonical format identifiers
MEDIA_FORMAT_LEGACY_AUTO = "legacy_auto"
MEDIA_FORMAT_SEPARATE_MEDIA_TOP = "separate_media_top"
MEDIA_FORMAT_SEPARATE_TEXT_TOP = "separate_text_top"
MEDIA_FORMAT_CAPTION = "caption"

CANONICAL_MEDIA_FORMATS = (
    MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
    MEDIA_FORMAT_SEPARATE_TEXT_TOP,
    MEDIA_FORMAT_CAPTION,
)
ALL_MEDIA_FORMATS = (MEDIA_FORMAT_LEGACY_AUTO, *CANONICAL_MEDIA_FORMATS)

# Platform limits
TELEGRAM_TEXT_LIMIT_CONSERVATIVE = 4000
TELEGRAM_TEXT_LIMIT_TECHNICAL = 4096
TELEGRAM_CAPTION_LIMIT_CONSERVATIVE = 900
TELEGRAM_CAPTION_LIMIT_TECHNICAL = 1024

MAX_TEXT_LIMIT_CONSERVATIVE = 3900
MAX_TEXT_LIMIT_TECHNICAL = 4000
MAX_TOTAL_ATTACHMENTS_LIMIT = 12
MAX_ATTACHMENTS_WITH_KEYBOARD_LIMIT = 11
MAX_RATE_LIMIT_MESSAGES_PER_SEC = 2

# Allowed HTML tags for Telegram parse_mode='HTML' in event messages
ALLOWED_EVENT_HTML_TAGS = frozenset({
    "b", "strong", "i", "em", "s", "strike", "u", "code", "pre", "a", "blockquote", "tg-spoiler", "tg-emoji"
})

_CUSTOM_EMOJI_PATTERN = re.compile(
    r'<tg-emoji\b[^>]*\b(?:emoji[-_]id|id)=["\']?([^"\'>]+)["\']?[^>]*>(.*?)</tg-emoji>',
    re.IGNORECASE | re.DOTALL,
)

_ANY_HTML_TAG_PATTERN = re.compile(r'<[^>]+>')


def utf16_code_units_len(text: str) -> int:
    """Calculates string length in UTF-16 code units (as counted by Telegram Bot API entities)."""
    return len(text.encode("utf-16-le")) // 2


def get_media_format(media_mode: str | None, media_order: str | None) -> str:
    """Translates database columns (media_mode, media_order) to canonical media format."""
    mode = (media_mode or "").strip().lower()
    order = (media_order or "").strip().lower()
    if mode == MEDIA_FORMAT_LEGACY_AUTO or not mode:
        return MEDIA_FORMAT_LEGACY_AUTO
    if mode == "caption":
        return MEDIA_FORMAT_CAPTION
    if mode == "separate":
        if order == "text_top":
            return MEDIA_FORMAT_SEPARATE_TEXT_TOP
        return MEDIA_FORMAT_SEPARATE_MEDIA_TOP
    # Fallback to separate
    if order == "text_top":
        return MEDIA_FORMAT_SEPARATE_TEXT_TOP
    return MEDIA_FORMAT_SEPARATE_MEDIA_TOP


def resolve_media_mode_and_order(media_format: str) -> tuple[str, str]:
    """Translates canonical media format to database columns (media_mode, media_order)."""
    fmt = (media_format or "").strip().lower()
    if fmt == MEDIA_FORMAT_LEGACY_AUTO:
        return "legacy_auto", "media_top"
    if fmt == MEDIA_FORMAT_CAPTION:
        return "caption", "media_top"
    if fmt == MEDIA_FORMAT_SEPARATE_TEXT_TOP:
        return "separate", "text_top"
    if fmt == MEDIA_FORMAT_SEPARATE_MEDIA_TOP:
        return "separate", "media_top"
    raise ValueError(f"Недопустимый формат медиа: {media_format}. Допустимые: {', '.join(ALL_MEDIA_FORMATS)}")


def strip_telegram_custom_emoji(text: str) -> str:
    """Replaces <tg-emoji emoji-id="...">fallback</tg-emoji> with its inner fallback text."""
    if not text or "<tg-emoji" not in text:
        return text
    return _CUSTOM_EMOJI_PATTERN.sub(r"\2", text)


def normalize_telegram_custom_emoji_tags(text: str) -> str:
    """
    Normalizes all variants of custom emoji tags (<tg-emoji id="...">, <tg-emoji emoji_id="...">)
    to official Telegram Bot API format: <tg-emoji emoji-id="...">fallback</tg-emoji>.
    """
    if not text or "<tg-emoji" not in text:
        return text
    return _CUSTOM_EMOJI_PATTERN.sub(r'<tg-emoji emoji-id="\1">\2</tg-emoji>', text)


def format_event_html_title(title: str, platform: str = "telegram") -> str:
    """
    Renders title safely:
    - For Telegram: preserves and normalizes valid <tg-emoji> tags while escaping other unsafe HTML characters.
    - For MAX: replaces <tg-emoji> tags with Unicode fallback and strips HTML markup tags.
    """
    if not title:
        return ""
    if platform == "max":
        plain = strip_telegram_custom_emoji(title)
        return html.escape(plain, quote=False)
    if "<tg-emoji" not in title:
        return html.escape(title, quote=False)

    emoji_tokens: list[str] = []

    def _repl(match: re.Match) -> str:
        emoji_id = match.group(1)
        fallback = match.group(2)
        emoji_tokens.append(f'<tg-emoji emoji-id="{emoji_id}">{fallback}</tg-emoji>')
        return f"\x00EMOJI_{len(emoji_tokens) - 1}\x00"

    tokenized = _CUSTOM_EMOJI_PATTERN.sub(_repl, title)
    escaped = html.escape(tokenized, quote=False)
    for idx, token in enumerate(emoji_tokens):
        escaped = escaped.replace(f"\x00EMOJI_{idx}\x00", token)
    return escaped


def render_plain_text_for_max(text: str) -> str:
    """
    Renders text for MAX Bot API:
    - Strips Telegram custom emoji tags to their inner fallback Unicode character.
    - Strips all HTML markup tags.
    - Unescapes HTML entities (&quot;, &amp;, &lt;, &gt;, etc.).
    """
    if not text:
        return ""
    # 1. First convert tg-emoji to its Unicode fallback
    no_emoji_tags = strip_telegram_custom_emoji(text)
    # 2. Extract link URLs: <a href="url">text</a> -> text (url) or text
    def _link_repl(match: re.Match) -> str:
        tag = match.group(0)
        href_match = re.search(r'href=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        href = href_match.group(1) if href_match else ""
        return href

    # 3. Strip all HTML tags
    cleaned = _ANY_HTML_TAG_PATTERN.sub("", no_emoji_tags)
    # 4. Unescape HTML entities
    return html.unescape(cleaned)


class TelegramHtmlValidator(HTMLParser):
    """
    Strict HTML parser for Telegram entities:
    - Verifies all tags are in ALLOWED_EVENT_HTML_TAGS.
    - Verifies tags are balanced and properly nested.
    - Preserves and validates tg-emoji and a tags.
    - Accumulates parsed plain text for UTF-16 length calculation.
    """

    void_tags = frozenset({"br", "hr", "img", "input", "meta", "link"})

    def __init__(self, allowed_tags: frozenset[str] = ALLOWED_EVENT_HTML_TAGS):
        super().__init__(convert_charrefs=False)
        self.allowed_tags = allowed_tags
        self.stack: list[str] = []
        self.plain_tokens: list[str] = []
        self.errors: list[str] = []

    @classmethod
    def validate(cls, text: str, allowed_tags: frozenset[str] = ALLOWED_EVENT_HTML_TAGS) -> None:
        """Parses and validates text, raising ValueError if HTML tags are invalid or unbalanced."""
        if not text:
            return
        parser = cls(allowed_tags=allowed_tags)
        parser.feed(text)
        _, errors = parser.finish()
        if errors:
            raise ValueError("; ".join(errors))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower not in self.allowed_tags:
            self.errors.append(f"Недопустимый HTML-тег: <{tag}> (неподдерживается)")
            return
        if tag_lower == "a":
            has_href = any(name.lower() == "href" and value for name, value in attrs)
            if not has_href:
                self.errors.append("Тег <a> обязан содержать атрибут href")
        elif tag_lower == "tg-emoji":
            has_emoji_id = any(name.lower() in ("emoji-id", "id", "emoji_id") and value for name, value in attrs)
            if not has_emoji_id:
                self.errors.append("Тег <tg-emoji> обязан содержать атрибут emoji-id")
        if tag_lower not in self.void_tags:
            self.stack.append(tag_lower)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower not in self.allowed_tags:
            self.errors.append(f"Неподдерживаемый закрывающий HTML-тег: </{tag}>")
            return
        if tag_lower in self.void_tags:
            return
        if not self.stack:
            self.errors.append(f"Лишний закрывающий HTML-тег </{tag}> без открывающего")
            return
        expected = self.stack.pop()
        if expected != tag_lower:
            self.errors.append(f"Несбалансированный HTML-тег (нарушен порядок закрытия): ожидался </{expected}>, но встречен </{tag}>")

    def handle_data(self, data: str) -> None:
        if "<" in data:
            self.errors.append("Неэкранированный символ '<' в тексте (используйте &lt;)")
        self.plain_tokens.append(data)

    def handle_entityref(self, name: str) -> None:
        try:
            char = html.unescape(f"&{name};")
            self.plain_tokens.append(char)
        except Exception:
            self.plain_tokens.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        try:
            char = html.unescape(f"&#{name};")
            self.plain_tokens.append(char)
        except Exception:
            self.plain_tokens.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.errors.append("HTML-комментарии не поддерживаются в сообщениях")

    def finish(self) -> tuple[str, list[str]]:
        self.close()
        if self.stack:
            unclosed = ", ".join(f"<{t}>" for t in self.stack)
            self.errors.append(f"Несбалансированный HTML (не закрыты теги): {unclosed}")
        return "".join(self.plain_tokens), self.errors


def parse_visible_telegram_text(text: str) -> tuple[str, int, list[str]]:
    """
    Parses Telegram HTML message:
    Returns (visible_text, utf16_length, errors).
    """
    if not text:
        return "", 0, []
    parser = TelegramHtmlValidator()
    try:
        parser.feed(text)
        visible_text, errors = parser.finish()
    except Exception as exc:
        return text, utf16_code_units_len(text), [f"Ошибка разбора HTML: {exc}"]
    return visible_text, utf16_code_units_len(visible_text), errors


class PathValidationResult:
    def __init__(
        self,
        *,
        path_id: str,
        platform: str,
        message_kind: str,
        rendered_text: str,
        visible_text: str,
        length: int,
        limit: int,
        is_valid: bool,
        excess: int,
        error_message: str | None = None,
        recommendation: str | None = None,
    ):
        self.path_id = path_id
        self.platform = platform
        self.message_kind = message_kind
        self.rendered_text = rendered_text
        self.visible_text = visible_text
        self.length = length
        self.limit = limit
        self.is_valid = is_valid
        self.excess = excess
        self.error_message = error_message
        self.recommendation = recommendation

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "platform": self.platform,
            "message_kind": self.message_kind,
            "length": self.length,
            "limit": self.limit,
            "excess": self.excess,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "recommendation": self.recommendation,
        }


class ScenarioValidationResult:
    def __init__(
        self,
        *,
        is_valid: bool,
        media_format: str,
        media_count: int,
        violations: list[PathValidationResult],
        all_checks: list[PathValidationResult],
        available_description_chars: dict[str, int],
        warnings: list[str] | None = None,
    ):
        self.is_valid = is_valid
        self.media_format = media_format
        self.media_count = media_count
        self.violations = violations
        self.all_checks = all_checks
        self.available_description_chars = available_description_chars
        self.warnings = warnings or []

    @property
    def primary_error_message(self) -> str:
        if not self.violations:
            return ""
        first = self.violations[0]
        if first.error_message:
            return first.error_message
        fmt_label = "подпись к медиа" if self.media_format == MEDIA_FORMAT_CAPTION else "текст сообщения"
        rec = f" {first.recommendation}" if first.recommendation else ""
        return (
            f"Превышен лимит ({first.platform.upper()} / {fmt_label}): "
            f"фактическая длина {first.length} симв. при лимите {first.limit} "
            f"(превышение на {first.excess} симв.).{rec}"
        )


class EventContentValidationError(ValueError):
    """Raised on service/backend boundary when event content exceeds safe delivery limits."""

    def __init__(
        self,
        message: str,
        result: ScenarioValidationResult | None = None,
        *,
        platform: str | None = None,
        actual_length: int | None = None,
        limit_length: int | None = None,
        recommendation: str | None = None,
        excess: int | None = None,
    ):
        super().__init__(message)
        self.result = result
        self._message = message
        first = result.violations[0] if result and result.violations else None
        self._platform = platform or (first.platform if first else "all")
        self._actual_length = actual_length if actual_length is not None else (first.length if first else 0)
        self._limit_length = limit_length if limit_length is not None else (first.limit if first else 0)
        self._recommendation = recommendation if recommendation is not None else (first.recommendation if first and first.recommendation else "")
        self._excess = excess if excess is not None else (first.excess if first else max(0, self._actual_length - self._limit_length))

    @property
    def message(self) -> str:
        return self._message

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def actual_length(self) -> int:
        return self._actual_length

    @property
    def limit_length(self) -> int:
        return self._limit_length

    @property
    def recommendation(self) -> str:
        return self._recommendation

    @property
    def excess(self) -> int:
        return self._excess

    @property
    def primary_violation(self) -> PathValidationResult | None:
        return self.result.violations[0] if self.result and self.result.violations else None


# Stress test user probe with realistic maximum name length in Telegram (up to 64 chars)
_STRESS_PROBE_USER = SimpleNamespace(
    id=999999999,
    first_name="Александра-Виктория-Константинопольская-Преображенская",
    name="Александра-Виктория-Константинопольская-Преображенская",
)
_DEFAULT_PROBE_USER = SimpleNamespace(
    id=999999998,
    first_name="друг",
    name="друг",
)


def _render_user_path_messages(
    scenario: Any,
    steps: Sequence[Any],
    user: Any,
    now: Any = None,
) -> list[tuple[str, str, str, bool]]:
    """
    Renders all user journey messages:
    returns list of tuples: (path_id, message_kind, template, include_details).
    """
    from event_service import (
        DEFAULT_CONFIRMATION_TEMPLATE,
        DEFAULT_ADAPTIVE_CONFIRMATION_TEMPLATE,
        DEFAULT_LIVE_TEMPLATE,
        DEFAULT_RESCHEDULE_TEMPLATE,
    )

    preview_template = (
        "✅ Вы зарегистрированы на «{event_title}»!\n\n"
        "📅 Дата: {event_date} в {event_time}\n"
        "📍 Место / Ссылка: {link_or_place}"
    )

    paths: list[tuple[str, str, str, bool]] = [
        ("immediate_confirmation", "immediate_confirmation", DEFAULT_CONFIRMATION_TEMPLATE, True),
        ("immediate_adaptive_confirmation", "immediate_confirmation", DEFAULT_ADAPTIVE_CONFIRMATION_TEMPLATE, True),
        ("immediate_live", "immediate_live", DEFAULT_LIVE_TEMPLATE, True),
        ("reschedule_notice", "reschedule_notice", DEFAULT_RESCHEDULE_TEMPLATE, True),
        ("preview_confirmation", "preview_confirmation", preview_template, True),
    ]

    for step in steps:
        if getattr(step, "is_archived", False):
            continue
        step_id = getattr(step, "id", None) or "draft"
        inc_details = bool(getattr(step, "include_event_details", False))
        paths.append((f"step_{step_id}", "scheduled_step", getattr(step, "message_text", "") or "", inc_details))

    return paths


def validate_scenario_content(
    scenario: Any,
    *,
    media_format: str | None = None,
    media_count: int | None = None,
    media: Any = None,
    steps: Sequence[Any] = (),
    is_existing_legacy: bool = False,
    now: Any = None,
) -> ScenarioValidationResult:
    """
    Validates complete rendered output of an event scenario for Telegram and MAX:
    - Checks Mode C caption compatibility (exactly 1 photo/video).
    - Checks all message paths with stress user probes.
    - Evaluates HTML markup, visible UTF-16 length, and MAX plain text length.
    - Calculates available remaining description characters.
    """
    from event_service import render_event_message

    if media is not None:
        actual_media_count = len(media) if hasattr(media, "__len__") else sum(1 for _ in media)
    elif media_count is not None:
        actual_media_count = media_count
    else:
        scen_media = getattr(scenario, "media", None)
        if scen_media is not None:
            actual_media_count = len(scen_media) if hasattr(scen_media, "__len__") else sum(1 for _ in scen_media)
        else:
            actual_media_count = 0

    effective_format = media_format or get_media_format(
        getattr(scenario, "media_mode", None),
        getattr(scenario, "media_order", None),
    )

    violations: list[PathValidationResult] = []
    all_checks: list[PathValidationResult] = []
    warnings: list[str] = []

    # 1. Mode C media count validation
    if effective_format == MEDIA_FORMAT_CAPTION:
        if actual_media_count != 1:
            err_msg = (
                f"Режим «подпись к медиа» требует ровно 1 медиа-вложение (сейчас: {actual_media_count}). "
                "Прикрепите одно фото/видео или переключите формат доставки на раздельный."
            )
            rec = "Выберите «Текст отдельно, медиа выше/ниже» или оставьте ровно 1 медиа."
            v = PathValidationResult(
                path_id="media_mode_caption_count",
                platform="all",
                message_kind="media_validation",
                rendered_text="",
                visible_text="",
                length=actual_media_count,
                limit=1,
                is_valid=False,
                excess=max(0, actual_media_count - 1),
                error_message=err_msg,
                recommendation=rec,
            )
            violations.append(v)
            all_checks.append(v)

    # 2. Check HTML markup of raw editable scenario fields
    raw_desc = getattr(scenario, "description", None)
    if raw_desc:
        _, _, desc_html_errors = parse_visible_telegram_text(raw_desc)
        if desc_html_errors:
            err_msg = f"Ошибка HTML-разметки в описании: {'; '.join(desc_html_errors)}"
            v = PathValidationResult(
                path_id="scenario_description_html",
                platform="telegram",
                message_kind="raw_field",
                rendered_text=raw_desc,
                visible_text="",
                length=len(raw_desc),
                limit=TELEGRAM_TEXT_LIMIT_CONSERVATIVE,
                is_valid=False,
                excess=0,
                error_message=err_msg,
                recommendation="Проверьте корректность и баланс HTML-тегов (b, i, u, code, a, tg-emoji).",
            )
            violations.append(v)
            all_checks.append(v)

    raw_title = getattr(scenario, "title", None)
    if raw_title and "<tg-emoji" in raw_title:
        _, _, title_html_errors = parse_visible_telegram_text(raw_title)
        if title_html_errors:
            err_msg = f"Ошибка HTML-разметки в названии: {'; '.join(title_html_errors)}"
            v = PathValidationResult(
                path_id="scenario_title_html",
                platform="telegram",
                message_kind="raw_field",
                rendered_text=raw_title,
                visible_text="",
                length=len(raw_title),
                limit=1000,
                is_valid=False,
                excess=0,
                error_message=err_msg,
                recommendation="Проверьте корректность и баланс HTML-тегов в названии.",
            )
            violations.append(v)
            all_checks.append(v)

    for step in steps:
        step_text = getattr(step, "message_text", None)
        if step_text:
            _, _, step_html_errors = parse_visible_telegram_text(step_text)
            if step_html_errors:
                step_id = getattr(step, "id", None) or "draft"
                err_msg = f"Ошибка HTML-разметки в шаге напоминания {step_id}: {'; '.join(step_html_errors)}"
                v = PathValidationResult(
                    path_id=f"step_{step_id}_html",
                    platform="telegram",
                    message_kind="step_raw_field",
                    rendered_text=step_text,
                    visible_text="",
                    length=len(step_text),
                    limit=TELEGRAM_TEXT_LIMIT_CONSERVATIVE,
                    is_valid=False,
                    excess=0,
                    error_message=err_msg,
                    recommendation="Проверьте корректность и баланс HTML-тегов в тексте шага.",
                )
                violations.append(v)
                all_checks.append(v)

    # 3. Check limits for each message path
    probes = [_STRESS_PROBE_USER, _DEFAULT_PROBE_USER]
    path_definitions = _render_user_path_messages(scenario, steps, _STRESS_PROBE_USER, now=now)

    for path_id, msg_kind, template, inc_details in path_definitions:
        for probe_user in probes:
            # 2a. Telegram render
            tg_rendered = render_event_message(
                template=template,
                scenario=scenario,
                user=probe_user,
                platform="telegram",
                include_event_details=inc_details,
                now=now,
            )
            tg_vis_text, tg_utf16_len, tg_html_errors = parse_visible_telegram_text(tg_rendered)

            is_caption = (effective_format == MEDIA_FORMAT_CAPTION)
            tg_limit = TELEGRAM_CAPTION_LIMIT_CONSERVATIVE if is_caption else TELEGRAM_TEXT_LIMIT_CONSERVATIVE
            tg_tech_limit = TELEGRAM_CAPTION_LIMIT_TECHNICAL if is_caption else TELEGRAM_TEXT_LIMIT_TECHNICAL

            tg_err: str | None = None
            tg_rec: str | None = None
            tg_valid = True

            if tg_html_errors:
                tg_valid = False
                tg_err = f"Ошибка HTML в сообщении ({path_id}): {'; '.join(tg_html_errors)}"
                tg_rec = "Проверьте закрытие всех тегов (b, i, u, code, a, tg-emoji)."
            elif effective_format != MEDIA_FORMAT_LEGACY_AUTO and tg_utf16_len > tg_limit:
                tg_valid = False
                excess = tg_utf16_len - tg_limit
                fmt_desc = "подпись" if is_caption else "текст"
                tg_err = (
                    f"Сообщение '{path_id}' превышает лимит Telegram для формата '{fmt_desc}': "
                    f"{tg_utf16_len} > {tg_limit} (превышение: {excess} симв.)"
                )
                tg_rec = (
                    "Переключите формат доставки на раздельный или сократите текст описания/шага."
                    if is_caption
                    else "Сократите текст описания или текст шага напоминания."
                )

            tg_res = PathValidationResult(
                path_id=f"{path_id}:telegram:{getattr(probe_user, 'first_name', '')[:4]}",
                platform="telegram",
                message_kind=msg_kind,
                rendered_text=tg_rendered,
                visible_text=tg_vis_text,
                length=tg_utf16_len,
                limit=tg_limit,
                is_valid=tg_valid,
                excess=max(0, tg_utf16_len - tg_limit),
                error_message=tg_err,
                recommendation=tg_rec,
            )
            all_checks.append(tg_res)
            if not tg_valid and not any(v.path_id == tg_res.path_id for v in violations):
                violations.append(tg_res)

            # 2b. MAX render
            max_rendered = render_event_message(
                template=template,
                scenario=scenario,
                user=probe_user,
                platform="max",
                include_event_details=inc_details,
                now=now,
            )
            max_to_send = strip_telegram_custom_emoji(max_rendered)
            max_plain = render_plain_text_for_max(max_rendered)
            max_len = len(max_to_send)
            max_limit = MAX_TEXT_LIMIT_CONSERVATIVE

            max_valid = True
            max_err: str | None = None
            max_rec: str | None = None

            if effective_format != MEDIA_FORMAT_LEGACY_AUTO and max_len > max_limit:
                max_valid = False
                excess = max_len - max_limit
                max_err = (
                    f"Сообщение '{path_id}' превышает рабочий лимит MAX ({max_limit} симв.): "
                    f"фактическая длина с разметкой {max_len} (превышение: {excess} симв.)"
                )
                max_rec = "Сократите текст описания или уменьшите HTML-разметку."
            elif max_len > MAX_TEXT_LIMIT_TECHNICAL and effective_format != MEDIA_FORMAT_LEGACY_AUTO:
                max_valid = False
                excess = max_len - MAX_TEXT_LIMIT_TECHNICAL
                max_err = (
                    f"Сообщение '{path_id}' превышает технический лимит MAX ({MAX_TEXT_LIMIT_TECHNICAL} симв.): "
                    f"фактическая длина с разметкой {max_len} (превышение: {excess} симв.)"
                )
                max_rec = "Сократите текст описания или уменьшите HTML-разметку."

            # MAX attachments count check
            # For caption: 1 media attachment + 1 keyboard attachment = 2 <= 12
            # For separate: media parts are up to 10-11 items <= 12
            if is_caption and actual_media_count + 1 > MAX_TOTAL_ATTACHMENTS_LIMIT:
                max_valid = False
                max_err = f"Превышено максимальное число вложений в MAX ({actual_media_count + 1} > {MAX_TOTAL_ATTACHMENTS_LIMIT})"
                max_rec = "Уменьшите количество прикреплённых медиа."

            max_res = PathValidationResult(
                path_id=f"{path_id}:max:{getattr(probe_user, 'first_name', '')[:4]}",
                platform="max",
                message_kind=msg_kind,
                rendered_text=max_rendered,
                visible_text=max_plain,
                length=max_len,
                limit=max_limit,
                is_valid=max_valid,
                excess=max(0, max_len - max_limit),
                error_message=max_err,
                recommendation=max_rec,
            )
            all_checks.append(max_res)
            if not max_valid and not any(v.path_id == max_res.path_id for v in violations):
                violations.append(max_res)

    # 3. Available description characters calculation
    available_chars = calculate_available_description_chars(
        scenario,
        media_format=effective_format,
        steps=steps,
        now=now,
    )

    is_overall_valid = len(violations) == 0

    return ScenarioValidationResult(
        is_valid=is_overall_valid,
        media_format=effective_format,
        media_count=media_count,
        violations=violations,
        all_checks=all_checks,
        available_description_chars=available_chars,
        warnings=warnings,
    )


def calculate_available_description_chars(
    scenario: Any = None,
    *,
    media_format: str = MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
    steps: Sequence[Any] = (),
    now: Any = None,
    title: str | None = None,
    link: str | None = None,
    place: str | None = None,
    duration_minutes: int | None = None,
) -> dict[str, int]:
    """
    Computes maximum characters available for event description so that
    no rendered message path exceeds conservative limits.
    """
    from event_service import render_event_message

    # Create probe scenario with empty description to measure overhead
    probe_scenario = SimpleNamespace(
        id=getattr(scenario, "id", 1) if scenario is not None else 1,
        code=getattr(scenario, "code", "code") if scenario is not None else "code",
        event_type=getattr(scenario, "event_type", "webinar") if scenario is not None else "webinar",
        title=title if title is not None else (getattr(scenario, "title", "") if scenario is not None else ""),
        starts_at=getattr(scenario, "starts_at", None) if scenario is not None else None,
        duration_minutes=duration_minutes if duration_minutes is not None else (getattr(scenario, "duration_minutes", 120) if scenario is not None else 120),
        place=place if place is not None else (getattr(scenario, "place", None) if scenario is not None else None),
        link=link if link is not None else (getattr(scenario, "link", None) if scenario is not None else None),
        description="",
        media_mode="caption" if media_format == MEDIA_FORMAT_CAPTION else "separate",
        media_order="text_top" if media_format == MEDIA_FORMAT_SEPARATE_TEXT_TOP else "media_top",
    )

    is_caption = (media_format == MEDIA_FORMAT_CAPTION)
    tg_limit = TELEGRAM_CAPTION_LIMIT_CONSERVATIVE if is_caption else TELEGRAM_TEXT_LIMIT_CONSERVATIVE
    max_limit = MAX_TEXT_LIMIT_CONSERVATIVE

    paths = _render_user_path_messages(probe_scenario, steps, _STRESS_PROBE_USER, now=now)

    min_tg_remaining = tg_limit
    min_max_remaining = max_limit

    # Overhead marker for description block in render_event_details:
    # "\n\n📝 Описание:\n" -> ~16 chars
    desc_wrapper_overhead = 16

    for path_id, _, template, inc_details in paths:
        if not inc_details and "{description}" not in template and "{event_description}" not in template:
            # Step without event details does not consume description budget
            continue

        tg_base = render_event_message(
            template=template,
            scenario=probe_scenario,
            user=_STRESS_PROBE_USER,
            platform="telegram",
            include_event_details=inc_details,
            now=now,
        )
        _, tg_base_len, _ = parse_visible_telegram_text(tg_base)
        tg_rem = tg_limit - (tg_base_len + desc_wrapper_overhead)
        if tg_rem < min_tg_remaining:
            min_tg_remaining = tg_rem

        max_base = render_event_message(
            template=template,
            scenario=probe_scenario,
            user=_STRESS_PROBE_USER,
            platform="max",
            include_event_details=inc_details,
            now=now,
        )
        max_to_send = strip_telegram_custom_emoji(max_base)
        max_rem = max_limit - (len(max_to_send) + desc_wrapper_overhead)
        if max_rem < min_max_remaining:
            min_max_remaining = max_rem

    overall = min(max(0, min_tg_remaining), max(0, min_max_remaining))

    return {
        "telegram": max(0, min_tg_remaining),
        "max": max(0, min_max_remaining),
        "overall": overall,
        "conservative": overall,
    }
