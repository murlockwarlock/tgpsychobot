from __future__ import annotations

from dataclasses import dataclass


CONTENT_PREFIXES = (
    "topic.", "content.", "plan.", "test_question.", "secret_test_question.",
    "mailing.", "automation_action.", "followup_step.", "referral_template.",
    "subscription_config.", "bot_general_config.", "media_library.", "case_study.",
)


def is_admin_content_key(key: str) -> bool:
    return isinstance(key, str) and key.startswith(CONTENT_PREFIXES)


@dataclass(frozen=True)
class ContentValue:
    text: str | None
    russian: str | None
    needs_review: bool = False

    @property
    def missing(self) -> bool:
        return not self.text

    def admin_label(self) -> str:
        if self.missing:
            return "⚠️ Перевод не задан" + (f" — {self.russian}" if self.russian else "")
        return self.text


def admin_label(kind, resource, field):
    from admin_authoring_context import content_editing_locale
    from translation_service import translation_cache
    locale = content_editing_locale.get() or "ru"
    source = getattr(resource, field, None) or ""
    if locale == "ru":
        return source or "⚠️ Перевод не задан"
    identity = resource.key if kind == "content" else resource.id
    value = translation_cache.snapshot.translations.get((locale, f"{kind}.{identity}.{field}"))
    return ContentValue(value, source).admin_label()
