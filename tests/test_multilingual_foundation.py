from collections import Counter
import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest

from database import BotGeneralConfig, BotTranslation, User
from keyboards import language_selection_keyboard
from translation_service import (
    LOCALE_LABELS,
    SUPPORTED_TELEGRAM_LOCALES,
    TranslationValidationError,
    format_signature,
    normalize_enabled_languages,
    resolve_effective_locale,
    source_hash,
    TranslationSnapshot,
    validate_translation_format,
)


def test_supported_locale_contract_is_exact():
    assert SUPPORTED_TELEGRAM_LOCALES == ("ru", "en", "pt")
    assert LOCALE_LABELS == {
        "ru": "🇷🇺 Русский",
        "en": "🇬🇧 English",
        "pt": "🇵🇹 Português",
    }


def test_enabled_languages_are_safe_deterministic_and_keep_russian():
    assert normalize_enabled_languages('["pt", "pt", "xx", "en"]') == ("ru", "en", "pt")
    assert normalize_enabled_languages("not json") == ("ru",)
    assert normalize_enabled_languages("[]") == ("ru",)


def test_effective_locale_respects_selector_and_platform_rules():
    assert resolve_effective_locale(
        requested="en",
        default="ru",
        selector_enabled=False,
        enabled=("ru", "en", "pt"),
    ) == "ru"
    assert resolve_effective_locale(
        requested="en",
        default="ru",
        selector_enabled=True,
        enabled=("ru", "en", "pt"),
    ) == "en"
    assert resolve_effective_locale(
        requested=None,
        default="pt",
        selector_enabled=True,
        enabled=("ru", "en", "pt"),
    ) == "pt"
    assert resolve_effective_locale(
        requested="en",
        default="en",
        selector_enabled=True,
        enabled=("ru", "en", "pt"),
        platform="max",
    ) == "ru"


def test_source_hash_normalizes_newlines_without_trimming():
    assert source_hash("a\r\nb") == source_hash("a\nb")
    assert source_hash("a\rb") == source_hash("a\nb")
    assert source_hash(" a\n") != source_hash("a\n")


def test_format_signature_is_a_multiset_and_allows_reordering():
    source = "{name} {name:>10} {value!r}"
    translated = "{value!r} {name:>10} {name}"
    assert format_signature(source) == Counter(
        {
            ("name", None, ""): 1,
            ("name", None, ">10"): 1,
            ("value", "r", ""): 1,
        }
    )
    validate_translation_format(source, translated)


def test_format_signature_rejects_duplicate_placeholder_changes():
    with pytest.raises(TranslationValidationError):
        validate_translation_format("{name} {name}", "{name}")

    with pytest.raises(TranslationValidationError):
        validate_translation_format("{name}", "{name} {name}")


def test_format_validation_allows_reordering_and_escaped_braces():
    validate_translation_format("{name} {date}", "{date} {name}")
    validate_translation_format("{{name}} {value}", "{{nombre}} {value}")


def test_format_validation_rejects_format_spec_and_conversion_changes():
    with pytest.raises(TranslationValidationError):
        validate_translation_format("{amount:.2f}", "{amount}")
    with pytest.raises(TranslationValidationError):
        validate_translation_format("{name!r}", "{name!s}")


def test_translation_schema_contains_locale_and_user_preference_fields():
    assert hasattr(User, "telegram_language_code")
    assert hasattr(BotGeneralConfig, "telegram_default_language")
    assert hasattr(BotGeneralConfig, "telegram_language_selection_enabled")
    assert hasattr(BotGeneralConfig, "telegram_enabled_languages")
    assert hasattr(BotGeneralConfig, "translations_revision")
    assert BotTranslation.__tablename__ == "bot_translations"


def test_translation_snapshot_fallback_never_uses_en_for_russian():
    snapshot = TranslationSnapshot(
        revision=3,
        sources={"ui.greeting": "Привет"},
        translations={
            ("en", "ui.greeting"): "Hello",
            ("pt", "ui.greeting"): "Olá",
        },
    )

    assert snapshot.get("ui.greeting", "ru") == "Привет"
    assert snapshot.get("ui.greeting", "en") == "Hello"
    assert snapshot.get("ui.greeting", "fr") == "Привет"
    assert snapshot.get("missing", "en") is None


def test_empty_translation_does_not_hide_russian_source():
    snapshot = TranslationSnapshot(
        revision=4,
        sources={"ui.greeting": "Привет"},
        translations={("en", "ui.greeting"): ""},
    )

    assert snapshot.get("ui.greeting", "en") == "Привет"


def test_language_selector_uses_exact_locale_labels_and_stable_callbacks():
    markup = language_selection_keyboard(("ru", "en", "pt"))
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert [button.text for button in buttons] == [
        "🇷🇺 Русский",
        "🇬🇧 English",
        "🇵🇹 Português",
    ]
    assert [button.callback_data for button in buttons] == [
        "select_telegram_language:ru",
        "select_telegram_language:en",
        "select_telegram_language:pt",
    ]
