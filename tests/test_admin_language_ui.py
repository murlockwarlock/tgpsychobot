import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from types import SimpleNamespace

from handlers import _language_locale_text, _language_readiness_summary
import keyboards


def _config(**overrides):
    values = {
        "telegram_default_language": "ru",
        "telegram_language_selection_enabled": False,
        "telegram_enabled_languages": '["ru"]',
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _readiness(en_ready=False, pt_ready=False):
    return {
        "ready": en_ready and pt_ready,
        "locales": {
            "en": {
                "ready": en_ready,
                "translated": 0 if not en_ready else 2,
                "required": 2,
                "missing": [] if en_ready else ["ui.one", "ui.format"],
                "stale": [],
                "invalid": [],
            },
            "pt": {
                "ready": pt_ready,
                "translated": 0 if not pt_ready else 2,
                "required": 2,
                "missing": [] if pt_ready else ["ui.one", "ui.format"],
                "stale": [],
                "invalid": [],
            },
        },
        "orphaned": [],
    }


def _buttons(markup):
    return [button for row in markup.inline_keyboard for button in row]


def test_main_readiness_summary_uses_locale_report():
    readiness = _readiness()
    assert "не готов" in _language_readiness_summary(readiness, "en")
    assert "пропущено 2" in _language_readiness_summary(readiness, "en")
    assert "не готов" in _language_locale_text(_config(), "en", readiness)


def test_main_language_screen_has_compact_locale_management_buttons():
    buttons = _buttons(keyboards.admin_language_settings_keyboard(_config(), _readiness()))
    texts = [button.text for button in buttons]
    callbacks = [button.callback_data for button in buttons]

    assert "⚙️ 🇷🇺 Русский" in texts
    assert "⚙️ 🇬🇧 English" in texts
    assert "⚙️ 🇵🇹 Português" in texts
    assert "🔴 Включить выбор языка" in texts
    assert "🔎 Общий аудит" in texts
    assert "Сделать 🇷🇺 Русский языком по умолчанию" not in texts
    assert "admin_language_locale_en" in callbacks
    assert "admin_language_locale_pt" in callbacks

    selector_buttons = _buttons(
        keyboards.admin_language_settings_keyboard(
            _config(telegram_language_selection_enabled=True),
            _readiness(),
        )
    )
    assert "🟢 Выключить выбор языка" in [button.text for button in selector_buttons]


def test_locale_detail_owns_locale_actions_and_hides_current_default_action():
    detail = getattr(keyboards, "admin_language_locale_keyboard")
    buttons = _buttons(
        detail(
            _config(),
            "en",
            _readiness(en_ready=True)["locales"]["en"],
        )
    )
    texts = [button.text for button in buttons]
    callbacks = [button.callback_data for button in buttons]

    assert "📤 Экспорт EN" in texts
    assert "📥 Импорт EN" in texts
    assert "🔎 Проверить EN" in texts
    assert "✅ Включить EN" in texts
    assert "⭐ По умолчанию" not in texts
    assert "admin_translation_export_en" in callbacks
    assert "admin_translation_import_en" in callbacks

    enabled_buttons = _buttons(
        detail(
            _config(telegram_enabled_languages='["ru", "en"]'),
            "en",
            _readiness(en_ready=True)["locales"]["en"],
        )
    )
    enabled_texts = [button.text for button in enabled_buttons]
    assert "🚫 Выключить EN" in enabled_texts
    assert "✅ Включить EN" not in enabled_texts
    assert "⭐ По умолчанию" in enabled_texts


def test_not_ready_locale_detail_does_not_offer_enable():
    detail = getattr(keyboards, "admin_language_locale_keyboard")
    buttons = _buttons(
        detail(
            _config(),
            "pt",
            _readiness()["locales"]["pt"],
        )
    )
    texts = [button.text for button in buttons]
    assert "✅ Включить PT" not in texts
    assert "⭐ По умолчанию" not in texts
