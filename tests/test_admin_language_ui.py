import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from types import SimpleNamespace

from handlers import (
    _language_locale_text,
    _language_overview_text,
    _language_readiness_summary,
    _language_readiness_text,
    _translation_pack_filename,
    _translation_pack_target,
)
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
    assert "отсутствует 2" in _language_readiness_summary(readiness, "en")
    assert "не готов" in _language_locale_text(_config(), "en", readiness)


def test_main_language_screen_has_compact_locale_management_buttons():
    buttons = _buttons(
        keyboards.admin_language_settings_keyboard(
            _config(multilingual_authoring_enabled=True),
            _readiness(),
        )
    )
    texts = [button.text for button in buttons]
    callbacks = [button.callback_data for button in buttons]

    assert "⚙️ 🇷🇺 Русский" in texts
    assert "⚙️ 🇬🇧 English" in texts
    assert "⚙️ 🇵🇹 Português" in texts
    assert "✅ Разрешить пользователям выбирать язык" in texts
    assert "🔎 Проверка готовности переводов" in texts
    assert "↩️ Вернуть только русский" in texts
    assert "Сделать 🇷🇺 Русский языком по умолчанию" not in texts
    assert "admin_language_locale_en" in callbacks
    assert "admin_language_locale_pt" in callbacks

    selector_buttons = _buttons(
        keyboards.admin_language_settings_keyboard(
            _config(multilingual_authoring_enabled=True, telegram_language_selection_enabled=True),
            _readiness(),
        )
    )
    assert "⛔ Запретить пользователям выбирать язык" in [button.text for button in selector_buttons]


def test_multilingual_off_keeps_configured_locales_but_locks_user_selector():
    buttons = _buttons(
        keyboards.admin_language_settings_keyboard(
            _config(telegram_enabled_languages='["ru", "en", "pt"]'),
            _readiness(),
        )
    )
    texts = [button.text for button in buttons]
    callbacks = [button.callback_data for button in buttons]
    assert "🌐 Мультиязычность: ВЫКЛ" in texts
    assert "🔒 Выбор языка неактивен до включения мультиязычности" in texts
    assert "admin_language_toggle_selector" not in callbacks
    assert "⚙️ 🇬🇧 English" in texts
    assert "⚙️ 🇵🇹 Português" in texts


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
    assert "⭐ По умолчанию" not in enabled_texts


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


def test_ru_detail_is_canonical_and_has_no_pack_actions():
    buttons = _buttons(
        keyboards.admin_language_locale_keyboard(
            _config(telegram_enabled_languages='["ru", "en", "pt"]'),
            "ru",
            {"canonical": True, "ready": True},
        )
    )

    assert [button.text for button in buttons] == ["⬅️ Назад"]
    assert all("translation_" not in (button.callback_data or "") for button in buttons)


def test_import_preview_and_ru_only_confirmation_buttons_are_explicit():
    import_buttons = _buttons(keyboards.admin_translation_import_confirmation_keyboard("pt"))
    assert [button.text for button in import_buttons] == [
        "✅ Подтвердить импорт PT",
        "❌ Отмена",
    ]
    assert import_buttons[0].callback_data == "admin_translation_import_confirm"
    assert import_buttons[1].callback_data == "admin_translation_import_cancel"

    reset_buttons = _buttons(keyboards.admin_language_ru_only_confirmation_keyboard())
    assert [button.text for button in reset_buttons] == [
        "✅ Вернуть только русский",
        "❌ Отмена",
    ]
    assert reset_buttons[0].callback_data == "admin_language_ru_only_confirm"
    assert reset_buttons[1].callback_data == "admin_language_ru_only_cancel"


def test_overview_shows_bot_default_enabled_and_selector_semantics_in_russian():
    config = _config(
        multilingual_authoring_enabled=True,
        telegram_language_selection_enabled=True,
        telegram_enabled_languages='["ru", "en", "pt"]',
    )
    text = _language_overview_text(
        config,
        _readiness(en_ready=True, pt_ready=False),
        "🤖 Бот: <b>Example Bot</b> (@example_bot)",
    )

    assert "Язык по умолчанию: <b>🇷🇺 Русский</b>" in text
    assert "Выбор языка: <b>Включён</b>" in text
    assert "Мультиязычность: <b>Включена</b>" in text
    assert "Если мультиязычность выключена" not in text
    assert "Сохранённые предпочтения при этом не удаляются." not in text
    assert "✅ 🇬🇧 English" in text
    assert "✅ 🇵🇹 Português" in text
    assert "🇬🇧 English — ✅ готов" in text
    assert "🇵🇹 Português — ❌ не готов" in text
    assert "EN/PT" not in text


def test_readiness_screen_shows_counts_and_stale_recovery_steps():
    readiness = _readiness()
    readiness["locales"]["en"].update(
        translated=8,
        required=10,
        missing=["ui.one"],
        stale=["ui.format"],
        invalid=["ui.other"],
    )
    text = _language_readiness_text(readiness, "🤖 Бот: <b>Example Bot</b> (@example_bot)")

    assert "Проверка готовности переводов" in text
    assert "Обязательных: 8 / 10" in text
    assert "Отсутствует: 1" in text
    assert "Устарело: 1" in text
    assert "Ошибок: 1" in text
    assert "Перевод устарел после изменения русского текста." in text
    assert "экспортируйте актуальный шаблон" in text

    buttons = _buttons(keyboards.admin_translation_readiness_keyboard())
    callbacks = [button.callback_data for button in buttons]
    assert "admin_translation_export_en" in callbacks
    assert "admin_translation_import_en" in callbacks
    assert "admin_translation_export_pt" in callbacks
    assert "admin_translation_import_pt" in callbacks


def test_pack_filename_and_identity_bind_bot_and_database_without_secrets():
    bot = SimpleNamespace(id=12345, username="Example_Bot", first_name="Example Bot")
    target = _translation_pack_target(bot, "someone01")

    assert target == {
        "telegram_bot_id": 12345,
        "telegram_username": "Example_Bot",
        "database": "someone01",
    }
    assert _translation_pack_filename(target, "en") == (
        "telegram_translations_someone01_example_bot_en.json"
    )
    assert "token" not in target
