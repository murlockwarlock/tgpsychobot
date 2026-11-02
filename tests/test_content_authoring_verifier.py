from scripts import verify_content_authoring_production as verifier


def test_language_configuration_accepts_true_and_false_selector_states():
    for selector in (True, False):
        config = verifier.normalize_language_configuration(
            ("ru", '["ru", "en", "pt"]', selector, 7)
        )

        assert config["default"] == "ru"
        assert config["enabled"] == ["en", "pt", "ru"]
        assert config["selector"] is selector


def test_language_configuration_detects_unexpected_selector_change():
    baseline = {
        "default": "ru",
        "enabled": ["en", "pt", "ru"],
        "selector": False,
    }
    current = {
        "language_config": {
            "default": "ru",
            "enabled": ["en", "pt", "ru"],
            "selector": True,
        },
        "default": "ru",
        "enabled": ["en", "pt", "ru"],
        "selector": True,
    }

    assert verifier.language_configuration_changes(current, baseline) == ["selector"]
