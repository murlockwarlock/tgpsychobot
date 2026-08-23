import ast
from pathlib import Path

from sqlalchemy import create_engine, text

from provider_models import (
    DEEPSEEK_DEFAULT_MODEL,
    DEEPSEEK_MODELS,
    normalize_deepseek_model,
)


def test_current_deepseek_models_are_used_in_telegram_and_max_admins():
    """Both Telegram and MAX now call get_selectable_models('Deepseek', channel=...)
    instead of reading a hardcoded PROVIDER_MODELS dict.  Verify the SSOT catalog
    returns the same list as DEEPSEEK_MODELS for both chat and fallback channels."""
    from provider_models import get_selectable_models

    expected = list(DEEPSEEK_MODELS)
    # Telegram MODELS_INFO still lists the models for primary picker
    module = ast.parse(Path("handlers.py").read_text(encoding="utf-8"))
    models_assignment = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODELS_INFO" for target in node.targets)
    )
    models_info = ast.literal_eval(models_assignment.value)
    telegram_models = [name for name in models_info["Deepseek"] if name != "pricing"]

    assert telegram_models == expected
    # Both Telegram and MAX now delegate model lists to get_selectable_models()
    assert list(get_selectable_models("Deepseek", channel="chat")) == expected
    assert list(get_selectable_models("Deepseek", channel="fallback")) == expected


def test_legacy_deepseek_names_are_normalized_before_request():
    assert normalize_deepseek_model(None) == DEEPSEEK_DEFAULT_MODEL
    assert normalize_deepseek_model("deepseek-chat") == DEEPSEEK_DEFAULT_MODEL
    assert normalize_deepseek_model("deepseek-reasoner") == DEEPSEEK_DEFAULT_MODEL
    assert normalize_deepseek_model("deepseek-coder") == DEEPSEEK_DEFAULT_MODEL
    assert normalize_deepseek_model("deepseek-v4-pro") == "deepseek-v4-pro"


def test_database_migration_replaces_saved_legacy_models():
    from database import _migrate_deepseek_models

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE ai_config (
                id INTEGER PRIMARY KEY,
                deepseek_model VARCHAR,
                fallback_provider VARCHAR,
                fallback_model VARCHAR
            )
        """))
        conn.execute(text("""
            INSERT INTO ai_config (id, deepseek_model, fallback_provider, fallback_model)
            VALUES
                (1, 'deepseek-chat', 'Deepseek', 'deepseek-reasoner'),
                (2, 'deepseek-v4-pro', 'KIE', 'gemini-3-flash')
        """))

        _migrate_deepseek_models(conn)
        rows = conn.execute(text("""
            SELECT id, deepseek_model, fallback_provider, fallback_model
            FROM ai_config ORDER BY id
        """)).all()

    assert rows == [
        (1, "deepseek-v4-flash", "Deepseek", "deepseek-v4-flash"),
        (2, "deepseek-v4-pro", "KIE", "gemini-3-flash"),
    ]
