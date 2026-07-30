from pathlib import Path

from prompt_blocks import build_test_context_injection, render_prompt_block


def test_finished_test_context_is_rendered_inside_active_topic_prompt():
    context = build_test_context_injection(
        "Самооценка: 7 из 10",
        None,
        secret_test_enabled=False,
    )
    prompt = render_prompt_block(
        "Служебный блок\n{test_context_injection}",
        test_context_injection=context,
    )

    assert "[КОНТЕКСТ ТЕСТА]" in prompt
    assert "Самооценка: 7 из 10" in prompt
    assert "Не предлагай секретный блок" in prompt


def test_secret_test_answers_are_included_with_finished_status():
    context = build_test_context_injection("Основной результат", "Секретный ответ")

    assert "Основной результат" in context
    assert "Секретный ответ" in context
    assert "УЖЕ прошел все тесты" in context


def test_empty_test_context_stays_empty():
    assert build_test_context_injection(None, None) == ""


def test_finished_test_context_is_not_limited_to_general_dialogue():
    source = Path("ai_integration.py").read_text(encoding="utf-8")

    assert "include_test_context and (test_results_txt or secret_answers_txt)" in source
    assert "active_topic_id is None and (test_results_txt" not in source
