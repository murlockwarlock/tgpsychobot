from types import SimpleNamespace

import memory_mode
from memory_mode import (
    MEMORY_MODE_GLOBAL,
    MEMORY_MODE_RESET,
    MEMORY_MODE_TOPIC,
)
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


def test_finished_test_context_matches_active_topic_and_dialogue_scope():
    session = SimpleNamespace(
        is_finished=True,
        invocation_dialogue_id=2,
        invocation_topic_id=10,
    )

    # In TOPIC or RESET mode, matching dialogue & topic is valid
    assert memory_mode.test_session_matches_active_scope(session, active_dialogue_id=2, active_topic_id=10, memory_mode=MEMORY_MODE_TOPIC) is True
    assert memory_mode.test_session_matches_active_scope(session, active_dialogue_id=2, active_topic_id=10, memory_mode=MEMORY_MODE_RESET) is True

    # Different topic in TOPIC/RESET mode does not match
    assert memory_mode.test_session_matches_active_scope(session, active_dialogue_id=2, active_topic_id=20, memory_mode=MEMORY_MODE_TOPIC) is False
    assert memory_mode.test_session_matches_active_scope(session, active_dialogue_id=2, active_topic_id=None, memory_mode=MEMORY_MODE_RESET) is False

    # Different dialogue never matches in any mode
    assert memory_mode.test_session_matches_active_scope(session, active_dialogue_id=3, active_topic_id=10, memory_mode=MEMORY_MODE_TOPIC) is False
    assert memory_mode.test_session_matches_active_scope(session, active_dialogue_id=3, active_topic_id=10, memory_mode=MEMORY_MODE_GLOBAL) is False

    # In GLOBAL mode, same dialogue across different topics matches
    assert memory_mode.test_session_matches_active_scope(session, active_dialogue_id=2, active_topic_id=20, memory_mode=MEMORY_MODE_GLOBAL) is True

    # Unfinished or missing scope does not match
    assert memory_mode.test_session_matches_active_scope(SimpleNamespace(is_finished=False, invocation_dialogue_id=2, invocation_topic_id=10), 2, 10, MEMORY_MODE_TOPIC) is False
    assert memory_mode.test_session_matches_active_scope(SimpleNamespace(is_finished=True, invocation_dialogue_id=None, invocation_topic_id=10), 2, 10, MEMORY_MODE_TOPIC) is False
