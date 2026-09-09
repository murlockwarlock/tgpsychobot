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


import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

import pytest
from prompt_blocks import (
    DEFAULT_SERVICE_PROMPT_TEMPLATE,
    MAX_CAPABILITIES,
    TELEGRAM_CAPABILITIES,
    ServiceCapabilities,
    build_media_instruction_block,
    render_service_prompt,
)


def test_known_legacy_block_max_capabilities():
    template = (
        DEFAULT_SERVICE_PROMPT_TEMPLATE + "\n\n"
        "СЛУЖЕБНЫЙ БЛОК ДАННЫХ:\n"
        "Используй <DATA>\nis_lead: true\n</DATA> для сохранения лида."
    )
    media_block = build_media_instruction_block("Аудио: audio1\nФото: photo1")
    rendered = render_service_prompt(
        template,
        capabilities=MAX_CAPABILITIES,
        available_media_text="Аудио: audio1\nФото: photo1",
        media_instruction_block=media_block,
    )
    # MAX hard invariants:
    assert "SEND_AUDIO" not in rendered
    assert "RANDOM_IMG" not in rendered
    assert "CHOICE_IMG" not in rendered
    assert "CHOICE_IMG_HIDDEN" not in rendered
    assert "SHOW_IMG" not in rendered
    assert "МЕДИА-ФАЙЛЫ:" not in rendered
    assert "audio1" not in rendered
    assert "photo1" not in rendered

    # Allowed capabilities:
    assert "GEN_IMG" in rendered
    assert "<DATA>" in rendered
    assert "is_lead: true" in rendered


def test_known_legacy_block_telegram_capabilities():
    template = (
        DEFAULT_SERVICE_PROMPT_TEMPLATE + "\n\n"
        "СЛУЖЕБНЫЙ БЛОК ДАННЫХ:\n"
        "Используй <DATA>\nis_lead: true\n</DATA> для сохранения лида."
    )
    media_block = build_media_instruction_block("Аудио: audio1")
    rendered = render_service_prompt(
        template,
        capabilities=TELEGRAM_CAPABILITIES,
        available_media_text="Аудио: audio1",
        media_instruction_block=media_block,
    )
    assert "SEND_AUDIO" in rendered
    assert "CHOICE_IMG" in rendered
    assert "GEN_IMG" in rendered
    assert "<DATA>" in rendered
    assert "Аудио: audio1" in rendered


def test_single_line_unsupported_logical_unit():
    template = (
        "1. Первое правило диалога.\n"
        "2. Для аудио используй SEND_AUDIO: file_id.\n"
        "3. Для генерации используй GEN_IMG: prompt.\n"
    )
    rendered = render_service_prompt(template, capabilities=MAX_CAPABILITIES)
    assert "SEND_AUDIO" not in rendered
    assert "Первое правило диалога." in rendered
    assert "GEN_IMG: prompt." in rendered


def test_multi_line_unsupported_paragraph():
    template = (
        "Вводная информация о системе.\n\n"
        "Для отправки аудиозаписи:\n"
        "используй команду SEND_AUDIO: track_name.\n"
        "Обязательно проверь наличие файла в медиатеке перед отправкой.\n\n"
        "Заключительное правило диалога."
    )
    rendered = render_service_prompt(template, capabilities=MAX_CAPABILITIES)
    assert "SEND_AUDIO" not in rendered
    assert "Для отправки аудиозаписи:" not in rendered
    assert "Обязательно проверь наличие файла" not in rendered
    assert "Вводная информация о системе." in rendered
    assert "Заключительное правило диалога." in rendered


def test_bulleted_unsupported_instruction():
    template = (
        "- Режим генерации изображений: GEN_IMG: prompt\n"
        "- Режим воспроизведения аудио: SEND_AUDIO: track\n"
        "- Режим работы с данными: используй блок <DATA>\n"
    )
    rendered = render_service_prompt(template, capabilities=MAX_CAPABILITIES)
    assert "SEND_AUDIO" not in rendered
    assert "- Режим генерации изображений: GEN_IMG: prompt" in rendered
    assert "- Режим работы с данными: используй блок <DATA>" in rendered


def test_unsupported_unit_plus_unrelated_following_suffix():
    template = (
        "ИНСТРУКЦИЯ ПО АУДИО:\n"
        "Используй команду SEND_AUDIO: id.\n\n"
        "ОБЩИЕ ПРАВИЛА:\n"
        "Всегда отвечай вежливо и профессионально."
    )
    rendered = render_service_prompt(template, capabilities=MAX_CAPABILITIES)
    assert "SEND_AUDIO" not in rendered
    assert "ИНСТРУКЦИЯ ПО АУДИО:" not in rendered
    assert "ОБЩИЕ ПРАВИЛА:" in rendered
    assert "Всегда отвечай вежливо и профессионально." in rendered


def test_mixed_unsupported_and_unrelated_text_in_same_unit_removed():
    # Contract 10: If a single logical unit mixes unsupported command with unrelated text,
    # safety has priority: the WHOLE unit is removed.
    template = (
        "Правило 1: Будь вежлив.\n\n"
        "Если клиент просит медитацию, сначала уточни его самочувствие, а затем вызови SEND_AUDIO: meditation_track.\n\n"
        "Правило 2: Следи за таймингом."
    )
    rendered = render_service_prompt(template, capabilities=MAX_CAPABILITIES)
    assert "SEND_AUDIO" not in rendered
    assert "Если клиент просит медитацию" not in rendered
    assert "уточни его самочувствие" not in rendered
    assert "Правило 1: Будь вежлив." in rendered
    assert "Правило 2: Следи за таймингом." in rendered


def test_custom_data_section_retained_on_max():
    template = (
        "ПОЛЬЗОВАТЕЛЬСКИЕ ДАННЫЕ:\n"
        "Всегда сохраняй статус клиента в формате <DATA>{\"status\": \"active\"}</DATA>.\n"
    )
    rendered = render_service_prompt(template, capabilities=MAX_CAPABILITIES)
    assert "<DATA>" in rendered
    assert "{\"status\": \"active\"}" in rendered


def test_custom_gen_img_retained_on_max():
    template = (
        "ГЕНЕРАЦИЯ КАРТИНОК:\n"
        "Если запрошена визуализация, вызови GEN_IMG: cute cat.\n"
    )
    rendered = render_service_prompt(template, capabilities=MAX_CAPABILITIES)
    assert "GEN_IMG: cute cat." in rendered


def test_telegram_media_sections_preserved():
    template = (
        "ИНСТРУКЦИЯ МЕДИА:\n"
        "- Аудио: SEND_AUDIO: track_1\n"
        "- Фото: SHOW_IMG: photo_1\n"
        "- Случайное: RANDOM_IMG: nature\n"
        "- Выбор: CHOICE_IMG: mood\n"
        "- Скрытый выбор: CHOICE_IMG_HIDDEN: secret\n"
        "- Генерация: GEN_IMG: art\n"
        "- Данные: <DATA>test</DATA>\n"
    )
    rendered = render_service_prompt(template, capabilities=TELEGRAM_CAPABILITIES)
    assert "SEND_AUDIO: track_1" in rendered
    assert "SHOW_IMG: photo_1" in rendered
    assert "RANDOM_IMG: nature" in rendered
    assert "CHOICE_IMG: mood" in rendered
    assert "CHOICE_IMG_HIDDEN: secret" in rendered
    assert "GEN_IMG: art" in rendered
    assert "<DATA>test</DATA>" in rendered


@pytest.mark.asyncio
async def test_max_direct_gets_max_capabilities():
    from unittest.mock import AsyncMock, MagicMock
    from ai_request_builder import build_isolated_request_layout

    mock_session = AsyncMock()
    mock_user = MagicMock(id=1, name="Тест", gender="Мужской", age=30, current_topic=None)
    mock_config = MagicMock(
        general_system_prompt="Общий промпт.",
        shared_prompt_block="",
        service_prompt_block=DEFAULT_SERVICE_PROMPT_TEMPLATE + "\n\n<DATA>\nis_lead: true\n</DATA>",
    )

    layout = await build_isolated_request_layout(
        mock_session,
        user=mock_user,
        ai_config=mock_config,
        system_prompt="Прямой системный промпт.",
        user_prompt="Пользовательский запрос.",
        dialogue_id=1,
        service_capabilities=MAX_CAPABILITIES,
    )

    # Check shared_instructions in isolated layout:
    joined_instructions = "\n".join(layout.shared_instructions)
    assert "SEND_AUDIO" not in joined_instructions
    assert "RANDOM_IMG" not in joined_instructions
    assert "CHOICE_IMG" not in joined_instructions
    assert "SHOW_IMG" not in joined_instructions
    assert "GEN_IMG" in joined_instructions
    assert "<DATA>" in joined_instructions

