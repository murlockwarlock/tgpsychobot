from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration
from ai_request_context import AIRequestLayout
from database import (
    AIConfig,
    Base,
    MediaCollection,
    MediaLibrary,
    SubscriptionConfig,
    Topic,
    User,
    media_collection_items,
    topic_collection_association,
)
import max_messenger_bot.ai as max_ai
from prompt_blocks import (
    DEFAULT_MEDIA_RULES_TEMPLATE,
    DEFAULT_SERVICE_PROMPT_TEMPLATE,
    build_media_instruction_block,
    render_prompt_block,
)

LEGACY_SERVICE_PROMPT_TEMPLATE = (
    "ТЕХНИЧЕСКИЕ ПРАВИЛА ОФОРМЛЕНИЯ (СТРОГО):\n"
    "1. Формат: Используй Markdown.\n\n"
    "📷 ВИЗУАЛИЗАЦИЯ (ГЕНЕРАЦИЯ):\n"
    "GEN_IMG: [подробный промпт на АНГЛИЙСКОМ языке]\n\n"
    "🎵 ДОСТУПНЫЙ МЕДИА-КОНТЕНТ В ЭТОЙ ТЕМЕ:\n"
    "{available_media_text}\n\n"
    "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ:\n"
    "1. АУДИО: [SEND_AUDIO: имя_файла] — отправить аудиофайл.\n"
    "2. КАРТЫ (ТАРО/МАК): В тегах указывай КАТЕГОРИЮ.\n"
    "3. [SHOW_IMG: имя_файла] — показать конкретную карту по имени файла.\n"
    "4. ВАЖНО: После RANDOM_IMG НЕ пиши интерпретацию.\n"
    "5. Теги с новой строки. Не выдумывай категории и имена файлов.\n\n"
    "{test_context_injection}\n"
    "{short_response_instruction}"
)


@pytest_asyncio.fixture
async def db_session_factory(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test_media_prompt.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield sessions
    finally:
        await engine.dispose()


# ═══════════════════════════════════════════════════════════════
#  1. Unit Tests: prompt_blocks rendering
# ═══════════════════════════════════════════════════════════════

def test_build_media_instruction_block_empty_or_whitespace():
    assert build_media_instruction_block(None) == ""
    assert build_media_instruction_block("") == ""
    assert build_media_instruction_block("   ") == ""
    assert build_media_instruction_block("Доступные медиа-файлы: не загружены.\n") == ""


def test_build_media_instruction_block_with_files():
    media_text = "Доступные медиа-файлы в этой теме:\n  - [PHOTO] card_fool.jpg — Дурак\n"
    block = build_media_instruction_block(media_text)
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" in block
    assert "card_fool.jpg" in block
    assert "SHOW_IMG" in block
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" in block


def test_render_prompt_block_modern_template_without_media():
    rendered = render_prompt_block(
        DEFAULT_SERVICE_PROMPT_TEMPLATE,
        available_media_text="",
        media_instruction_block="",
    )
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" not in rendered
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" not in rendered
    assert "GEN_IMG" in rendered


def test_render_prompt_block_modern_template_with_media():
    media_text = "Доступные медиа-файлы в этой теме:\n  - [PHOTO] sun.jpg\n"
    media_block = build_media_instruction_block(media_text)
    rendered = render_prompt_block(
        DEFAULT_SERVICE_PROMPT_TEMPLATE,
        available_media_text=media_text,
        media_instruction_block=media_block,
    )
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" in rendered
    assert "sun.jpg" in rendered
    assert "SHOW_IMG" in rendered
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" in rendered


def test_render_prompt_block_legacy_template_without_media_omits_media_block():
    rendered = render_prompt_block(
        LEGACY_SERVICE_PROMPT_TEMPLATE,
        available_media_text="",
        media_instruction_block="",
    )
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" not in rendered
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" not in rendered
    assert "GEN_IMG" in rendered


def test_render_prompt_block_legacy_template_with_media_keeps_media_block():
    media_text = "Доступные медиа-файлы в этой теме:\n  - [PHOTO] tarot_magician.jpg\n"
    rendered = render_prompt_block(
        LEGACY_SERVICE_PROMPT_TEMPLATE,
        available_media_text=media_text,
    )
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" in rendered
    assert "tarot_magician.jpg" in rendered
    assert "SHOW_IMG" in rendered


# ═══════════════════════════════════════════════════════════════
#  2. Integration Tests: AI Request Context & Payload
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_topic_without_media_omits_entire_media_instruction_block(db_session_factory, monkeypatch):
    """Scenario 1: Topic exists and is selected. No media available."""
    async with db_session_factory() as session:
        topic = Topic(id=1, name="Отношения", system_prompt="Промпт отношений")
        user = User(id=101, username="testuser", first_name="Иван", current_topic_id=1, current_dialogue_id=1)
        ai_config = AIConfig(
            id=1,
            provider="OpenAI",
            openai_api_key="sk-test",
            openai_model="gpt-5.6-turbo",
            service_prompt_block=DEFAULT_SERVICE_PROMPT_TEMPLATE,
        )
        sub_config = SubscriptionConfig(id=1)
        session.add_all([topic, user, ai_config, sub_config])
        await session.commit()

    captured_layouts = []

    async def mock_call_openai(api_key, model, history, context, system_prompt, *args, **kwargs):
        layout = kwargs.get("request_layout")
        if layout:
            captured_layouts.append(layout)
        return "Ответ модели"

    monkeypatch.setattr(ai_integration, "async_session_maker", db_session_factory)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_call_openai)

    response = await ai_integration.get_ai_response(101, "Привет", "Иван", "male")
    assert response == "Ответ модели"
    assert len(captured_layouts) == 1

    layout = captured_layouts[0]
    full_system_context = "\n\n".join(layout.shared_instructions) + "\n\n" + layout.stable_system_prompt
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" not in full_system_context
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" not in full_system_context
    assert "{available_media_text}" not in full_system_context
    assert "{media_instruction_block}" not in full_system_context
    assert "GEN_IMG" in full_system_context


@pytest.mark.asyncio
async def test_topic_with_media_includes_media_block_and_show_img(db_session_factory, monkeypatch):
    """Scenario 2: Topic exists with real usable media attached."""
    async with db_session_factory() as session:
        topic = Topic(id=2, name="Таро", system_prompt="Промпт таро")
        user = User(id=102, username="tarot_user", first_name="Анна", current_topic_id=2, current_dialogue_id=1)
        ai_config = AIConfig(
            id=1,
            provider="OpenAI",
            openai_api_key="sk-test",
            openai_model="gpt-5.6-turbo",
            service_prompt_block=DEFAULT_SERVICE_PROMPT_TEMPLATE,
        )
        sub_config = SubscriptionConfig(id=1)
        coll = MediaCollection(id=10, name="Колода таро")
        media1 = MediaLibrary(
            id=100,
            file_id="fid_100",
            file_name="fool.jpg",
            category="tarot",
            media_type="photo",
            description="Шут — начало пути",
        )
        media2 = MediaLibrary(
            id=101,
            file_id="fid_101",
            file_name="magician.jpg",
            category="tarot",
            media_type="photo",
            description="Маг — мастерство",
        )
        session.add_all([topic, user, ai_config, sub_config, coll, media1, media2])
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=2, collection_id=10))
        await session.execute(media_collection_items.insert().values([
            {"collection_id": 10, "media_id": 100},
            {"collection_id": 10, "media_id": 101},
        ]))
        await session.commit()

    captured_layouts = []

    async def mock_call_openai(api_key, model, history, context, system_prompt, *args, **kwargs):
        layout = kwargs.get("request_layout")
        if layout:
            captured_layouts.append(layout)
        return "Ответ таро"

    monkeypatch.setattr(ai_integration, "async_session_maker", db_session_factory)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_call_openai)

    response = await ai_integration.get_ai_response(102, "Вытяни карту", "Анна", "female")
    assert response == "Ответ таро"
    assert len(captured_layouts) == 1

    layout = captured_layouts[0]
    full_system_context = "\n\n".join(layout.shared_instructions) + "\n\n" + layout.stable_system_prompt
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" in full_system_context
    assert "fool.jpg" in full_system_context
    assert "Шут — начало пути" in full_system_context
    assert "magician.jpg" in full_system_context
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" in full_system_context
    assert "[SHOW_IMG: имя_файла]" in full_system_context


@pytest.mark.asyncio
async def test_empty_or_unusable_collection_omits_media_block(db_session_factory, monkeypatch):
    """Scenario 3: Collection association exists, but resolves to zero usable media items."""
    async with db_session_factory() as session:
        topic = Topic(id=3, name="Пустая тема", system_prompt="Промпт пустой")
        user = User(id=103, username="empty_user", first_name="Тест", current_topic_id=3, current_dialogue_id=1)
        ai_config = AIConfig(
            id=1,
            provider="OpenAI",
            openai_api_key="sk-test",
            openai_model="gpt-5.6-turbo",
            service_prompt_block=DEFAULT_SERVICE_PROMPT_TEMPLATE,
        )
        sub_config = SubscriptionConfig(id=1)
        coll = MediaCollection(id=20, name="Пустая коллекция")
        # Media item with empty file_name (unusable)
        media_unusable = MediaLibrary(
            id=200,
            file_id="fid_200",
            file_name="",
            category="tarot",
            media_type="photo",
            description="Невалидный файл",
        )
        session.add_all([topic, user, ai_config, sub_config, coll, media_unusable])
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=3, collection_id=20))
        await session.execute(media_collection_items.insert().values(collection_id=20, media_id=200))
        await session.commit()

    captured_layouts = []

    async def mock_call_openai(api_key, model, history, context, system_prompt, *args, **kwargs):
        layout = kwargs.get("request_layout")
        if layout:
            captured_layouts.append(layout)
        return "Ответ пустой темы"

    monkeypatch.setattr(ai_integration, "async_session_maker", db_session_factory)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_call_openai)

    response = await ai_integration.get_ai_response(103, "Вопрос", "Тест", "unknown")
    assert response == "Ответ пустой темы"
    assert len(captured_layouts) == 1

    layout = captured_layouts[0]
    full_system_context = "\n\n".join(layout.shared_instructions) + "\n\n" + layout.stable_system_prompt
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" not in full_system_context
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" not in full_system_context


@pytest.mark.asyncio
async def test_legacy_database_service_prompt_block_omits_media_when_empty(db_session_factory, monkeypatch):
    """Legacy service prompt block stored in database from older release."""
    async with db_session_factory() as session:
        topic = Topic(id=4, name="Тема со старым шаблоном", system_prompt="Старый промпт")
        user = User(id=104, username="legacy_user", first_name="Олег", current_topic_id=4, current_dialogue_id=1)
        ai_config = AIConfig(
            id=1,
            provider="OpenAI",
            openai_api_key="sk-test",
            openai_model="gpt-5.6-turbo",
            service_prompt_block=LEGACY_SERVICE_PROMPT_TEMPLATE,
        )
        sub_config = SubscriptionConfig(id=1)
        session.add_all([topic, user, ai_config, sub_config])
        await session.commit()

    captured_layouts = []

    async def mock_call_openai(api_key, model, history, context, system_prompt, *args, **kwargs):
        layout = kwargs.get("request_layout")
        if layout:
            captured_layouts.append(layout)
        return "Ответ"

    monkeypatch.setattr(ai_integration, "async_session_maker", db_session_factory)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_call_openai)

    await ai_integration.get_ai_response(104, "Тест", "Олег", "male")
    assert len(captured_layouts) == 1

    layout = captured_layouts[0]
    full_system_context = "\n\n".join(layout.shared_instructions) + "\n\n" + layout.stable_system_prompt
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" not in full_system_context
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" not in full_system_context
    assert "GEN_IMG" in full_system_context


@pytest.mark.asyncio
async def test_main_dialogue_without_media_omits_media_instruction_block(db_session_factory, monkeypatch):
    """Main dialogue (active_topic_id=None) without media."""
    async with db_session_factory() as session:
        user = User(id=106, username="main_user", first_name="Юрий", current_topic_id=None, current_dialogue_id=1)
        ai_config = AIConfig(
            id=1,
            provider="OpenAI",
            openai_api_key="sk-test",
            openai_model="gpt-5.6-turbo",
            service_prompt_block=DEFAULT_SERVICE_PROMPT_TEMPLATE,
        )
        sub_config = SubscriptionConfig(id=1)
        session.add_all([user, ai_config, sub_config])
        await session.commit()

    captured_layouts = []

    async def mock_call_openai(api_key, model, history, context, system_prompt, *args, **kwargs):
        layout = kwargs.get("request_layout")
        if layout:
            captured_layouts.append(layout)
        return "Ответ"

    monkeypatch.setattr(ai_integration, "async_session_maker", db_session_factory)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_call_openai)

    await ai_integration.get_ai_response(106, "Привет", "Юрий", "male")
    assert len(captured_layouts) == 1

    layout = captured_layouts[0]
    full_system_context = "\n\n".join(layout.shared_instructions) + "\n\n" + layout.stable_system_prompt
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" not in full_system_context
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" not in full_system_context


@pytest.mark.asyncio
async def test_max_ai_does_not_contain_media_instructions(db_session_factory, monkeypatch):
    """Scenario 4: MAX platform AI integration check."""
    async with db_session_factory() as session:
        topic = Topic(id=5, name="MAX Тема", system_prompt="MAX системный промпт")
        user = User(id=105, first_name="Макс", current_topic_id=5, current_dialogue_id=1)
        ai_config = AIConfig(
            id=1,
            provider="OpenAI",
            openai_api_key="sk-test",
            openai_model="gpt-5.6-turbo",
            shared_prompt_block="",
        )
        session.add_all([topic, user, ai_config])
        await session.commit()

    captured_layouts = []

    async def mock_max_dispatch(config, request_layout):
        captured_layouts.append(request_layout)
        return "Ответ MAX"

    monkeypatch.setattr(max_ai, "async_session_maker", db_session_factory)
    monkeypatch.setattr(max_ai, "_dispatch_provider", mock_max_dispatch)

    response = await max_ai.get_ai_response(105, "Вопрос в MAX")
    assert response == "Ответ MAX"
    assert len(captured_layouts) == 1

    layout = captured_layouts[0]
    full_context = layout.stable_system_prompt + "\n\n" + "\n\n".join(layout.shared_instructions)
    assert "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" not in full_context
    assert "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" not in full_context
