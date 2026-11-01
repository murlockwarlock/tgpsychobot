import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_request_builder import build_conversational_request_layout, build_isolated_request_layout
from database import AIConfig, Base, BotGeneralConfig, BotTranslation, Content, User
from system_messages import (
    reset_system_message,
    resolve_addressing_style,
    resolve_system_text,
    save_addressing_style,
    save_system_message,
)
from translation_registry import STATIC_TRANSLATION_SOURCES
from translation_service import source_hash


@pytest.mark.asyncio
async def test_system_message_variants_locale_and_database_are_independent(tmp_path):
    formal_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'formal.db'}")
    informal_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'informal.db'}")
    formal_sessions = async_sessionmaker(formal_engine, expire_on_commit=False)
    informal_sessions = async_sessionmaker(informal_engine, expire_on_commit=False)
    try:
        async with formal_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with informal_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with formal_sessions() as session:
            session.add(BotGeneralConfig(id=1, addressing_style="formal"))
            session.add(BotTranslation(
                locale="en",
                translation_key="ui.ai.busy",
                text="Please wait for the previous request to finish.",
                source_hash=source_hash(STATIC_TRANSLATION_SOURCES["ui.ai.busy"][0]),
            ))
            await session.commit()

            assert "Ваше" in await resolve_system_text(session, "ui.ai.busy", "ru")
            assert await resolve_system_text(session, "ui.ai.busy", "en") == "Please wait for the previous request to finish."
            await save_system_message(session, "ui.ai.busy", "ru", "formal", "FORMAL CUSTOM")
            await save_system_message(session, "ui.ai.busy", "ru", "informal", "INFORMAL CUSTOM")
            await session.commit()
            assert await resolve_system_text(session, "ui.ai.busy", "ru", "formal") == "FORMAL CUSTOM"
            assert await resolve_system_text(session, "ui.ai.busy", "ru", "informal") == "INFORMAL CUSTOM"
            await reset_system_message(session, "ui.ai.busy", "ru", "formal")
            await session.commit()
            assert "Ваше" in await resolve_system_text(session, "ui.ai.busy", "ru", "formal")
            assert await resolve_system_text(session, "ui.ai.busy", "ru", "informal") == "INFORMAL CUSTOM"

        async with informal_sessions() as session:
            session.add(BotGeneralConfig(id=1, addressing_style="informal"))
            await session.commit()
            assert "твоё" in await resolve_system_text(session, "ui.ai.busy", "ru")
    finally:
        await formal_engine.dispose()
        await informal_engine.dispose()


@pytest.mark.asyncio
async def test_telegram_and_max_system_message_scopes_are_independent(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scoped-bots.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add(BotGeneralConfig(
                id=1,
                addressing_style="informal",
                ai_processing_message_text="SHARED PROCESSING TEXT",
            ))
            session.add(Content(key="shared_content", text_content="SHARED CONTENT"))
            await save_addressing_style(session, "formal", bot_scope="telegram")
            await save_addressing_style(session, "informal", bot_scope="max")
            await save_system_message(
                session,
                "ui.ai.busy",
                "ru",
                "formal",
                "TELEGRAM TEXT",
                bot_scope="telegram",
            )
            await save_system_message(
                session,
                "ui.ai.busy",
                "ru",
                "informal",
                "MAX TEXT",
                bot_scope="max",
            )
            await session.commit()
            assert await resolve_addressing_style(session, "telegram") == "formal"
            assert await resolve_addressing_style(session, "max") == "informal"
            assert await resolve_system_text(session, "ui.ai.busy", "ru", bot_scope="telegram") == "TELEGRAM TEXT"
            assert await resolve_system_text(session, "ui.ai.busy", "ru", bot_scope="max") == "MAX TEXT"
            general = await session.get(BotGeneralConfig, 1)
            shared_content = await session.get(Content, "shared_content")
            assert general.addressing_style == "formal"
            assert general.ai_processing_message_text == "SHARED PROCESSING TEXT"
            assert shared_content.text_content == "SHARED CONTENT"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_system_message_runtime_preserves_html_and_unknown_placeholders(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'custom-text.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            await save_system_message(
                session,
                "ui.ai.button_accepted",
                "ru",
                "formal",
                "<b>Ваш выбор</b>: {unknown_placeholder}",
                bot_scope="max",
            )
            await session.commit()
            assert await resolve_system_text(
                session,
                "ui.ai.button_accepted",
                "ru",
                "formal",
                bot_scope="max",
            ) == "<b>Ваш выбор</b>: {unknown_placeholder}"
            assert await resolve_system_text(
                session,
                "ui.ai.button_accepted",
                "ru",
                "formal",
                bot_scope="max",
                button_text="готово",
            ) == "<b>Ваш выбор</b>: {unknown_placeholder}"
            with pytest.raises(ValueError):
                await save_system_message(
                    session,
                    "ui.ai.button_accepted",
                    "ru",
                    "formal",
                    "x" * 4097,
                    bot_scope="max",
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ai_layout_includes_addressing_style_without_rewriting_admin_prompt(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'layout.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add_all([
                User(id=77, first_name="Иван", current_dialogue_id=1),
                AIConfig(id=1, provider="OpenAI", system_prompt="Авторский промпт администратора: ты помощник."),
                BotGeneralConfig(id=1, addressing_style="formal"),
            ])
            await session.commit()

        async with sessions() as session:
            user = await session.get(User, 77)
            ai_config = await session.get(AIConfig, 1)
            conversational = await build_conversational_request_layout(
                session,
                user=user,
                ai_config=ai_config,
                dialogue_id=1,
                current_user_content="Вопрос",
            )
            isolated = await build_isolated_request_layout(
                session,
                user=user,
                ai_config=ai_config,
                system_prompt=ai_config.system_prompt,
                user_prompt="Данные",
                dialogue_id=1,
                preferred_response_locale="en",
            )
            assert "на «Вы»" in "\n".join(conversational.runtime_context)
            assert "polite, formal tone" in "\n".join(isolated.runtime_context)
            assert conversational.stable_system_prompt == ai_config.system_prompt
            assert isolated.stable_system_prompt == ai_config.system_prompt

            config = await session.get(BotGeneralConfig, 1)
            config.addressing_style = "informal"
            await session.commit()
            informal = await build_conversational_request_layout(
                session,
                user=user,
                ai_config=ai_config,
                dialogue_id=1,
                current_user_content="Вопрос",
            )
            assert "на «ты»" in "\n".join(informal.runtime_context)
    finally:
        await engine.dispose()
