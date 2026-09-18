import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_request_builder import build_conversational_request_layout, build_isolated_request_layout
from database import AIConfig, Base, Topic, User


@pytest.mark.asyncio
async def test_language_directive_is_runtime_only_for_both_builders(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ai.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            user = User(id=77, first_name="Иван", current_dialogue_id=1, current_topic_id=1)
            topic = Topic(id=1, name="Тема", is_active=True, system_prompt="Тематический промпт")
            config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="test",
                openai_model="test-model",
                system_prompt="Стабильный промпт",
            )
            session.add_all([user, topic, config])
            await session.commit()

        async with sessions() as session:
            user = await session.get(User, 77)
            config = await session.get(AIConfig, 1)
            for locale, expected in ((None, None), ("ru", None), ("MAX", None), ("en", "English"), ("pt", "Portuguese")):
                conversational = await build_conversational_request_layout(
                    session,
                    user=user,
                    ai_config=config,
                    dialogue_id=1,
                    topic_id=1,
                    current_user_content="Вопрос",
                    preferred_response_locale=locale,
                )
                isolated = await build_isolated_request_layout(
                    session,
                    user=user,
                    ai_config=config,
                    system_prompt="Изолированный промпт",
                    user_prompt="Данные",
                    dialogue_id=1,
                    topic_id=1,
                    preferred_response_locale=locale,
                )
                for layout in (conversational, isolated):
                    runtime = "\n".join(layout.runtime_context)
                    if expected is None:
                        assert "LANGUAGE DIRECTIVE" not in runtime
                    else:
                        assert f"Respond to the user in {expected}" in runtime
                        assert "LANGUAGE DIRECTIVE" not in layout.stable_system_prompt
                        assert "LANGUAGE DIRECTIVE" not in str(layout.shared_instructions)
                        assert "LANGUAGE DIRECTIVE" not in str(layout.scenario_context)
                        assert "LANGUAGE DIRECTIVE" not in str(layout.request_context)
                        assert "LANGUAGE DIRECTIVE" not in str(layout.history)
                        assert "LANGUAGE DIRECTIVE" not in str(layout.current_user_content)
    finally:
        await engine.dispose()
