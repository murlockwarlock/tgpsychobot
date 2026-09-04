import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sqlalchemy.ext.asyncio as sqlalchemy_asyncio
from sqlalchemy.pool import StaticPool

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

original_create_async_engine = sqlalchemy_asyncio.create_async_engine


def _sqlite_compatible_engine(*args, **kwargs):
    kwargs.pop("pool_recycle", None)
    kwargs.pop("pool_use_lifo", None)
    return original_create_async_engine(*args, **kwargs)


with patch.object(sqlalchemy_asyncio, "create_async_engine", _sqlite_compatible_engine):
    import database
    from database import Base, Content, async_session_maker
    import max_messenger_bot.legacy as max_legacy
    import max_messenger_bot.storage as max_storage
    from max_messenger_bot import app as max_app
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.services import common as max_common
    from max_messenger_bot.storage import MaxContentMedia, StorageBase


class MaxWelcomeMediaRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = sqlalchemy_asyncio.create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(StorageBase.metadata.create_all)

        self.session_factory = sqlalchemy_asyncio.async_sessionmaker(self.engine, expire_on_commit=False)

        self.patches = [
            patch.object(database, "async_session_maker", self.session_factory),
            patch.object(max_legacy, "async_session_maker", self.session_factory),
            patch.object(max_storage, "async_session_maker", self.session_factory),
            patch.object(max_common, "async_session_maker", self.session_factory),
            patch.object(max_app, "async_session_maker", self.session_factory),
        ]
        for p in self.patches:
            p.start()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()
        await self.engine.dispose()

    async def test_start_message_media_single_card_layout(self):
        """When start_message has a photo/media attachment, render_static_content sends a SINGLE message with media + text + buttons."""
        async with self.session_factory() as session:
            start_content = Content(
                key="start_message",
                text_content="Добро пожаловать в бот!\n[BTN:Начать:svc:continue]",
                is_visible=True,
                content_order="media_top",
            )
            media = MaxContentMedia(
                content_key="start_message",
                media_type="photo",
                token="media_tok_123",
            )
            session.add_all([start_content, media])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()

        rendered = await max_common.render_static_content(client, chat_id=501, user_id=501, content_key="start_message", is_start=True)
        self.assertTrue(rendered)

        # Must send exactly 1 message (single-card layout)
        self.assertEqual(client.send_message.call_count, 1)
        call_kwargs = client.send_message.call_args.kwargs
        self.assertIn("Добро пожаловать", call_kwargs["text"])
        attachments = call_kwargs["attachments"]
        # Attachments must contain image + inline_keyboard
        att_types = [a.get("type") for a in attachments]
        self.assertIn("image", att_types)
        self.assertIn("inline_keyboard", att_types)

    async def test_ordinary_content_media_top_preserves_two_messages(self):
        """Ordinary non-start content with order='media_top' preserves two messages (media first, text second)."""
        async with self.session_factory() as session:
            custom_content = Content(
                key="about_us",
                text_content="Информация о нас",
                is_visible=True,
                content_order="media_top",
            )
            media = MaxContentMedia(
                content_key="about_us",
                media_type="photo",
                token="media_tok_456",
            )
            session.add_all([custom_content, media])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()

        rendered = await max_common.render_static_content(client, chat_id=502, user_id=502, content_key="about_us", is_start=False)
        self.assertTrue(rendered)

        # Must send 2 messages: media first, text second
        self.assertEqual(client.send_message.call_count, 2)
        first_call = client.send_message.call_args_list[0].kwargs
        second_call = client.send_message.call_args_list[1].kwargs
        self.assertEqual(first_call["text"], "")
        self.assertEqual(first_call["attachments"][0]["type"], "image")
        self.assertIn("Информация о нас", second_call["text"])
