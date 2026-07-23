import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import (
    Base,
    IndexingQueue,
    KnowledgeBase,
    Topic,
    topic_knowledgebase_association,
)
from knowledge_base_admin import (
    delete_knowledge_base_record,
    find_original_kb_file_id,
    kb_text_export_filename,
)
from keyboards import knowledge_base_paginator_keyboard


class KnowledgeBaseAdminTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_finds_latest_completed_original_upload(self):
        now = datetime(2026, 7, 23, 10, 0)
        async with self.sessions() as session:
            entry = KnowledgeBase(filename="guide.pdf", indexed_content="text")
            session.add(entry)
            session.add_all([
                IndexingQueue(file_id="old", filename="guide.pdf", status="completed", created_at=now),
                IndexingQueue(file_id="failed", filename="guide.pdf", status="failed", created_at=now + timedelta(minutes=1)),
                IndexingQueue(file_id="new", filename="guide.pdf", status="completed", created_at=now + timedelta(minutes=2)),
            ])
            await session.flush()
            self.assertEqual(await find_original_kb_file_id(session, entry), "new")

    async def test_deletes_topic_links_before_kb_record(self):
        async with self.sessions() as session:
            topic = Topic(name="Тема")
            entry = KnowledgeBase(filename="linked.txt", indexed_content="text")
            session.add_all([topic, entry])
            await session.flush()
            await session.execute(topic_knowledgebase_association.insert().values(
                topic_id=topic.id,
                knowledge_base_id=entry.id,
            ))
            entry_id = entry.id
            self.assertEqual(await delete_knowledge_base_record(session, entry_id), "linked.txt")
            await session.commit()

        async with self.sessions() as session:
            self.assertIsNone(await session.get(KnowledgeBase, entry_id))
            link_count = await session.scalar(
                select(func.count()).select_from(topic_knowledgebase_association)
            )
            self.assertEqual(link_count, 0)
            self.assertEqual(await session.scalar(select(func.count(Topic.id))), 1)

    def test_file_button_downloads_and_text_fallback_has_txt_extension(self):
        markup = knowledge_base_paginator_keyboard(
            0,
            1,
            [SimpleNamespace(id=7, filename="guide.pdf", use_in_general_mode=True)],
        )
        callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
        self.assertIn("download_kb_7", callbacks)
        self.assertEqual(kb_text_export_filename("guide.pdf"), "guide_БЗ.txt")


if __name__ == "__main__":
    unittest.main()
