import os
import unittest

from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import (  # noqa: E402
    Base,
    KnowledgeBase,
    MediaCollection,
    MediaLibrary,
    Message,
    RandomMessage,
    TestSession as DBTestSession,
    Topic,
    TopicMediaDeck,
    User,
    UserTopicState,
    topic_collection_association,
    topic_knowledgebase_association,
)
from topic_management import delete_topic_with_dependencies  # noqa: E402


class TopicDeletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")

        @event.listens_for(self.engine.sync_engine, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_delete_preserves_history_and_media_while_removing_topic_data(self):
        async with self.sessions() as session:
            topic = Topic(name="Удаляемая тема", sort_order=1)
            user = User(id=100, first_name="User", current_topic=topic, current_dialogue_id=1)
            message = Message(user=user, role="user", content="История", topic=topic, dialogue_id=1)
            media = MediaLibrary(topic_id=None, file_id="file", media_type="photo", category="cards")
            phrase = RandomMessage(content="Фраза", category="default")
            kb = KnowledgeBase(filename="kb.txt", indexed_content="text")
            collection = MediaCollection(name="Коллекция")
            session.add_all([topic, user, message, media, phrase, kb, collection])
            await session.flush()
            media.topic_id = topic.id
            phrase.topic_id = topic.id
            session.add_all([
                DBTestSession(user_id=user.id, invocation_topic_id=topic.id, answers="[]"),
                UserTopicState(user_id=user.id, topic_id=topic.id, dialogue_id=1),
                TopicMediaDeck(topic_id=topic.id, deck_name="cards"),
            ])
            await session.execute(topic_knowledgebase_association.insert().values(topic_id=topic.id, knowledge_base_id=kb.id))
            await session.execute(topic_collection_association.insert().values(topic_id=topic.id, collection_id=collection.id))
            topic_id = topic.id
            await session.commit()

        async with self.sessions() as session:
            async with session.begin():
                deleted = await delete_topic_with_dependencies(session, topic_id)
        self.assertTrue(deleted)

        async with self.sessions() as session:
            self.assertIsNone(await session.get(Topic, topic_id))
            self.assertIsNone((await session.get(User, 100)).current_topic_id)
            self.assertIsNone((await session.get(Message, message.id)).topic_id)
            self.assertIsNone((await session.get(MediaLibrary, media.id)).topic_id)
            self.assertIsNone((await session.get(DBTestSession, 100)).invocation_topic_id)
            self.assertEqual(await session.scalar(select(func.count(RandomMessage.id))), 0)
            self.assertEqual(await session.scalar(select(func.count(UserTopicState.user_id))), 0)
            self.assertEqual(await session.scalar(select(func.count(TopicMediaDeck.topic_id))), 0)
            self.assertEqual((await session.execute(select(func.count()).select_from(topic_knowledgebase_association))).scalar_one(), 0)
            self.assertEqual((await session.execute(select(func.count()).select_from(topic_collection_association))).scalar_one(), 0)

    async def test_missing_topic_is_idempotent(self):
        async with self.sessions() as session:
            async with session.begin():
                deleted = await delete_topic_with_dependencies(session, 999)
        self.assertFalse(deleted)


if __name__ == "__main__":
    unittest.main()
