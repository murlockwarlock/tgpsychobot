from __future__ import annotations

from sqlalchemy import delete, update

from database import (
    MediaLibrary,
    Message,
    RandomMessage,
    TestSession,
    Topic,
    TopicMediaDeck,
    User,
    UserTopicState,
    topic_collection_association,
    topic_knowledgebase_association,
)


async def delete_topic_with_dependencies(session, topic_id: int) -> bool:
    """Delete a topic without deleting user history or reusable media."""
    topic = await session.get(Topic, topic_id, with_for_update=True)
    if not topic:
        return False

    await session.execute(update(User).where(User.current_topic_id == topic_id).values(current_topic_id=None))
    await session.execute(update(Message).where(Message.topic_id == topic_id).values(topic_id=None))
    await session.execute(update(MediaLibrary).where(MediaLibrary.topic_id == topic_id).values(topic_id=None))
    await session.execute(update(TestSession).where(TestSession.invocation_topic_id == topic_id).values(invocation_topic_id=None))

    await session.execute(delete(RandomMessage).where(RandomMessage.topic_id == topic_id))
    await session.execute(delete(UserTopicState).where(UserTopicState.topic_id == topic_id))
    await session.execute(delete(TopicMediaDeck).where(TopicMediaDeck.topic_id == topic_id))
    await session.execute(topic_knowledgebase_association.delete().where(topic_knowledgebase_association.c.topic_id == topic_id))
    await session.execute(topic_collection_association.delete().where(topic_collection_association.c.topic_id == topic_id))
    await session.delete(topic)
    return True
