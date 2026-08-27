from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import and_, false, select

from database import (
    MediaLibrary,
    main_dialogue_collection_association,
    media_collection_items,
    topic_collection_association,
)


PHOTO_MEDIA_TYPES = ("photo", "image")
AI_MEDIA_TYPES = (*PHOTO_MEDIA_TYPES, "audio", "video", "document")


@dataclass(frozen=True)
class MediaScope:
    topic_id: int | None
    collection_ids: tuple[int, ...]
    collection_media_ids: tuple[int, ...]

    @property
    def is_main_dialogue(self) -> bool:
        return self.topic_id is None

    def predicate(self, category: str | None = None):
        if not self.collection_media_ids:
            return false()
        predicate = MediaLibrary.id.in_(self.collection_media_ids)
        if category:
            predicate = and_(predicate, MediaLibrary.category == category)
        return predicate


TopicMediaScope = MediaScope


def make_media_scope(
    topic_id: int | None,
    collection_media_ids: list[int] | tuple[int, ...] | None = None,
    collection_ids: list[int] | tuple[int, ...] | None = None,
) -> MediaScope:
    return MediaScope(
        topic_id=topic_id or None,
        collection_ids=tuple(collection_ids or ()),
        collection_media_ids=tuple(collection_media_ids or ()),
    )


def make_topic_media_scope(
    topic_id: int | None,
    collection_media_ids: list[int] | tuple[int, ...] | None = None,
    assigned_decks: list[str] | tuple[str, ...] | None = None,
) -> MediaScope:
    return make_media_scope(topic_id, collection_media_ids=collection_media_ids)


async def load_media_scope(session, topic_id: int | None) -> MediaScope:
    normalized_topic_id = topic_id or None
    if normalized_topic_id is None:
        collection_result = await session.execute(
            select(main_dialogue_collection_association.c.collection_id)
        )
    else:
        collection_result = await session.execute(
            select(topic_collection_association.c.collection_id).where(
                topic_collection_association.c.topic_id == normalized_topic_id
            )
        )
    collection_ids = tuple(sorted({row[0] for row in collection_result.all()}))
    if not collection_ids:
        return make_media_scope(normalized_topic_id)

    media_result = await session.execute(
        select(media_collection_items.c.media_id)
        .where(media_collection_items.c.collection_id.in_(collection_ids))
        .distinct()
    )
    media_ids = tuple(sorted({row[0] for row in media_result.all()}))
    return make_media_scope(
        normalized_topic_id,
        collection_media_ids=media_ids,
        collection_ids=collection_ids,
    )


async def load_topic_media_scope(session, topic_id: int | None) -> MediaScope:
    return await load_media_scope(session, topic_id)


async def load_available_media(session, topic_id: int | None):
    scope = await load_media_scope(session, topic_id)
    result = await session.execute(
        select(MediaLibrary)
        .where(
            scope.predicate(),
            MediaLibrary.media_type.in_(AI_MEDIA_TYPES),
            MediaLibrary.file_name.is_not(None),
            MediaLibrary.file_name != "",
        )
        .order_by(MediaLibrary.id)
    )
    return scope, result.scalars().all()


def photo_media_predicate():
    return MediaLibrary.media_type.in_(PHOTO_MEDIA_TYPES)
