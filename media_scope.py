from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import and_, false, select

from database import (
    MediaCollection,
    MediaLibrary,
    Topic,
    main_dialogue_collection_association,
    media_collection_items,
    topic_collection_association,
)


PHOTO_MEDIA_TYPES = ("photo", "image")
AI_MEDIA_TYPES = (*PHOTO_MEDIA_TYPES, "audio", "video", "document")


def _media_membership_predicate(collection_ids):
    return (
        select(1)
        .select_from(media_collection_items)
        .where(
            media_collection_items.c.collection_id.in_(collection_ids),
            media_collection_items.c.media_id == MediaLibrary.id,
        )
        .exists()
    )


def media_scope_predicate(topic_id: int | None, category: str | None = None):
    normalized_topic_id = topic_id or None
    if normalized_topic_id is None:
        collection_ids = select(main_dialogue_collection_association.c.collection_id)
    else:
        collection_ids = select(topic_collection_association.c.collection_id).where(
            topic_collection_association.c.topic_id == normalized_topic_id
        )
    predicate = _media_membership_predicate(collection_ids)
    if category:
        predicate = and_(predicate, MediaLibrary.category == category)
    return predicate


@dataclass(frozen=True)
class MediaScope:
    topic_id: int | None
    collection_ids: tuple[int, ...]
    collection_media_ids: tuple[int, ...]

    @property
    def is_main_dialogue(self) -> bool:
        return self.topic_id is None

    def predicate(self, category: str | None = None):
        if self.collection_ids:
            predicate = _media_membership_predicate(self.collection_ids)
        elif self.collection_media_ids:
            predicate = MediaLibrary.id.in_(self.collection_media_ids)
        else:
            return false()
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


async def load_media_scope(
    session,
    topic_id: int | None,
    *,
    include_media_ids: bool = True,
) -> MediaScope:
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
    if not collection_ids or not include_media_ids:
        return make_media_scope(normalized_topic_id, collection_ids=collection_ids)

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


async def load_topic_media_scope(
    session,
    topic_id: int | None,
    *,
    include_media_ids: bool = True,
) -> MediaScope:
    return await load_media_scope(session, topic_id, include_media_ids=include_media_ids)


async def ensure_topic_collection(session, topic_id: int, name_prefix: str = "__max_topic") -> int | None:
    scope = await load_media_scope(session, topic_id, include_media_ids=False)
    if scope.collection_ids:
        return scope.collection_ids[0]
    if not await session.get(Topic, topic_id):
        return None

    name = f"{name_prefix}_{topic_id}"
    collection = await session.scalar(select(MediaCollection).where(MediaCollection.name == name))
    if collection is None:
        collection = MediaCollection(name=name)
        session.add(collection)
        await session.flush()
    linked = await session.scalar(
        select(topic_collection_association.c.collection_id).where(
            topic_collection_association.c.topic_id == topic_id,
            topic_collection_association.c.collection_id == collection.id,
        )
    )
    if linked is None:
        await session.execute(
            topic_collection_association.insert().values(
                topic_id=topic_id,
                collection_id=collection.id,
            )
        )
    return collection.id


async def load_available_media(session, topic_id: int | None):
    scope = await load_media_scope(session, topic_id, include_media_ids=False)
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
