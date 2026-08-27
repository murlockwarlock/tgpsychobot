from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.legacy import (
    MediaCollection,
    MediaLibrary,
    Topic,
    async_session_maker,
    media_collection_items,
    topic_collection_association,
    init_db,
)
from max_messenger_bot.settings import get_settings
from media_scope import ensure_topic_collection, load_media_scope


async def _ensure_topic_collection(session, topic_id: int, collection_id: int | None) -> int:
    topic = await session.get(Topic, topic_id)
    if not topic:
        raise RuntimeError(f"Topic {topic_id} не найден.")

    if collection_id is not None:
        linked = await session.scalar(
            select(topic_collection_association.c.collection_id).where(
                topic_collection_association.c.topic_id == topic_id,
                topic_collection_association.c.collection_id == collection_id,
            )
        )
        if linked is None:
            raise RuntimeError(f"Collection {collection_id} не привязана к topic {topic_id}.")
        return collection_id

    scope = await load_media_scope(session, topic_id)
    if len(scope.collection_ids) > 1:
        raise RuntimeError("Укажите --collection-id для темы с несколькими коллекциями.")
    return await ensure_topic_collection(session, topic_id)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Upload a local file to MAX and bind it to a topic/media library entry.")
    parser.add_argument("--type", required=True, choices=["image", "video", "audio", "file"])
    parser.add_argument("--file", required=True)
    parser.add_argument("--topic-id", type=int, default=None)
    parser.add_argument("--collection-id", type=int, default=None)
    parser.add_argument("--file-name", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--description", default=None)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.max_token:
        raise RuntimeError("MAX_BOT_TOKEN не задан.")

    await init_db()

    async with MaxApiClient(settings.max_token, settings.max_api_base) as client:
        result = await client.upload_file(args.type, args.file)
        token = result.get("token")
        if not token:
            raise RuntimeError(f"MAX upload did not return token: {result}")

    async with async_session_maker() as session:
        if args.topic_id is None and args.collection_id is None:
            raise RuntimeError("Укажите --topic-id или --collection-id.")
        if args.topic_id is not None:
            collection_id = await _ensure_topic_collection(session, args.topic_id, args.collection_id)
        else:
            collection = await session.get(MediaCollection, args.collection_id)
            if not collection:
                raise RuntimeError(f"Collection {args.collection_id} не найдена.")
            collection_id = collection.id
        media_type = {"image": "photo", "file": "document"}.get(args.type, args.type)
        media = MediaLibrary(
            media_type=media_type,
            file_id=token,
            file_name=args.file_name,
            category=args.category,
            description=args.description,
        )
        session.add(media)
        await session.flush()
        await session.execute(
            media_collection_items.insert().values(
                collection_id=collection_id,
                media_id=media.id,
            )
        )
        await session.commit()

    print("Uploaded topic media")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
