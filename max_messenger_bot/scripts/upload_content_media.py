from __future__ import annotations

import argparse
import asyncio

from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.settings import get_settings
from max_messenger_bot.storage import MaxContentMedia, init_storage
from max_messenger_bot.legacy import async_session_maker, init_db


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Upload a local file to MAX and bind it to a content key.")
    parser.add_argument("--content-key", required=True)
    parser.add_argument("--type", required=True, choices=["image", "video", "audio", "file"])
    parser.add_argument("--file", required=True)
    parser.add_argument("--description", default=None)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.max_token:
        raise RuntimeError("MAX_BOT_TOKEN не задан.")

    await init_db()
    await init_storage()

    async with MaxApiClient(settings.max_token, settings.max_api_base) as client:
        result = await client.upload_file(args.type, args.file)
        token = result.get("token")
        if not token:
            raise RuntimeError(f"MAX upload did not return token: {result}")

    async with async_session_maker() as session:
        session.add(
            MaxContentMedia(
                content_key=args.content_key,
                media_type=args.type,
                token=token,
                description=args.description,
            )
        )
        await session.commit()

    print(f"Uploaded and linked to content_key={args.content_key}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

