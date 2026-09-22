import asyncio
import json

from provider_adapters import verify_openrouter_catalog


async def main() -> None:
    result = await verify_openrouter_catalog()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if any(not item.get("exists") or not item.get("matches_static") for item in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
