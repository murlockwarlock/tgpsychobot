import argparse
import asyncio
import json
import os

import httpx

from kie_chat import build_kie_chat_request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3-flash")
    parser.add_argument("--base-url", default=os.getenv("KIE_BASE_URL", "https://api.kie.ai"))
    parser.add_argument("--api-key", default=os.getenv("KIE_API_KEY"))
    parser.add_argument("--live", action="store_true")
    return parser


def _request(api_key: str, base_url: str, model: str):
    return build_kie_chat_request(
        api_key,
        base_url,
        model,
        [{"role": "user", "content": "hello"}],
        "",
        temperature=0.0,
        max_output_tokens=32,
    )


async def _run(args: argparse.Namespace) -> None:
    api_key = args.api_key or "dry-run-key"
    request = _request(api_key, args.base_url, args.model)
    print(json.dumps({
        "mode": "live" if args.live else "dry-run",
        "endpoint": request.endpoint,
        "protocol": request.protocol,
        "stream": request.stream,
        "payload": request.payload,
    }, ensure_ascii=False, indent=2))
    if not args.live:
        return
    if not args.api_key:
        raise SystemExit("--live requires --api-key or KIE_API_KEY")
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        response = await client.post(
            request.endpoint,
            headers=request.headers,
            json=request.payload,
        )
    try:
        body = response.json()
    except (TypeError, ValueError):
        body = {"text": response.text[:1000]}
    if isinstance(body, dict):
        body = {
            key: body.get(key)
            for key in ("code", "msg", "message", "data", "choices", "output", "candidates")
            if key in body
        }
    print(json.dumps({"http_status": response.status_code, "body": body}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_run(_parser().parse_args()))
