#!/usr/bin/env python3
"""Diagnostic script: Probe KIE Vision Budget and Performance.

Standalone, read-only/diagnostic.
Generates synthetic 1x1 image in-memory.
Probes KIE upload + multimodal chat completion.
Reports:
- upload latency & status
- inference latency & status
- token usage if present
- finish_reason
- effective output length

Safe: does not touch DB, does not mutate config, does not contact Telegram/MAX users.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import httpx

# 1x1 transparent PNG bytes
TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?"
    b"\x03\x05\xfe\x02\xfe\x1c\xdd\x00\x00\x00\x00IEND\xaeB`\x82"
)


import re

_URL_OR_MEDIA_PATTERN = re.compile(
    r"https?:\/\/[^\s\"'<>]+|data:image\/[^\s\"'<>]+|(?:[A-Za-z0-9+/_-]{4}){16,}={0,2}",
    re.IGNORECASE,
)


def _sanitize_probe_diagnostic(val: object) -> str:
    if val is None:
        return ""
    text = str(val)
    return _URL_OR_MEDIA_PATTERN.sub("<redacted_url_or_payload>", text)


async def probe_kie_vision(
    api_key: str,
    base_url: str,
    upload_base_url: str,
    model: str,
    max_tokens: int = 16384,
) -> int:
    print("=" * 60)
    print("KIE Vision Diagnostic Probe")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Upload URL: {upload_base_url}")
    print(f"Max Tokens: {max_tokens}")
    print("=" * 60)

    # 1. Upload stage
    print("\n[Stage 1] Uploading synthetic 1x1 image...")
    upload_url = f"{upload_base_url.rstrip('/')}/api/file-stream-upload"
    files = {"file": ("probe_1x1.png", TINY_PNG_BYTES, "image/png")}
    form_data = {"uploadPath": "images", "fileName": "probe_1x1.png"}
    headers = {"Authorization": f"Bearer {api_key}"}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            resp = await client.post(upload_url, headers=headers, data=form_data, files=files)
        upload_lat = time.monotonic() - t0
        print(f"Upload HTTP status: {resp.status_code} (latency: {upload_lat:.3f}s)")
        try:
            upload_json = resp.json()
        except Exception:
            upload_json = {}
    except Exception as exc:
        print(f"FAILED: Upload error: {type(exc).__name__}: {_sanitize_probe_diagnostic(exc)}")
        return 1

    file_url = None
    if isinstance(upload_json, dict):
        data = upload_json.get("data")
        if isinstance(data, dict):
            file_url = data.get("downloadUrl") or data.get("fileUrl")
        elif isinstance(data, str):
            file_url = data
        if not file_url:
            file_url = upload_json.get("downloadUrl") or upload_json.get("fileUrl")

    if not file_url:
        print(f"FAILED: Upload returned no file URL (HTTP {resp.status_code})")
        return 1

    print("Upload successful.")

    # 2. Inference stage
    print(f"\n[Stage 2] Running multimodal inference with model {model}...")
    inf_url = f"{base_url.rstrip('/')}/{model}/v1/chat/completions"

    prompt_text = (
        "Produce a numbered synthetic sequence of repeated neutral tokens for at least 5000 completion tokens. "
        "Do not summarize. Do not stop early voluntarily."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": file_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }

    t1 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=45.0, trust_env=False) as client:
            resp = await client.post(inf_url, headers=headers, json=payload)
        inf_lat = time.monotonic() - t1
        print(f"Inference HTTP status: {resp.status_code} (latency: {inf_lat:.3f}s)")
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = {}
    except Exception as exc:
        print(f"FAILED: Inference error: {type(exc).__name__}: {_sanitize_probe_diagnostic(exc)}")
        return 1

    choices = resp_json.get("choices") or []
    finish_reason = None
    output_text = ""
    if choices and isinstance(choices, list):
        c = choices[0]
        finish_reason = c.get("finish_reason")
        msg = c.get("message") or {}
        output_text = msg.get("content") or ""

    usage = resp_json.get("usage") or {}
    comp_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
    print("\n[Results]")
    print(f"- Finish reason: {finish_reason}")
    print(f"- Token usage: {usage}")
    print(f"- Output text length: {len(output_text)} characters")
    print(f"- Total probe duration: {(time.monotonic() - t0):.3f}s")
    print("=" * 60)

    if resp.status_code == 200 and output_text:
        if comp_tokens > 4096:
            print(f"Probe status: ELEVATION EVIDENCE OBSERVED (completion_tokens={comp_tokens} > 4096)")
        else:
            print(f"Probe status: SUCCESS (ceiling > 4096 NOT PROVEN; completion_tokens={comp_tokens} <= 4096)")
        return 0
    elif finish_reason == "length":
        print("Probe status: OUTPUT BUDGET EXHAUSTED (finish_reason=length)")
        return 0
    else:
        print(f"Probe status: UNEXPECTED RESPONSE (code={resp_json.get('code')}, msg={_sanitize_probe_diagnostic(resp_json.get('msg'))})")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe KIE Vision budget and endpoints.")
    parser.add_argument("--api-key", default=os.getenv("KIE_API_KEY", ""), help="KIE API Key")
    parser.add_argument("--base-url", default=os.getenv("KIE_BASE_URL", "https://api.kie.ai"), help="KIE Base URL")
    parser.add_argument("--upload-url", default=os.getenv("KIE_UPLOAD_BASE_URL", "https://upload.kie.ai"), help="KIE Upload Base URL")
    parser.add_argument("--model", default="gemini-3-flash", help="KIE multimodal model name")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Initial max tokens budget")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key or KIE_API_KEY env var is required.", file=sys.stderr)
        sys.exit(2)

    code = asyncio.run(probe_kie_vision(
        api_key=args.api_key,
        base_url=args.base_url,
        upload_base_url=args.upload_url,
        model=args.model,
        max_tokens=args.max_tokens,
    ))
    sys.exit(code)


if __name__ == "__main__":
    main()
