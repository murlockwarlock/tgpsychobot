"""Utilities for hidden structured data returned by AI responses."""

from __future__ import annotations

import json
import re
from typing import Any


DATA_BLOCK_RE = re.compile(r"\[DATA\]\s*(.*?)\s*\[/DATA\]", re.IGNORECASE | re.DOTALL)


def merge_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge objects; new scalar and list values replace old ones."""
    result = dict(existing)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_metadata(result[key], value)
        else:
            result[key] = value
    return result


def load_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def extract_data_blocks(text: str | None) -> tuple[str, dict[str, Any], int]:
    """Return visible text, all valid data merged together, and invalid block count.

    Every [DATA] block is removed from the visible text, including invalid JSON, so a
    formatting error in a prompt never exposes technical data to the client.
    """
    raw_text = text or ""
    metadata: dict[str, Any] = {}
    invalid_count = 0

    for match in DATA_BLOCK_RE.finditer(raw_text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            invalid_count += 1
            continue
        if not isinstance(payload, dict):
            invalid_count += 1
            continue
        metadata = merge_metadata(metadata, payload)

    visible_text = DATA_BLOCK_RE.sub("", raw_text)
    visible_text = re.sub(r"[ \t]+\n", "\n", visible_text)
    visible_text = re.sub(r"\n{3,}", "\n\n", visible_text).strip()
    return visible_text, metadata, invalid_count
