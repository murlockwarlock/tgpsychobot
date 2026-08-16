"""Utilities for hidden structured data returned by AI responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DATA_BLOCK_RE = re.compile(
    r"(?:^[ \t]*```(?:json)?[ \t]*\r?\n)?"
    r"(?:<DATA(?:\s[^>]*)?>\s*(?P<xml>.*?)\s*</DATA>|"
    r"\[DATA\]\s*(?P<square>.*?)\s*\[/DATA\])"
    r"(?:\r?\n[ \t]*```[ \t]*(?=\r?\n|$))?",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
SERVICE_DATA_KEYS = frozenset({"current_state", "events", "metadata", "save_mode"})


@dataclass(frozen=True)
class ServiceDataBlock:
    """Validated service payload hidden from the user-facing AI answer."""

    current_state: dict[str, Any]
    events: list[str]
    metadata: dict[str, Any]
    save_mode: str
    raw_json: str
    legacy: bool = False


def _normalize_service_payload(
    payload: dict[str, Any],
    raw_json: str,
    *,
    xml_syntax: bool,
) -> ServiceDataBlock:
    # Square brackets are the legacy metadata protocol. XML plus envelope keys is
    # the versioned service protocol, so an old metadata field named "events"
    # cannot accidentally be executed as an event.
    is_envelope = xml_syntax and bool(SERVICE_DATA_KEYS.intersection(payload))
    if not is_envelope:
        return ServiceDataBlock({}, [], payload, "merge", raw_json, legacy=True)

    current_state = payload.get("current_state")
    metadata = payload.get("metadata")
    raw_events = payload.get("events")
    save_mode = str(payload.get("save_mode") or "merge").strip().lower()
    if save_mode not in {"merge", "snapshot"}:
        save_mode = "merge"

    events: list[str] = []
    if isinstance(raw_events, str):
        raw_events = [raw_events]
    if isinstance(raw_events, list):
        for event in raw_events:
            if isinstance(event, str) and event.strip():
                events.append(event.strip())
            elif isinstance(event, dict):
                name = event.get("name") or event.get("event")
                if isinstance(name, str) and name.strip():
                    events.append(name.strip())

    return ServiceDataBlock(
        current_state=current_state if isinstance(current_state, dict) else {},
        events=events,
        metadata=metadata if isinstance(metadata, dict) else {},
        save_mode=save_mode,
        raw_json=raw_json,
    )


def _repair_json_payload(raw_json: str) -> dict[str, Any] | None:
    text = (raw_json or "").strip()
    if not text:
        return None
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except (TypeError, json.JSONDecodeError):
        pass

    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        val = json.loads(cleaned)
        if isinstance(val, dict):
            return val
    except (TypeError, json.JSONDecodeError):
        pass

    replaced = re.sub(r"\bTrue\b", "true", cleaned)
    replaced = re.sub(r"\bFalse\b", "false", replaced)
    replaced = re.sub(r"\bNone\b", "null", replaced)
    replaced = re.sub(r",\s*([\}\]])", r"\1", replaced)
    try:
        val = json.loads(replaced)
        if isinstance(val, dict):
            return val
    except (TypeError, json.JSONDecodeError):
        pass

    start_idx = replaced.find("{")
    if start_idx != -1:
        candidate = replaced[start_idx:]
        open_count = 0
        close_count = 0
        in_string = False
        escape = False
        repaired_chars: list[str] = []
        for ch in candidate:
            if escape:
                repaired_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                repaired_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                repaired_chars.append(ch)
                continue
            if not in_string:
                if ch == "{":
                    open_count += 1
                    repaired_chars.append(ch)
                elif ch == "}":
                    if open_count > close_count:
                        close_count += 1
                        repaired_chars.append(ch)
                    else:
                        continue
                else:
                    repaired_chars.append(ch)
            else:
                repaired_chars.append(ch)

        if open_count > close_count:
            repaired_chars.append("}" * (open_count - close_count))

        repaired_text = "".join(repaired_chars)
        repaired_text = re.sub(r",\s*([\}\]])", r"\1", repaired_text)
        try:
            val = json.loads(repaired_text)
            if isinstance(val, dict):
                return val
        except (TypeError, json.JSONDecodeError):
            pass

    return None


def extract_service_data(text: str | None) -> tuple[str, list[ServiceDataBlock], int]:
    """Extract legacy ``[DATA]`` and the unified ``<DATA>`` JSON envelope."""
    raw_text = text or ""
    blocks: list[ServiceDataBlock] = []
    invalid_count = 0

    for match in DATA_BLOCK_RE.finditer(raw_text):
        raw_json = (match.group("xml") or match.group("square") or "").strip()
        payload = _repair_json_payload(raw_json)
        if payload is None or not isinstance(payload, dict):
            invalid_count += 1
            continue
        blocks.append(
            _normalize_service_payload(
                payload,
                raw_json,
                xml_syntax=match.group("xml") is not None,
            )
        )

    visible_text = DATA_BLOCK_RE.sub("", raw_text)
    visible_text = re.sub(r"</?DATA(?:\s[^>]*)?>|\[/?DATA\]", "", visible_text, flags=re.IGNORECASE)
    visible_text = re.sub(r"[ \t]+\n", "\n", visible_text)
    visible_text = re.sub(r"\n{3,}", "\n\n", visible_text).strip()
    return visible_text, blocks, invalid_count


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


def load_metadata_records(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []

    if isinstance(value, dict) and value.get("_format") == "records_v1":
        records = value.get("records")
        if not isinstance(records, list):
            return []
        result = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("data"), dict):
                continue
            data = record["data"]
            result.append({
                "saved_at": record.get("saved_at"),
                "data": data,
                "raw_json": record.get("raw_json") or json.dumps(data, ensure_ascii=False, indent=2),
            })
        return result

    if isinstance(value, dict) and value:
        return [{
            "saved_at": None,
            "data": value,
            "raw_json": json.dumps(value, ensure_ascii=False, indent=2),
        }]
    return []


def append_metadata_records(
    raw: str | None,
    blocks: list[dict[str, Any]],
    *,
    saved_at: str | None = None,
) -> str:
    records = load_metadata_records(raw)
    timestamp = saved_at or datetime.now(timezone.utc).isoformat()
    for block in blocks:
        data = block.get("data")
        if not isinstance(data, dict):
            continue
        records.append({
            "saved_at": timestamp,
            "data": data,
            "raw_json": block.get("raw_json") or json.dumps(data, ensure_ascii=False, indent=2),
        })
    return json.dumps(
        {"_format": "records_v1", "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def extract_data_blocks(text: str | None) -> tuple[str, list[dict[str, Any]], int]:
    """Return visible text, valid data blocks in source order, and invalid count.

    Every [DATA] block is removed from the visible text, including invalid JSON, so a
    formatting error in a prompt never exposes technical data to the client.
    """
    visible_text, service_blocks, invalid_count = extract_service_data(text)
    blocks = [
        {"data": block.metadata, "raw_json": block.raw_json}
        for block in service_blocks
        if block.metadata
    ]
    return visible_text, blocks, invalid_count
