"""Utilities for hidden structured data returned by AI responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DATA_BLOCK_RE = re.compile(
    r"<DATA(?:\s+[^>\r\n]*)?>|\[DATA\]",
    re.IGNORECASE,
)
DATA_CLOSE_RE = {
    "xml": re.compile(r"</DATA>", re.IGNORECASE),
    "square": re.compile(r"\[/DATA\]", re.IGNORECASE),
}
TRUNCATED_XML_OPEN_RE = re.compile(
    r"<DATA(?:\s+[^>\r\n]*)?(?=\r?\n|$)",
    re.IGNORECASE,
)
SERVICE_DATA_KEYS = frozenset({
    "current_state",
    "current_step",
    "event",
    "events",
    "metadata",
    "save_mode",
})
FALLBACK_SIGNATURE_RE = re.compile(
    r"^[ \t]*(?:служебный блок данных|заполненный current_state|json|data:)[ \t]*:?$",
    re.IGNORECASE,
)
MARKDOWN_FENCE_RE = re.compile(
    r"[ \t]*`{3,}(?:[a-zA-Z0-9_-]+)?[ \t]*(?:\r?\n)?$"
)


@dataclass(frozen=True)
class ServiceDataBlock:
    """Validated service payload hidden from the user-facing AI answer."""

    current_state: dict[str, Any]
    events: list[str]
    metadata: dict[str, Any]
    save_mode: str
    raw_json: str
    legacy: bool = False


@dataclass(frozen=True)
class ServiceDataExtraction:
    visible_text: str
    blocks: list[ServiceDataBlock]
    invalid_count: int
    classification: str


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
    if not isinstance(current_state, dict):
        current_state = {}
    current_state = dict(current_state)
    current_step = payload.get("current_step")
    if (
        isinstance(current_step, str)
        and current_step.strip()
        and not current_state.get("current_step")
    ):
        current_state["current_step"] = current_step.strip()
    metadata = payload.get("metadata")
    raw_events = payload.get("events")
    if not raw_events and payload.get("event"):
        raw_events = payload.get("event")
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
        current_state=current_state,
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
            return None

        repaired_text = "".join(repaired_chars)
        repaired_text = re.sub(r",\s*([\}\]])", r"\1", repaired_text)
        try:
            val = json.loads(repaired_text)
            if isinstance(val, dict):
                return val
        except (TypeError, json.JSONDecodeError):
            pass

    return None


SELF_CLOSING_DATA_RE = re.compile(
    r"<DATA(?:\s+[^>]*)?\s*/>",
    re.IGNORECASE,
)

def _line_start(text: str, position: int) -> int:
    return text.rfind("\n", 0, position) + 1


def _is_line_start_tag(text: str, position: int) -> bool:
    return not text[_line_start(text, position):position].strip()


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _looks_like_json_payload(text: str) -> bool:
    position = _skip_whitespace(text, 0)
    return text[position:position + 1] in {"{", "["} or text[position:].startswith("```")


def _tag_kind(match: re.Match[str]) -> str:
    return "square" if match.group(0).startswith("[") else "xml"


def _is_tag_candidate(
    text: str,
    match: re.Match[str],
    body: str = "",
    *,
    has_close: bool,
) -> bool:
    return _looks_like_json_payload(body) or (
        has_close and _is_line_start_tag(text, match.start())
    )


def _fence_start(text: str, position: int) -> int:
    current_line_start = _line_start(text, position)
    if current_line_start == 0:
        return position
    previous_line_end = current_line_start
    previous_line_start = text.rfind("\n", 0, previous_line_end - 1) + 1
    previous_line = text[previous_line_start:previous_line_end]
    if MARKDOWN_FENCE_RE.fullmatch(previous_line):
        return previous_line_start
    return position


def _fence_end(text: str, position: int) -> int:
    match = re.match(
        r"\r?\n[ \t]*`{3,}(?:[a-zA-Z0-9_-]+)?[ \t]*(?=\r?\n|$)",
        text[position:],
    )
    return position + match.end() if match else position


def _find_tag_close(text: str, start: int, kind: str) -> re.Match[str] | None:
    close_re = DATA_CLOSE_RE[kind]
    in_string = False
    escaped = False
    position = start
    while position < len(text):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            position += 1
            continue
        if char == '"':
            in_string = True
            position += 1
            continue
        match = close_re.match(text, position)
        if match:
            return match
        position += 1
    return None


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    result: list[str] = []
    cursor = 0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    for start, end in merged:
        result.append(text[cursor:start])
        cursor = end
    result.append(text[cursor:])
    return "".join(result)


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    masked = list(text)
    for start, end in spans:
        for position in range(start, end):
            if masked[position] not in "\r\n":
                masked[position] = " "
    return "".join(masked)


def _find_balanced_json_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(text):
        if text[position] not in "{[":
            position += 1
            continue
        stack = [text[position]]
        in_string = False
        escaped = False
        end: int | None = None
        cursor = position + 1
        while cursor < len(text):
            char = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                cursor += 1
                continue
            if char == '"':
                in_string = True
                cursor += 1
                continue
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                pairs = {"}": "{", "]": "["}
                if not stack or stack[-1] != pairs[char]:
                    break
                stack.pop()
                if not stack:
                    end = cursor + 1
                    break
            cursor += 1
        if end is None:
            position += 1
            continue
        try:
            payload = json.loads(text[position:end])
        except (TypeError, json.JSONDecodeError):
            position += 1
            continue
        if isinstance(payload, (dict, list)):
            spans.append((position, end))
            position = end
        else:
            position += 1
    return spans


def _is_inside_json_string(text: str, position: int) -> bool:
    for root_start, root_end in _find_balanced_json_spans(text):
        if not root_start <= position < root_end:
            continue
        in_string = False
        escaped = False
        for char in text[root_start:position]:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
        return in_string
    return False


def _is_fallback_tail(text: str, end: int, *, fenced: bool) -> bool:
    tail = text[end:]
    if not tail.strip():
        return True
    return fenced and bool(re.fullmatch(r"\s*```[ \t]*", tail))


def _signature_line_start(text: str, position: int) -> int | None:
    fence_start = _fence_start(text, position)
    line_start = _line_start(text, fence_start)
    cursor = line_start
    found: int | None = None
    while cursor > 0:
        previous_end = cursor
        previous_start = text.rfind("\n", 0, previous_end - 1) + 1
        line = text[previous_start:previous_end].rstrip("\r\n")
        if not line.strip():
            cursor = previous_start
            continue
        if not FALLBACK_SIGNATURE_RE.fullmatch(line):
            break
        found = previous_start
        cursor = previous_start
    return found


def _fallback_removal_start(text: str, object_start: int) -> tuple[int, bool]:
    fence_start = _fence_start(text, object_start)
    if fence_start != object_start:
        signature_start = _signature_line_start(text, fence_start)
        return signature_start if signature_start is not None else fence_start, True
    signature_start = _signature_line_start(text, object_start)
    if signature_start is not None:
        return signature_start, False
    return object_start, False


def _has_explicit_service_signature(text: str, object_start: int) -> bool:
    signature_start = _signature_line_start(text, object_start)
    if signature_start is None:
        return False
    signature_text = text[signature_start:_line_start(text, object_start)]
    return bool(re.search(
        r"(?im)^[ \t]*(?:служебный блок данных|заполненный current_state|data:)[ \t]*:?$",
        signature_text,
    ))


def _has_unclosed_json_container(text: str, object_start: int) -> bool:
    signature_start = _signature_line_start(text, object_start)
    if signature_start is None:
        context_start = _line_start(text, object_start)
    else:
        newline = text.find("\n", signature_start)
        context_start = len(text) if newline == -1 else newline + 1
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for char in text[context_start:object_start]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack and stack[-1] == pairs[char]:
            stack.pop()
    return bool(stack)


def _is_service_fallback_payload(payload: dict[str, Any], explicit_signature: bool) -> bool:
    if explicit_signature:
        return bool(SERVICE_DATA_KEYS.intersection(payload) or {
            "ws",
            "safety_flag",
        }.intersection(payload))
    if isinstance(payload.get("current_state"), dict):
        return True
    return (
        isinstance(payload.get("current_step"), str)
        and bool(payload.get("current_step", "").strip())
        and bool({"event", "events", "metadata", "save_mode"}.intersection(payload))
    )


def _find_fallback_service_tail(
    text: str,
    excluded_starts: set[int] | None = None,
) -> tuple[int, int] | None:
    excluded_starts = excluded_starts or set()
    signatures = re.finditer(
        r"(?im)^[ \t]*(?:служебный блок данных|заполненный current_state|data:)[ \t]*:?\s*(?:\r?\n|$)",
        text,
    )
    for signature in signatures:
        if signature.start() in excluded_starts:
            continue
        if _looks_like_json_payload(text[signature.end():]):
            return signature.start(), len(text)
    return None


def _normalize_visible_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_tagged_data(
    raw_text: str,
) -> tuple[str, list[tuple[int, ServiceDataBlock]], int, list[tuple[int, int]]]:
    located_blocks: list[tuple[int, ServiceDataBlock]] = []
    removals: list[tuple[int, int]] = []
    invalid_count = 0
    search_position = 0

    while search_position < len(raw_text):
        match = DATA_BLOCK_RE.search(raw_text, search_position)
        if match is None:
            break
        if _is_inside_json_string(raw_text, match.start()):
            search_position = match.end()
            continue
        if re.search(r"/\s*>$", match.group(0)):
            search_position = match.end()
            continue
        kind = _tag_kind(match)
        close = _find_tag_close(raw_text, match.end(), kind)
        body = raw_text[match.end():close.start()] if close else raw_text[match.end():]
        if not _is_tag_candidate(raw_text, match, body, has_close=close is not None):
            search_position = match.end()
            continue
        if close and not _looks_like_json_payload(body):
            nested = DATA_BLOCK_RE.search(raw_text, match.end(), close.start())
            if nested is not None and not re.search(r"/\s*>$", nested.group(0)):
                search_position = match.end()
                continue

        if close is None:
            start = _fence_start(raw_text, match.start())
            removals.append((start, len(raw_text)))
            invalid_count += 1
            break

        start = _fence_start(raw_text, match.start())
        end = _fence_end(raw_text, close.end())
        raw_json = body.strip()
        payload = _repair_json_payload(raw_json)
        if payload is None or not isinstance(payload, dict):
            invalid_count += 1
        else:
            located_blocks.append(
                (
                    start,
                    _normalize_service_payload(
                        payload,
                        raw_json,
                        xml_syntax=kind == "xml",
                    ),
                )
            )
        removals.append((start, end))
        search_position = end

    visible_text = _mask_spans(raw_text, removals)
    self_closing_matches = [
        match
        for match in SELF_CLOSING_DATA_RE.finditer(visible_text)
        if not _is_inside_json_string(visible_text, match.start())
    ]
    if self_closing_matches:
        invalid_count += len(self_closing_matches)
        removals.extend((match.start(), match.end()) for match in self_closing_matches)
        visible_text = _mask_spans(raw_text, removals)

    search_position = 0
    while True:
        truncated = TRUNCATED_XML_OPEN_RE.search(visible_text, search_position)
        if truncated is None:
            break
        if _is_inside_json_string(visible_text, truncated.start()):
            search_position = truncated.end()
            continue
        is_attr_opener = bool(re.search(r"\s+[^>\r\n]*$", truncated.group(0)))
        is_candidate = (
            is_attr_opener or _is_line_start_tag(visible_text, truncated.start())
        ) and (
            _looks_like_json_payload(visible_text[truncated.end():])
            or not visible_text[truncated.end():].strip()
        )
        if is_candidate:
            start = _fence_start(visible_text, truncated.start())
            removals.append((start, len(raw_text)))
            invalid_count += 1
            visible_text = _mask_spans(raw_text, removals)
            break
        search_position = truncated.end()

    return visible_text, located_blocks, invalid_count, removals


def _extract_fallback_data(
    visible_text: str,
) -> tuple[str, list[tuple[int, ServiceDataBlock]], int, bool, list[tuple[int, int]]]:
    spans = _find_balanced_json_spans(visible_text)
    removals: list[tuple[int, int]] = []
    located_blocks: list[tuple[int, ServiceDataBlock]] = []
    invalid_count = 0
    excluded_signature_starts: set[int] = set()
    for object_start, object_end in spans:
        if _has_unclosed_json_container(visible_text, object_start):
            continue
        start, fenced = _fallback_removal_start(visible_text, object_start)
        explicit_signature = _has_explicit_service_signature(visible_text, object_start)
        tail_is_allowed = _is_fallback_tail(visible_text, object_end, fenced=fenced)
        if not tail_is_allowed and not explicit_signature:
            continue
        candidate = visible_text[object_start:object_end]
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            if explicit_signature:
                end = _fence_end(visible_text, object_end) if fenced else object_end
                removals.append((start, end))
                excluded_signature_starts.add(start)
                invalid_count += 1
            continue
        if not isinstance(payload, dict) or not _is_service_fallback_payload(payload, explicit_signature):
            continue
        end = _fence_end(visible_text, object_end) if fenced else object_end
        located_blocks.append(
            (
                start,
                _normalize_service_payload(payload, candidate, xml_syntax=True),
            )
        )
        removals.append((start, end))
        excluded_signature_starts.add(start)

    tail = _find_fallback_service_tail(visible_text, excluded_signature_starts)
    if tail:
        removals.append(tail)
        invalid_count += 1

    if removals:
        return visible_text, located_blocks, invalid_count, True, removals
    return visible_text, located_blocks, 0, False, removals


def extract_service_data_details(text: str | None) -> ServiceDataExtraction:
    raw_text = text or ""
    masked_text, tagged_blocks, invalid_count, tagged_removals = _extract_tagged_data(raw_text)
    classification = "none"
    if tagged_blocks or invalid_count:
        classification = "tags" if tagged_blocks and not invalid_count else "parse_error"

    _, fallback_blocks, fallback_invalid, used_fallback, fallback_removals = _extract_fallback_data(masked_text)
    if used_fallback:
        invalid_count += fallback_invalid
        classification = "fallback" if not invalid_count else "parse_error"

    located_blocks = sorted(
        tagged_blocks + fallback_blocks,
        key=lambda item: item[0],
    )
    visible_text = _remove_spans(raw_text, tagged_removals + fallback_removals)

    return ServiceDataExtraction(
        visible_text=_normalize_visible_text(visible_text),
        blocks=[block for _, block in located_blocks],
        invalid_count=invalid_count,
        classification=classification,
    )


def service_data_diagnostics(extraction: ServiceDataExtraction) -> dict[str, Any]:
    return {
        "data_extraction": extraction.classification,
        "data_block_count": len(extraction.blocks),
        "invalid_data_blocks": extraction.invalid_count,
    }


def extract_service_data(text: str | None) -> tuple[str, list[ServiceDataBlock], int]:
    extraction = extract_service_data_details(text)
    return extraction.visible_text, extraction.blocks, extraction.invalid_count


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


def service_data_history_block(block: ServiceDataBlock) -> dict[str, Any] | None:
    """Build one backward-compatible user history entry for a service block."""
    if block.legacy:
        if not block.metadata:
            return None
        data = block.metadata
    else:
        data = {
            "current_state": block.current_state,
            "events": block.events,
            "metadata": block.metadata,
            "save_mode": block.save_mode,
        }
    return {"data": data, "raw_json": block.raw_json}


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
