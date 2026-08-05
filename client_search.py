from __future__ import annotations


def normalize_client_search_query(value: str) -> str:
    return value.strip().lstrip("@").strip()
