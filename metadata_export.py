"""Pure helpers for regular and anonymized metadata exports."""

from __future__ import annotations


def metadata_export_entry(user, metadata: list[dict], *, anonymize: bool, anonymous_index: int = 1) -> dict:
    if anonymize:
        user_info = {"label": f"user_{anonymous_index}"}
    else:
        user_info = {
            "label": str(user.id),
            "id": user.id,
            "name": user.name or user.first_name,
            "username": user.username,
        }
    return {"user_info": user_info, "metadata": metadata}
