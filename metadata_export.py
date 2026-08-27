"""Pure helpers for regular and anonymized metadata exports."""

from __future__ import annotations

from max_messenger_bot.identity import is_max_user_id, max_communication_name, raw_max_user_id


def metadata_export_entry(user, metadata: list[dict], *, anonymize: bool, anonymous_index: int = 1) -> dict:
    if anonymize:
        user_info = {"label": f"user_{anonymous_index}"}
    elif is_max_user_id(user.id):
        display_id = raw_max_user_id(user.id)
        user_info = {
            "label": str(display_id),
            "id": display_id,
            "name": max_communication_name(user),
            "username": user.username,
        }
    else:
        user_info = {
            "label": str(user.id),
            "id": user.id,
            "name": user.name or user.first_name,
            "username": user.username,
        }
    return {"user_info": user_info, "metadata": metadata}
