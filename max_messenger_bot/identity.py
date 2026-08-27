from __future__ import annotations

from typing import Any

from .models import MAX_ID_OFFSET


def is_max_user_id(user_id: int | None) -> bool:
    return user_id is not None and int(user_id) >= MAX_ID_OFFSET


def raw_max_user_id(user_id: int) -> int:
    return int(user_id) - MAX_ID_OFFSET


def max_public_name(user: Any) -> str:
    return str(getattr(user, "first_name", None) or "").strip()


def max_communication_name(user: Any) -> str:
    return str(getattr(user, "name", None) or "").strip() or max_public_name(user) or str(raw_max_user_id(user.id))


def max_username(user: Any) -> str:
    username = str(getattr(user, "username", None) or "").strip()
    return f"@{username}" if username and not username.isdigit() else "не указан"


def max_client_list_label(user: Any) -> str:
    communication_name = max_communication_name(user)
    public_name = max_public_name(user)
    label = communication_name
    if public_name and public_name != communication_name:
        label = f"{label} ({public_name})"
    username = max_username(user)
    if username != "не указан":
        label = f"{label} ({username})"
    return label
