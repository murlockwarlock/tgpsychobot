"""Profile fields collected before the first visible bot content."""

from __future__ import annotations


PROFILE_FIELDS = ("name", "gender", "age")


def configured_profile_fields(config) -> list[str]:
    defaults = {"name": True, "gender": True, "age": False}
    return [
        field
        for field in PROFILE_FIELDS
        if bool(getattr(config, f"profile_collect_{field}", defaults[field]))
    ]


def missing_profile_fields(config, user) -> list[str]:
    return [field for field in configured_profile_fields(config) if not getattr(user, field, None)]
