from __future__ import annotations


def extract_birthdate_parts(birthdate) -> tuple[int | None, int | None, int | None]:
    if not birthdate:
        return None, None, None
    day = getattr(birthdate, "day", None)
    month = getattr(birthdate, "month", None)
    year = getattr(birthdate, "year", None)
    return day, month, year


def has_birthdate(day: int | None, month: int | None) -> bool:
    return bool(day and month)


def format_birthdate(day: int | None, month: int | None, year: int | None) -> str | None:
    if not has_birthdate(day, month):
        return None
    if year:
        return f"{day:02d}.{month:02d}.{year}"
    return f"{day:02d}.{month:02d}"
