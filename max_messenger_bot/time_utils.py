from __future__ import annotations

from datetime import datetime, timedelta, timezone


UTC = timezone.utc
MSK = timezone(timedelta(hours=3))


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def to_msk(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(MSK)


def format_msk(dt: datetime, fmt: str = "%d.%m.%Y %H:%M") -> str:
    return to_msk(dt).strftime(fmt)
