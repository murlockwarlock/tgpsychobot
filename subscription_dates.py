from __future__ import annotations

from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta


def extend_subscription_end_date(
    current_end_date: datetime | None,
    paid_at: datetime,
    duration_value: int,
    duration_unit: str,
) -> datetime:
    base_end_date = current_end_date if current_end_date and current_end_date > paid_at else paid_at
    if duration_unit == "months":
        return base_end_date + relativedelta(months=duration_value)
    return base_end_date + timedelta(days=duration_value)
