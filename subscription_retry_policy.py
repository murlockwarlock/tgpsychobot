from __future__ import annotations

from datetime import datetime, timedelta


MAX_PAYMENT_ATTEMPTS = 3
FIRST_RETRY_DELAY = timedelta(hours=2)
SECOND_RETRY_DELAY = timedelta(hours=24)


def get_next_retry_at(payment_attempt_count: int, last_payment_attempt: datetime | None) -> datetime | None:
    if payment_attempt_count <= 0 or last_payment_attempt is None:
        return None
    if payment_attempt_count == 1:
        return last_payment_attempt + FIRST_RETRY_DELAY
    if payment_attempt_count == 2:
        return last_payment_attempt + SECOND_RETRY_DELAY
    return None


def can_retry_now(payment_attempt_count: int, last_payment_attempt: datetime | None, now: datetime) -> bool:
    if payment_attempt_count <= 0:
        if last_payment_attempt is None:
            return True
        return now >= last_payment_attempt + FIRST_RETRY_DELAY
    next_retry_at = get_next_retry_at(payment_attempt_count, last_payment_attempt)
    if next_retry_at is None:
        return False
    return now >= next_retry_at


def can_retry_manually(payment_attempt_count: int) -> bool:
    return payment_attempt_count < MAX_PAYMENT_ATTEMPTS
