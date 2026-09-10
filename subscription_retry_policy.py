from __future__ import annotations

from datetime import datetime, timedelta


MAX_PAYMENT_ATTEMPTS = 3
FIRST_RETRY_DELAY = timedelta(hours=2)
SECOND_RETRY_DELAY = timedelta(hours=24)


def get_next_retry_at(
    payment_attempt_count: int,
    last_payment_attempt: datetime | None,
    retry_not_before: datetime | None = None,
) -> datetime | None:
    if retry_not_before is not None:
        standard_next = None
        if payment_attempt_count == 1 and last_payment_attempt is not None:
            standard_next = last_payment_attempt + FIRST_RETRY_DELAY
        elif payment_attempt_count == 2 and last_payment_attempt is not None:
            standard_next = last_payment_attempt + SECOND_RETRY_DELAY
        if standard_next is not None:
            return max(standard_next, retry_not_before)
        return retry_not_before

    if payment_attempt_count <= 0 or last_payment_attempt is None:
        return None
    if payment_attempt_count == 1:
        return last_payment_attempt + FIRST_RETRY_DELAY
    if payment_attempt_count == 2:
        return last_payment_attempt + SECOND_RETRY_DELAY
    return None


def can_retry_now(
    payment_attempt_count: int,
    last_payment_attempt: datetime | None,
    now: datetime,
    retry_not_before: datetime | None = None,
) -> bool:
    if retry_not_before is not None and now < retry_not_before:
        return False
    if payment_attempt_count <= 0:
        if last_payment_attempt is None:
            return True
        return now >= last_payment_attempt + FIRST_RETRY_DELAY
    next_retry_at = get_next_retry_at(payment_attempt_count, last_payment_attempt, retry_not_before=retry_not_before)
    if next_retry_at is None:
        return False
    return now >= next_retry_at


def can_retry_manually(payment_attempt_count: int) -> bool:
    return payment_attempt_count < MAX_PAYMENT_ATTEMPTS
