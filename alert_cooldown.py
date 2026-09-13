from datetime import datetime, timedelta


class AlertCooldown:
    def __init__(self, duration: timedelta):
        self.duration = duration
        self.last_sent_at: datetime | None = None

    def should_send(self, now: datetime | None = None) -> bool:
        current_time = now or datetime.utcnow()
        if self.last_sent_at is not None and current_time - self.last_sent_at < self.duration:
            return False
        self.last_sent_at = current_time
        return True


class KeyedAlertCooldown:
    """Keyed cooldown tracker to prevent alert storms across distinct incidents."""

    def __init__(self, duration: timedelta, max_entries: int = 2000):
        self.duration = duration
        self.max_entries = max_entries
        self._last_sent: dict[str, datetime] = {}

    def should_send(self, key: str, now: datetime | None = None) -> bool:
        current_time = now or datetime.utcnow()
        self._prune_if_needed(current_time)
        last_time = self._last_sent.get(key)
        if last_time is not None and current_time - last_time < self.duration:
            return False
        self._last_sent[key] = current_time
        return True

    def _prune_if_needed(self, current_time: datetime) -> None:
        if len(self._last_sent) > self.max_entries:
            expired = [k for k, v in self._last_sent.items() if current_time - v >= self.duration]
            for k in expired:
                del self._last_sent[k]

