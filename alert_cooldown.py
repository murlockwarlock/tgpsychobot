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
