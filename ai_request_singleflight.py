"""Per-user single-flight request admission gate.

Enforces that at most one conversational AI turn is active per user at any time.
Uses an opaque generation lease token to prevent stale releases.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import uuid


AI_BUSY_MESSAGE = (
    "Пожалуйста, не так быстро — я ещё разбираю твоё предыдущее сообщение. "
    "Дай мне немного времени, чтобы всё хорошенько обдумать."
)


@dataclass(frozen=True)
class SingleFlightLease:
    """Opaque ownership token representing an active single-flight admission lease."""
    platform: str
    user_id: str
    token: str


class UserAISingleFlight:
    """Process-local thread-safe single-flight admission tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_leases: dict[tuple[str, str], SingleFlightLease] = {}

    def _normalize_key(self, platform: str, user_id: int | str) -> tuple[str, str]:
        return (str(platform).lower().strip(), str(user_id).strip())

    def try_claim(self, platform: str, user_id: int | str) -> SingleFlightLease | None:
        """Atomically attempt to claim a single-flight slot.

        Returns a SingleFlightLease if admission is granted, or None if busy.
        """
        key = self._normalize_key(platform, user_id)
        with self._lock:
            if key in self._active_leases:
                return None
            lease = SingleFlightLease(
                platform=key[0],
                user_id=key[1],
                token=uuid.uuid4().hex,
            )
            self._active_leases[key] = lease
            return lease

    def release(self, lease: SingleFlightLease | None) -> bool:
        """Release an active slot.

        Releases ONLY if the active entry matches this exact lease token.
        Stale or duplicate releases are safe no-ops returning False.
        """
        if lease is None or not isinstance(lease, SingleFlightLease):
            return False
        key = (lease.platform, lease.user_id)
        with self._lock:
            current = self._active_leases.get(key)
            if current is not None and current.token == lease.token:
                del self._active_leases[key]
                return True
            return False

    def is_busy(self, platform: str, user_id: int | str) -> bool:
        """Check whether a single-flight slot is currently claimed."""
        key = self._normalize_key(platform, user_id)
        with self._lock:
            return key in self._active_leases

    def clear(self) -> None:
        """Reset all active leases. Intended for test isolation."""
        with self._lock:
            self._active_leases.clear()


single_flight = UserAISingleFlight()
