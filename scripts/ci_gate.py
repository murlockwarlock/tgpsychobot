#!/usr/bin/env python3
"""Fail-closed evaluation for GitHub Actions CI Gate.

Rules:
1. FULL CI mode:
   - fast-ci must be 'skipped'
   - sqlite must be 'success'
   - postgres-event-chains must be 'success'
   Any 'skipped', 'failure', or 'cancelled' in sqlite or postgres is a hard failure.

2. FAST CI mode:
   - fast-ci must be 'success'
   - sqlite must be 'skipped'
   - postgres-event-chains must be 'skipped'
   Any 'skipped', 'failure', or 'cancelled' in fast-ci is a hard failure.

3. All other states are fail-closed (hard failure).
"""

from __future__ import annotations

import sys


def evaluate_ci_gate(fast_status: str, sqlite_status: str, pg_status: str) -> tuple[bool, str]:
    fast = (fast_status or "").strip().lower()
    sqlite = (sqlite_status or "").strip().lower()
    pg = (pg_status or "").strip().lower()

    # Fast CI mode
    if fast == "success" and sqlite == "skipped" and pg == "skipped":
        return True, "Fast CI completed successfully (draft mode, sqlite and postgres-event-chains skipped as expected)."

    # Full CI mode
    if fast == "skipped" and sqlite == "success" and pg == "success":
        return True, "Full CI completed successfully (sqlite shard suite and postgres-event-chains succeeded)."

    # Fail-closed error diagnostics
    reasons: list[str] = []
    if fast == "skipped":
        # Evaluated as Full CI attempt
        if sqlite != "success":
            reasons.append(f"Full CI requires sqlite=success, but got '{sqlite}'")
        if pg != "success":
            reasons.append(f"Full CI requires postgres-event-chains=success, but got '{pg}' (skipped is not allowed)")
    elif fast == "success":
        # Evaluated as Fast CI attempt
        if sqlite != "skipped":
            reasons.append(f"Fast CI requires sqlite=skipped, but got '{sqlite}'")
        if pg != "skipped":
            reasons.append(f"Fast CI requires postgres-event-chains=skipped, but got '{pg}'")
    else:
        reasons.append(f"Fast CI failed or was skipped unexpectedly (status='{fast}')")
        if sqlite != "success":
            reasons.append(f"sqlite status='{sqlite}'")
        if pg != "success":
            reasons.append(f"postgres-event-chains status='{pg}'")

    msg = "CI Gate FAILED: " + "; ".join(reasons)
    return False, msg


def main() -> int:
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <fast_status> <sqlite_status> <pg_status>")
        return 1

    fast_status = sys.argv[1]
    sqlite_status = sys.argv[2]
    pg_status = sys.argv[3]

    print("========================================")
    print("Evaluating CI Gate Results (Fail-Closed)")
    print(f"  • Fast CI:                 {fast_status}")
    print(f"  • SQLite shard suite:      {sqlite_status}")
    print(f"  • PostgreSQL event chains: {pg_status}")
    print("========================================")

    ok, message = evaluate_ci_gate(fast_status, sqlite_status, pg_status)
    if ok:
        print(f"✓ {message}")
        return 0
    else:
        print(f"✗ {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
