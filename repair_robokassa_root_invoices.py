#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


MANUAL_PAYMENT_MARKER = "| ОПЛАТА | Robokassa |"
USER_ID_RE = re.compile(r"(?:\[id=(?:<code>)?(?P<bracket>\d+)(?:</code>)?\]|\(ID:\s*(?P<paren>\d+)\))")
INV_ID_RE = re.compile(r"InvId=(\d+)")


@dataclass(slots=True)
class SubscriptionRow:
    user_id: int
    payment_method_id: str | None
    first_name: str | None
    username: str | None
    auto_renewal: bool | None
    payment_attempt_count: int

    @property
    def user_label(self) -> str:
        label = self.first_name or ""
        if self.username:
            label = f"{label} (@{self.username})".strip()
        return label or str(self.user_id)


@dataclass(slots=True)
class RepairCandidate:
    user_id: int
    user_label: str
    current_payment_method_id: str | None
    root_invoice_id: str
    auto_renewal: bool | None
    payment_attempt_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repairs Robokassa root invoice ids in user_subscriptions.payment_method_id "
            "by taking the latest manual 'ОПЛАТА | Robokassa' from payment logs."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy database URL. Defaults to DATABASE_URL from environment.",
    )
    parser.add_argument(
        "--app-port",
        default=os.environ.get("APP_PORT", "8080"),
        help="Bot APP_PORT used to discover payment_events_<port>.log files.",
    )
    parser.add_argument(
        "--log-path",
        action="append",
        default=[],
        help="Explicit payment log path. Can be passed multiple times. If omitted, default paths are used.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        action="append",
        default=[],
        help="Restrict repair to specific Telegram user ids. Can be passed multiple times.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates to the database. Without this flag, the script only prints the planned changes.",
    )
    return parser.parse_args()


def discover_log_paths(app_port: str, explicit_paths: Iterable[str]) -> list[Path]:
    if explicit_paths:
        paths = [Path(path).expanduser() for path in explicit_paths]
    else:
        patterns = [
            Path("logs").glob(f"payment_events_{app_port}.log*"),
            Path(".").glob(f"payment_events_{app_port}.log*"),
        ]
        paths = []
        for pattern in patterns:
            paths.extend(pattern)

    unique_paths: dict[Path, None] = {}
    for path in paths:
        if path.exists():
            unique_paths[path.resolve()] = None

    resolved_paths = list(unique_paths.keys())
    if not resolved_paths:
        return []
    return sorted(resolved_paths, key=_log_sort_key)


def _log_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"\.log(?:\.(\d+))?$", path.name)
    if not match:
        return (0, 0, path.name)
    suffix = match.group(1)
    if suffix is None:
        return (1, 0, path.name)
    return (0, -int(suffix), path.name)


def extract_manual_payment_roots(log_paths: Iterable[Path]) -> dict[int, str]:
    latest_root_by_user: dict[int, str] = {}

    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                if MANUAL_PAYMENT_MARKER not in raw_line:
                    continue

                user_match = USER_ID_RE.search(raw_line)
                inv_match = INV_ID_RE.search(raw_line)
                if not user_match or not inv_match:
                    continue

                user_id = user_match.group("bracket") or user_match.group("paren")
                latest_root_by_user[int(user_id)] = inv_match.group(1)

    return latest_root_by_user


async def load_current_robokassa_subscriptions(database_url: str, user_ids: set[int]) -> list[SubscriptionRow]:
    engine = create_async_engine(database_url, echo=False)
    try:
        query = text(
            """
            SELECT
                us.user_id,
                us.payment_method_id,
                us.auto_renewal,
                us.payment_attempt_count,
                u.first_name,
                u.username
            FROM user_subscriptions us
            LEFT JOIN users u ON u.id = us.user_id
            WHERE us.payment_provider = 'Robokassa'
              AND us.payment_method_id IS NOT NULL
            ORDER BY us.user_id
            """
        )
        async with engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
    finally:
        await engine.dispose()

    subscriptions = [
        SubscriptionRow(
            user_id=row["user_id"],
            payment_method_id=row["payment_method_id"],
            first_name=row["first_name"],
            username=row["username"],
            auto_renewal=row["auto_renewal"],
            payment_attempt_count=row["payment_attempt_count"],
        )
        for row in rows
    ]
    if not user_ids:
        return subscriptions
    return [row for row in subscriptions if row.user_id in user_ids]


def build_candidates(
    subscriptions: Iterable[SubscriptionRow],
    root_invoice_by_user: dict[int, str],
) -> tuple[list[RepairCandidate], list[SubscriptionRow]]:
    candidates: list[RepairCandidate] = []
    missing_roots: list[SubscriptionRow] = []

    for sub in subscriptions:
        root_invoice_id = root_invoice_by_user.get(sub.user_id)
        if not root_invoice_id:
            missing_roots.append(sub)
            continue
        if str(sub.payment_method_id) == root_invoice_id:
            continue
        candidates.append(
            RepairCandidate(
                user_id=sub.user_id,
                user_label=sub.user_label,
                current_payment_method_id=sub.payment_method_id,
                root_invoice_id=root_invoice_id,
                auto_renewal=sub.auto_renewal,
                payment_attempt_count=sub.payment_attempt_count,
            )
        )
    return candidates, missing_roots


async def apply_candidates(database_url: str, candidates: list[RepairCandidate]) -> None:
    if not candidates:
        return

    engine = create_async_engine(database_url, echo=False)
    try:
        async with engine.begin() as conn:
            for candidate in candidates:
                await conn.execute(
                    text(
                        """
                        UPDATE user_subscriptions
                        SET payment_method_id = :root_invoice_id
                        WHERE user_id = :user_id
                          AND payment_provider = 'Robokassa'
                        """
                    ),
                    {
                        "root_invoice_id": candidate.root_invoice_id,
                        "user_id": candidate.user_id,
                    },
                )
    finally:
        await engine.dispose()


def print_report(
    log_paths: list[Path],
    candidates: list[RepairCandidate],
    missing_roots: list[SubscriptionRow],
    apply: bool,
) -> None:
    print("Robokassa root invoice repair")
    print(f"Logs scanned: {', '.join(str(path) for path in log_paths)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Missing roots: {len(missing_roots)}")
    print()

    for candidate in candidates:
        print(
            f"user_id={candidate.user_id} | {candidate.user_label} | "
            f"payment_method_id {candidate.current_payment_method_id} -> {candidate.root_invoice_id} | "
            f"auto_renewal={candidate.auto_renewal} | attempts={candidate.payment_attempt_count}"
        )
        print(
            "UPDATE user_subscriptions "
            f"SET payment_method_id = '{candidate.root_invoice_id}' "
            f"WHERE user_id = {candidate.user_id} AND payment_provider = 'Robokassa';"
        )
        print()

    if missing_roots:
        print("Users without a root manual Robokassa payment found in scanned logs:")
        for row in missing_roots:
            print(
                f"user_id={row.user_id} | {row.user_label} | "
                f"current payment_method_id={row.payment_method_id}"
            )
        print()

    if apply:
        print("Updates applied.")
    else:
        print("Dry run only. Re-run with --apply to write changes.")


async def main() -> int:
    load_dotenv()
    args = parse_args()

    if not args.database_url:
        print("DATABASE_URL is required. Pass --database-url or export DATABASE_URL.")
        return 1

    user_ids = set(args.user_id)
    log_paths = discover_log_paths(str(args.app_port), args.log_path)
    if not log_paths:
        print("No payment logs found.")
        return 1

    root_invoice_by_user = extract_manual_payment_roots(log_paths)
    subscriptions = await load_current_robokassa_subscriptions(args.database_url, user_ids)
    candidates, missing_roots = build_candidates(subscriptions, root_invoice_by_user)

    if args.apply and candidates:
        await apply_candidates(args.database_url, candidates)

    print_report(log_paths, candidates, missing_roots, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
