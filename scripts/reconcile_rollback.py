#!/usr/bin/env python3
"""
Safe Rollback Reconciliation Utility.

Compares a target runtime directory with a pre-deploy backup archive,
identifies orphan files introduced by an aborted deployment, and safely
removes only those orphan files while strictly preserving:
  - Runtime environment files (runtime.env, .env*)
  - Local database files (*.db, *.sqlite, *.sqlite3)
  - Vector stores and caches (chroma_db_*)
  - Media assets (media/*)
  - Log files (logs/*)
  - Revision pointers (REVISION)

Correctly normalizes archive member names and disk paths to prevent
erroneous deletion caused by leading './' or '/' prefixes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tarfile
from typing import Collection, Iterable

PROTECTED_EXACT_NAMES = frozenset(
    {
        "runtime.env",
        ".env",
        ".env.deploy",
        "REVISION",
        "PREV_SHA.txt",
        "manifest_pre86.txt",
        "manifest_pr86_staged.txt",
    }
)

PROTECTED_PREFIXES = (
    "logs/",
    "media/",
    "chroma_db_",
    ".git/",
    ".pytest_cache/",
)

PROTECTED_EXTENSIONS = (
    ".db",
    ".sqlite",
    ".sqlite3",
)


def normalize_relpath(path: str) -> str:
    """Normalize a path to a clean forward-slash relative path without leading ./ or /."""
    if not path:
        return ""
    norm = os.path.normpath(path).replace("\\", "/")
    parts = [p for p in norm.split("/") if p and p != "."]
    return "/".join(parts)


def is_protected(rel_path: str) -> bool:
    """Determine whether a relative path must be preserved during rollback."""
    norm = normalize_relpath(rel_path)
    if not norm:
        return True
    basename = os.path.basename(norm)
    if basename in PROTECTED_EXACT_NAMES or norm in PROTECTED_EXACT_NAMES:
        return True
    if basename.startswith(".env"):
        return True
    if any(norm.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return True
    if any(norm.endswith(ext) for ext in PROTECTED_EXTENSIONS):
        return True
    return False


def get_archive_members(archive_path: str | Path) -> set[str]:
    """Extract normalized member paths from a tar archive."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    members: set[str] = set()
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            norm = normalize_relpath(member.name)
            if norm:
                members.add(norm)
    return members


def find_orphan_files(
    target_dir: str | Path,
    archive_members: Collection[str],
) -> list[Path]:
    """Find files on disk in target_dir that are neither in archive_members nor protected."""
    target_dir = Path(target_dir).resolve()
    orphans: list[Path] = []

    for root, dirs, files in os.walk(target_dir):
        for fname in files:
            full_path = Path(root) / fname
            try:
                rel_path = normalize_relpath(str(full_path.relative_to(target_dir)))
            except ValueError:
                continue

            if is_protected(rel_path):
                continue

            if rel_path not in archive_members:
                orphans.append(full_path)

    return sorted(orphans)


def reconcile_rollback_directory(
    archive_path: str | Path,
    target_dir: str | Path,
    dry_run: bool = False,
) -> list[str]:
    """Reconcile target_dir against archive_path and remove orphan files."""
    target_path = Path(target_dir).resolve()
    members = get_archive_members(archive_path)
    orphans = find_orphan_files(target_path, members)

    removed: list[str] = []
    for orphan in orphans:
        rel = normalize_relpath(str(orphan.relative_to(target_path)))
        if not dry_run:
            try:
                orphan.unlink()
                removed.append(rel)
            except OSError as err:
                print(f"[!] Warning: failed to remove {orphan}: {err}", file=sys.stderr)
        else:
            removed.append(rel)

    # Clean up any leftover empty directories (bottom-up)
    if not dry_run:
        for root, dirs, _ in os.walk(target_path, topdown=False):
            for dname in dirs:
                dir_path = Path(root) / dname
                try:
                    rel_dir = normalize_relpath(str(dir_path.relative_to(target_path)))
                    if is_protected(rel_dir + "/"):
                        continue
                    if rel_dir not in members and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                except (OSError, ValueError):
                    pass

    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely reconcile runtime directory against pre-deploy backup archive."
    )
    parser.add_argument(
        "--archive",
        required=True,
        help="Path to the pre-deploy tar.gz backup archive.",
    )
    parser.add_argument(
        "--target-dir",
        default=".",
        help="Target directory to reconcile (defaults to current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only display files that would be removed without deleting.",
    )

    args = parser.parse_args(argv)

    try:
        removed = reconcile_rollback_directory(
            archive_path=args.archive,
            target_dir=args.target_dir,
            dry_run=args.dry_run,
        )
        mode = "Would remove" if args.dry_run else "Removed"
        print(f"[*] {mode} {len(removed)} orphan file(s):")
        for f in removed:
            print(f"    - {f}")
        return 0
    except Exception as err:
        print(f"[!] Reconciliation error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
