#!/usr/bin/env python3
"""Deterministic test suite partitioning for CI runners.

Partitions test files across independent parallel runner jobs without
cross-process state corruption or concurrency races.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def get_shards(total_shards: int = 3, tests_dir: str = "tests") -> list[list[str]]:
    if total_shards <= 0:
        raise ValueError(f"total_shards must be positive, got {total_shards}")

    pattern = os.path.join(tests_dir, "test_*.py")
    test_files = sorted(glob.glob(pattern))
    if not test_files:
        raise RuntimeError(f"No test files found matching {pattern}")

    weights = {f: os.path.getsize(f) for f in test_files}
    # Sort deterministically by weight descending, then filename ascending
    sorted_files = sorted(test_files, key=lambda f: (-weights[f], f))

    shards: list[list[str]] = [[] for _ in range(total_shards)]
    shard_weights: list[int] = [0] * total_shards

    for f in sorted_files:
        lightest_idx = min(range(total_shards), key=lambda i: shard_weights[i])
        shards[lightest_idx].append(f)
        shard_weights[lightest_idx] += weights[f]

    # Sort each shard deterministically
    for shard in shards:
        shard.sort()

    # Invariant assertion: every file is included exactly once
    flattened = [f for shard in shards for f in shard]
    if len(flattened) != len(test_files) or set(flattened) != set(test_files):
        raise RuntimeError(
            f"Shard partition invariant failed: {len(flattened)} partitioned vs {len(test_files)} discovered"
        )

    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic test sharding for CI")
    parser.add_argument("--total-shards", type=int, default=3, help="Total number of parallel shards (default: 3)")
    parser.add_argument("--shard", type=int, default=None, help="1-indexed shard number to run or print")
    parser.add_argument("--print-files", action="store_true", help="Print space-separated files for the selected shard")
    parser.add_argument("--run", action="store_true", help="Execute pytest for the selected shard")
    parser.add_argument("--verify", action="store_true", help="Verify partitioning invariants and print stats")
    parser.add_argument("pytest_args", nargs="*", help="Extra arguments passed to pytest when using --run")

    args = parser.parse_args()

    shards = get_shards(total_shards=args.total_shards)

    if args.verify:
        print(f"Partitioned {sum(len(s) for s in shards)} test files across {args.total_shards} shards:")
        for idx, shard in enumerate(shards, start=1):
            total_bytes = sum(os.path.getsize(f) for f in shard)
            print(f"  Shard {idx}/{args.total_shards}: {len(shard)} files ({total_bytes:,} bytes)")
        return 0

    if args.shard is None:
        parser.error("--shard is required unless using --verify")

    if not (1 <= args.shard <= args.total_shards):
        parser.error(f"--shard must be between 1 and {args.total_shards}, got {args.shard}")

    selected_shard = shards[args.shard - 1]

    if args.print_files:
        print(" ".join(selected_shard))
        return 0

    if args.run:
        cmd = [sys.executable, "-m", "pytest", "-q", "-rs"] + selected_shard + args.pytest_args
        print(f"Running CI test shard {args.shard}/{args.total_shards} ({len(selected_shard)} files)...")
        res = subprocess.run(cmd)
        return res.returncode

    # Default to printing files
    print(" ".join(selected_shard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
