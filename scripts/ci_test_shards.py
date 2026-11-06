#!/usr/bin/env python3
"""Deterministic test suite partitioning and verification for CI runners.

Partitions test files recursively across independent parallel runner jobs without
cross-process state corruption or concurrency races.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path


def discover_test_files(tests_dir: str = "tests") -> list[str]:
    """Recursively discover all pytest test files in tests_dir and nested directories."""
    base = Path(tests_dir)
    if not base.exists():
        return []

    discovered: list[str] = []
    for p in base.rglob("test_*.py"):
        if not p.is_file():
            continue
        rel = p.relative_to(base) if p.is_absolute() else p
        if any(part.startswith((".", "__pycache__")) for part in rel.parts):
            continue
        discovered.append(p.as_posix())

    return sorted(discovered)


def get_shards(total_shards: int = 3, tests_dir: str = "tests") -> list[list[str]]:
    if total_shards <= 0:
        raise ValueError(f"total_shards must be positive, got {total_shards}")

    test_files = discover_test_files(tests_dir)
    if not test_files:
        return [[] for _ in range(total_shards)]

    weights = {f: os.path.getsize(f) for f in test_files}
    sorted_files = sorted(test_files, key=lambda f: (-weights[f], f))

    shards: list[list[str]] = [[] for _ in range(total_shards)]
    shard_weights: list[int] = [0] * total_shards

    for f in sorted_files:
        lightest_idx = min(range(total_shards), key=lambda i: shard_weights[i])
        shards[lightest_idx].append(f)
        shard_weights[lightest_idx] += weights[f]

    for shard in shards:
        shard.sort()

    flattened = [f for shard in shards for f in shard]
    if len(flattened) != len(test_files) or set(flattened) != set(test_files):
        raise RuntimeError(
            f"Shard partition invariant failed: {len(flattened)} partitioned vs {len(test_files)} discovered"
        )

    return shards


def verify_collection_coverage(
    shards: list[list[str]],
    tests_dir: str = "tests",
    python_exe: str | None = None,
) -> tuple[bool, str]:
    """Verify that every test file collected by pytest is partitioned into the shards.

    Fails if pytest collection discovers any test files not accounted for in shards.
    """
    cmd = [python_exe or sys.executable, "-m", "pytest", tests_dir, "--collect-only", "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 5):  # 5 means no tests collected
        return False, f"pytest collection failed with returncode {proc.returncode}:\n{proc.stderr}"

    collected_files: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line:
            raw_path = line.split("::", 1)[0].strip()
            collected_files.add(Path(raw_path).as_posix())

    all_sharded = {Path(f).as_posix() for shard in shards for f in shard}

    uncovered = collected_files - all_sharded
    if uncovered:
        return False, f"Pytest collection discrepancy! Test files collected by pytest but missing in shards: {sorted(uncovered)}"

    return True, f"All {len(collected_files)} test files from pytest collection are partitioned across shards."


def audit_postgres_tests(tests_dir: str = "tests") -> tuple[bool, str, list[str]]:
    """Verify that all tests interacting with real PostgreSQL are decorated with @pytest.mark.postgres_integration.

    Any test referencing real PostgreSQL or POSTGRES_TEST_DATABASE_URL without the marker
    causes a verification failure.
    """
    base = Path(tests_dir)
    if not base.exists():
        return True, "Tests directory does not exist", []

    registered_tests: list[str] = []
    unregistered_tests: list[str] = []

    for test_file in sorted(base.rglob("test_*.py")):
        if not test_file.is_file() or test_file.name == "test_ci_optimization.py":
            continue
        try:
            content = test_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(test_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("test_"):
                    continue

                has_marker = any(
                    (isinstance(dec, ast.Attribute) and dec.attr == "postgres_integration")
                    or (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "postgres_integration")
                    for dec in node.decorator_list
                )

                node_code = ast.get_source_segment(content, node) or ""
                is_pg_test = (
                    "real_postgresql" in node.name.lower()
                    or "POSTGRES_TEST_DATABASE_URL" in node_code
                    or "real_postgresql" in node_code
                )

                test_ref = f"{test_file.as_posix()}::{node.name}"
                if has_marker:
                    registered_tests.append(test_ref)
                elif is_pg_test:
                    unregistered_tests.append(test_ref)

    if unregistered_tests:
        return (
            False,
            f"Unregistered PostgreSQL tests detected ({len(unregistered_tests)}): {unregistered_tests}. "
            f"All tests targeting real PostgreSQL must be decorated with @pytest.mark.postgres_integration.",
            registered_tests,
        )

    if len(registered_tests) < 9:
        return (
            False,
            f"Expected at least 9 registered PostgreSQL tests, found {len(registered_tests)}",
            registered_tests,
        )

    return (
        True,
        f"All {len(registered_tests)} PostgreSQL tests are properly registered with @pytest.mark.postgres_integration.",
        registered_tests,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic test sharding for CI")
    parser.add_argument("--total-shards", type=int, default=3, help="Total number of parallel shards (default: 3)")
    parser.add_argument("--shard", type=int, default=None, help="1-indexed shard number to run or print")
    parser.add_argument("--print-files", action="store_true", help="Print space-separated files for the selected shard")
    parser.add_argument("--run", action="store_true", help="Execute pytest for the selected shard")
    parser.add_argument("--verify", action="store_true", help="Verify partitioning invariants and print stats")
    parser.add_argument("--verify-collection", action="store_true", help="Verify pytest collection coverage across shards")
    parser.add_argument("--audit-postgres", action="store_true", help="Audit PostgreSQL test registration")
    parser.add_argument("pytest_args", nargs="*", help="Extra arguments passed to pytest when using --run")

    args = parser.parse_args()

    shards = get_shards(total_shards=args.total_shards)

    if args.verify or args.verify_collection or args.audit_postgres:
        print(f"Partitioned {sum(len(s) for s in shards)} test files across {args.total_shards} shards:")
        for idx, shard in enumerate(shards, start=1):
            total_bytes = sum(os.path.getsize(f) for f in shard)
            print(f"  Shard {idx}/{args.total_shards}: {len(shard)} files ({total_bytes:,} bytes)")

        if args.verify or args.verify_collection:
            print("\nVerifying pytest collection coverage...")
            ok, msg = verify_collection_coverage(shards)
            if not ok:
                print(f"✗ {msg}", file=sys.stderr)
                return 1
            print(f"✓ {msg}")

        if args.verify or args.audit_postgres:
            print("\nAuditing PostgreSQL test registration...")
            ok, msg, _ = audit_postgres_tests()
            if not ok:
                print(f"✗ {msg}", file=sys.stderr)
                return 1
            print(f"✓ {msg}")

        return 0

    if args.shard is None:
        parser.error("--shard is required unless using --verify/--verify-collection/--audit-postgres")

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
