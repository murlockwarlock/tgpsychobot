#!/usr/bin/env python3
"""Fast CI execution script for Draft PRs and local pre-push validation.

Performs:
1. Syntax validation (py_compile across all repository python files).
2. Clean imports of core platform and bot entrypoints.
3. Git diff analysis against the base branch to detect affected modules.
4. Targeted test suite execution including mandatory UI journey tests
   and boundary contracts (Telegram Pydantic & MAX HTTP boundary).
"""

from __future__ import annotations

import importlib
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path


CORE_IMPORT_MODULES = [
    "database",
    "event_content_validator",
    "event_admin_service",
    "event_admin_telegram",
    "event_delivery_worker",
    "max_messenger_bot.api",
    "max_messenger_bot.app",
    "max_messenger_bot.formatting",
    "max_messenger_bot.services.admin_events",
]

MANDATORY_BOUNDARY_TESTS = [
    "tests/test_max_api_contract.py",
    "tests/test_event_content_validator.py",
]

MODULE_TEST_MAPPINGS: dict[str, list[str]] = {
    "event_admin_telegram.py": [
        "tests/test_event_admin_telegram.py",
        "tests/test_event_content_validator.py",
    ],
    "event_admin_service.py": [
        "tests/test_event_admin_service.py",
        "tests/test_event_admin_telegram.py",
        "tests/test_event_admin_max.py",
    ],
    "event_content_validator.py": [
        "tests/test_event_content_validator.py",
        "tests/test_event_admin_telegram.py",
        "tests/test_event_admin_max.py",
    ],
    "event_delivery_worker.py": [
        "tests/test_event_media_delivery.py",
        "tests/test_event_chains_worker.py",
    ],
    "event_media.py": [
        "tests/test_event_media_delivery.py",
        "tests/test_event_chains_db.py",
    ],
    "event_service.py": [
        "tests/test_event_chains_service.py",
        "tests/test_event_media_delivery.py",
    ],
    "database.py": [
        "tests/test_database_init.py",
        "tests/test_event_content_validator.py",
    ],
    "ai_integration.py": [
        "tests/test_ai_terminal_failure_retry.py",
        "tests/test_ai_request_builder_parity.py",
    ],
}


def check_syntax(root_dir: Path) -> None:
    print("Step 1: Syntax compilation check...")
    t0 = time.time()
    compiled = 0
    errors = []

    for path in root_dir.rglob("*.py"):
        try:
            rel = path.relative_to(root_dir)
        except ValueError:
            continue
        # Skip virtualenvs, git, cache within the project root
        if any(part.startswith((".", "venv", ".venv")) or part == "__pycache__" for part in rel.parts):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            compiled += 1
        except py_compile.PyCompileError as exc:
            errors.append(f"{rel}: {exc}")

    if errors:
        print(f"✗ Syntax compilation failed on {len(errors)} files:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)

    print(f"✓ Syntax OK: compiled {compiled} Python files in {time.time() - t0:.2f}s")


def check_imports(root_dir: Path) -> None:
    print("Step 2: Core module imports check...")
    t0 = time.time()
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    os.environ.setdefault("BOT_TOKEN", "ci-test-token")
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("TESTING", "true")

    for mod_name in CORE_IMPORT_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            print(f"✗ Import failed for {mod_name}: {exc}")
            sys.exit(1)

    print(f"✓ Imports OK: loaded {len(CORE_IMPORT_MODULES)} modules in {time.time() - t0:.2f}s")


def get_changed_files(base_ref: str | None = None) -> list[str]:
    # Determine base git reference
    targets = []
    if base_ref:
        targets.append(base_ref)
        if not base_ref.startswith("origin/"):
            targets.append(f"origin/{base_ref}")
    base_env = os.environ.get("GITHUB_BASE_REF")
    if base_env:
        targets.extend([f"origin/{base_env}", base_env])
    targets.extend(["origin/main", "main", "HEAD~1"])

    chosen_base = None
    for candidate in targets:
        proc = subprocess.run(["git", "rev-parse", "--verify", candidate], capture_output=True, text=True)
        if proc.returncode == 0:
            chosen_base = candidate
            break

    if not chosen_base:
        return []

    diff_proc = subprocess.run(
        ["git", "diff", "--name-only", f"{chosen_base}...HEAD"],
        capture_output=True,
        text=True,
    )
    if diff_proc.returncode != 0:
        diff_proc = subprocess.run(
            ["git", "diff", "--name-only", chosen_base],
            capture_output=True,
            text=True,
        )

    if diff_proc.returncode != 0:
        return []

    lines = [line.strip() for line in diff_proc.stdout.splitlines() if line.strip()]

    # Also detect uncommitted / working tree changes when run locally
    status_proc = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status_proc.returncode == 0:
        for st_line in status_proc.stdout.splitlines():
            st_line = st_line.strip()
            if not st_line:
                continue
            parts = st_line.split(maxsplit=1)
            if len(parts) == 2:
                lines.append(parts[1].strip())

    # Return unique sorted paths
    return sorted(set(lines))


def resolve_targeted_tests(changed_files: list[str]) -> list[str]:
    selected_tests: set[str] = set()

    for path_str in changed_files:
        path = Path(path_str)
        # Direct test modification
        if path_str.startswith("tests/test_") and path_str.endswith(".py") and path.exists():
            selected_tests.add(path_str)
            continue

        # Check known mappings
        name = path.name
        if name in MODULE_TEST_MAPPINGS:
            for test_file in MODULE_TEST_MAPPINGS[name]:
                if Path(test_file).exists():
                    selected_tests.add(test_file)

        # MAX bot modifications
        if "max_messenger_bot" in path_str:
            selected_tests.add("tests/test_event_admin_max.py")
            selected_tests.add("tests/test_max_api_contract.py")

        # Telegram admin modifications
        if "event_admin_telegram" in path_str or "event_admin" in path_str:
            selected_tests.add("tests/test_event_admin_telegram.py")

        # Heuristic: test_{stem}.py
        candidate = f"tests/test_{path.stem}.py"
        if Path(candidate).exists():
            selected_tests.add(candidate)

    # Always ensure mandatory boundary contract tests are included
    for mb in MANDATORY_BOUNDARY_TESTS:
        if Path(mb).exists():
            selected_tests.add(mb)

    # If no specific tests were selected (e.g. only docs or workflows changed),
    # include representative Telegram & MAX UI journeys + boundaries.
    if len(selected_tests) <= len(MANDATORY_BOUNDARY_TESTS):
        selected_tests.add("tests/test_event_admin_telegram.py")
        selected_tests.add("tests/test_event_admin_max.py")

    return sorted(selected_tests)


def main() -> int:
    root_dir = Path.cwd()
    check_syntax(root_dir)
    check_imports(root_dir)

    print("Step 3: Determining affected modules and UI journeys...")
    changed = get_changed_files()
    print(f"Changed files detected ({len(changed)}):")
    for f in changed[:10]:
        print(f"  • {f}")
    if len(changed) > 10:
        print(f"  ... and {len(changed) - 10} more")

    targeted_tests = resolve_targeted_tests(changed)
    print(f"\nTargeted test suite ({len(targeted_tests)} test files):")
    for t in targeted_tests:
        print(f"  ✓ {t}")

    print("\nStep 4: Executing FAST CI test suite...")
    t0 = time.time()
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs"] + targeted_tests + sys.argv[1:]
    res = subprocess.run(cmd)

    elapsed = time.time() - t0
    print(f"\nFast CI completed in {elapsed:.2f}s with status code {res.returncode}")
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
