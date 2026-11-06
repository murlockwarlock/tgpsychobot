from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest

from scripts.ci_test_shards import get_shards
from scripts.run_fast_ci import check_syntax, get_changed_files, resolve_targeted_tests


def test_ci_shards_invariants_and_coverage():
    all_files = sorted(glob.glob("tests/test_*.py"))
    assert len(all_files) >= 135

    shards = get_shards(total_shards=3, tests_dir="tests")
    assert len(shards) == 3

    flattened = [f for shard in shards for f in shard]
    # Invariant: exactly the same set of files, no duplicates, no missing
    assert len(flattened) == len(all_files)
    assert set(flattened) == set(all_files)

    # Invariant: reasonably balanced weights (within 20% of average)
    weights = [sum(os.path.getsize(f) for f in shard) for shard in shards]
    avg_weight = sum(weights) / 3
    for w in weights:
        diff_pct = abs(w - avg_weight) / avg_weight
        assert diff_pct < 0.05, f"Shard weight {w} deviates by {diff_pct:.2%} from avg {avg_weight}"


def test_ci_shards_custom_count():
    all_files = sorted(glob.glob("tests/test_*.py"))
    shards_4 = get_shards(total_shards=4, tests_dir="tests")
    assert len(shards_4) == 4
    flattened_4 = [f for shard in shards_4 for f in shard]
    assert set(flattened_4) == set(all_files)


def test_resolve_targeted_tests_mapping():
    # Telegram admin mapping
    tg_tests = resolve_targeted_tests(["event_admin_telegram.py"])
    assert "tests/test_event_admin_telegram.py" in tg_tests
    assert "tests/test_event_content_validator.py" in tg_tests
    assert "tests/test_max_api_contract.py" in tg_tests

    # MAX admin mapping
    max_tests = resolve_targeted_tests(["max_messenger_bot/services/admin_events.py"])
    assert "tests/test_event_admin_max.py" in max_tests
    assert "tests/test_max_api_contract.py" in max_tests

    # Content validator mapping
    val_tests = resolve_targeted_tests(["event_content_validator.py"])
    assert "tests/test_event_content_validator.py" in val_tests
    assert "tests/test_event_admin_telegram.py" in val_tests
    assert "tests/test_event_admin_max.py" in val_tests

    # Direct test file mapping
    direct_tests = resolve_targeted_tests(["tests/test_subscription_context.py"])
    assert "tests/test_subscription_context.py" in direct_tests

    # Fallback when only docs or config changed
    empty_diff_tests = resolve_targeted_tests([])
    assert "tests/test_event_admin_telegram.py" in empty_diff_tests
    assert "tests/test_event_admin_max.py" in empty_diff_tests
    assert "tests/test_max_api_contract.py" in empty_diff_tests
    assert "tests/test_event_content_validator.py" in empty_diff_tests
