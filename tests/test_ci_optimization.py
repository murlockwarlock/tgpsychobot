from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest

from scripts.ci_gate import evaluate_ci_gate
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

    # Invariant: reasonably balanced weights (within 5% of average)
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


def test_resolve_targeted_tests_ui_journeys_and_boundaries():
    tg_journey_node = "tests/test_event_admin_telegram.py::test_telegram_visible_event_buttons_execute_real_dispatcher_journeys"
    tg_back_node = "tests/test_event_admin_telegram.py::test_telegram_pagination_page_2_edit_menu_back_roundtrip_journey"
    tg_emoji_node = "tests/test_event_admin_telegram.py::test_telegram_premium_emoji_paginated_journey_a_to_ya_to_a"

    max_journey_node = "tests/test_event_admin_max.py::test_max_visible_event_buttons_execute_real_application_journeys"
    max_roundtrip_node = "tests/test_event_admin_max.py::test_max_real_handle_update_paginated_format_save_cancel_journey"
    max_emoji_node = "tests/test_event_admin_max.py::test_max_premium_emoji_paginated_journey_a_to_ya_to_a"

    # 1. Non-UI / CI files change only (e.g. .github/workflows/ci.yml or docs)
    ci_tests = resolve_targeted_tests([".github/workflows/ci.yml", "scripts/ci_gate.py"])
    # Mandatory boundaries
    assert "tests/test_max_api_contract.py" in ci_tests
    assert "tests/test_event_content_validator.py" in ci_tests
    # Mandatory Telegram Dispatcher journeys
    assert tg_journey_node in ci_tests
    assert tg_back_node in ci_tests
    assert tg_emoji_node in ci_tests
    # Mandatory MAX Application journeys
    assert max_journey_node in ci_tests
    assert max_roundtrip_node in ci_tests
    assert max_emoji_node in ci_tests
    # Do NOT include whole heavy UI test files if not changed
    assert "tests/test_event_admin_telegram.py" not in ci_tests
    assert "tests/test_event_admin_max.py" not in ci_tests

    # 2. Empty diff
    empty_diff_tests = resolve_targeted_tests([])
    assert "tests/test_max_api_contract.py" in empty_diff_tests
    assert "tests/test_event_content_validator.py" in empty_diff_tests
    assert tg_journey_node in empty_diff_tests
    assert max_journey_node in empty_diff_tests

    # 3. Telegram admin UI change
    tg_tests = resolve_targeted_tests(["event_admin_telegram.py"])
    assert "tests/test_event_admin_telegram.py" in tg_tests
    assert "tests/test_event_content_validator.py" in tg_tests
    assert "tests/test_max_api_contract.py" in tg_tests
    # Whole file is present -> do not duplicate specific node IDs for Telegram
    assert tg_journey_node not in tg_tests
    # But MAX journey nodes must be present since whole MAX file was not changed
    assert max_journey_node in tg_tests

    # 4. MAX admin UI change
    max_tests = resolve_targeted_tests(["max_messenger_bot/services/admin_events.py"])
    assert "tests/test_event_admin_max.py" in max_tests
    assert "tests/test_max_api_contract.py" in max_tests
    # Whole file is present -> do not duplicate specific node IDs for MAX
    assert max_journey_node not in max_tests
    # But Telegram journey nodes must be present
    assert tg_journey_node in max_tests

    # 5. Core event admin service change
    svc_tests = resolve_targeted_tests(["event_admin_service.py"])
    assert "tests/test_event_admin_telegram.py" in svc_tests
    assert "tests/test_event_admin_max.py" in svc_tests
    assert tg_journey_node not in svc_tests
    assert max_journey_node not in svc_tests


def test_evaluate_ci_gate_matrix_including_skipped_postgres():
    # Full CI: valid success
    ok, msg = evaluate_ci_gate("skipped", "success", "success")
    assert ok is True
    assert "Full CI completed successfully" in msg

    # Full CI: skipped postgres (MUST FAIL)
    ok, msg = evaluate_ci_gate("skipped", "success", "skipped")
    assert ok is False
    assert "postgres-event-chains=success" in msg
    assert "skipped is not allowed" in msg

    # Full CI: failed postgres
    ok, msg = evaluate_ci_gate("skipped", "success", "failure")
    assert ok is False
    assert "postgres-event-chains=success" in msg

    # Full CI: skipped sqlite
    ok, msg = evaluate_ci_gate("skipped", "skipped", "success")
    assert ok is False
    assert "sqlite=success" in msg

    # Full CI: failed sqlite
    ok, msg = evaluate_ci_gate("skipped", "failure", "success")
    assert ok is False
    assert "sqlite=success" in msg

    # Full CI: cancelled sqlite
    ok, msg = evaluate_ci_gate("skipped", "cancelled", "success")
    assert ok is False

    # Fast CI: valid success
    ok, msg = evaluate_ci_gate("success", "skipped", "skipped")
    assert ok is True
    assert "Fast CI completed successfully" in msg

    # Fast CI: failure
    ok, msg = evaluate_ci_gate("failure", "skipped", "skipped")
    assert ok is False
    assert "Fast CI failed" in msg

    # Fast CI: cancelled
    ok, msg = evaluate_ci_gate("cancelled", "skipped", "skipped")
    assert ok is False

    # Fast CI: unexpected postgres or sqlite state
    ok, msg = evaluate_ci_gate("success", "failure", "skipped")
    assert ok is False

    ok, msg = evaluate_ci_gate("success", "skipped", "failure")
    assert ok is False

    # All skipped (fail-closed)
    ok, msg = evaluate_ci_gate("skipped", "skipped", "skipped")
    assert ok is False

    # Inconsistent / all success (fail-closed)
    ok, msg = evaluate_ci_gate("success", "success", "success")
    assert ok is False
