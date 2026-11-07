from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest

from scripts.ci_gate import evaluate_ci_gate
from scripts.ci_test_shards import (
    audit_postgres_tests,
    discover_test_files,
    get_shards,
    verify_collection_coverage,
)
from scripts.run_fast_ci import (
    check_syntax,
    discover_ui_journey_nodes,
    get_changed_files,
    resolve_targeted_tests,
)
from tests.test_event_admin_max import _max_event_callback_contract
from tests.test_event_admin_telegram import _event_callback_contract


def test_ci_shards_invariants_and_coverage():
    all_files = discover_test_files("tests")
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
    all_files = discover_test_files("tests")
    shards_4 = get_shards(total_shards=4, tests_dir="tests")
    assert len(shards_4) == 4
    flattened_4 = [f for shard in shards_4 for f in shard]
    assert set(flattened_4) == set(all_files)


def test_ci_shards_edge_cases_and_nested_directories(tmp_path: Path):
    # 1. 0 files
    empty_dir = tmp_path / "empty_tests"
    empty_dir.mkdir()
    empty_shards = get_shards(total_shards=3, tests_dir=str(empty_dir))
    assert empty_shards == [[], [], []]

    # 2. 1 file
    one_dir = tmp_path / "one_test"
    one_dir.mkdir()
    f1 = one_dir / "test_single.py"
    f1.write_text("def test_one(): pass\n")
    one_shards = get_shards(total_shards=3, tests_dir=str(one_dir))
    assert len(one_shards) == 3
    flat_one = [f for s in one_shards for f in s]
    assert len(flat_one) == 1
    assert flat_one[0] == f1.as_posix()

    # 3. Multiple nested directories
    nested_dir = tmp_path / "nested_tests"
    nested_dir.mkdir()
    sub1 = nested_dir / "sub1"
    sub2 = nested_dir / "sub1" / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    f_root = nested_dir / "test_root.py"
    f_sub1 = sub1 / "test_sub1.py"
    f_sub2 = sub2 / "test_sub2.py"
    f_root.write_text("def test_r(): pass\n")
    f_sub1.write_text("def test_s1(): pass\n")
    f_sub2.write_text("def test_s2(): pass\n")

    discovered = discover_test_files(str(nested_dir))
    assert len(discovered) == 3
    assert f_root.as_posix() in discovered
    assert f_sub1.as_posix() in discovered
    assert f_sub2.as_posix() in discovered

    nested_shards = get_shards(total_shards=2, tests_dir=str(nested_dir))
    assert len(nested_shards) == 2
    flat_nested = [f for s in nested_shards for f in s]
    assert len(flat_nested) == 3
    assert set(flat_nested) == {f_root.as_posix(), f_sub1.as_posix(), f_sub2.as_posix()}


def test_verify_collection_coverage_discrepancy_protection():
    shards = get_shards(total_shards=3, tests_dir="tests")
    ok, msg = verify_collection_coverage(shards, tests_dir="tests")
    assert ok is True
    total_files = sum(len(s) for s in shards)
    assert f"All {total_files} test files from pytest collection are partitioned" in msg

    # Simulate discrepancy: omit one shard
    incomplete_shards = [shards[0], shards[1], []]
    ok_bad, msg_bad = verify_collection_coverage(incomplete_shards, tests_dir="tests")
    assert ok_bad is False
    assert "Pytest collection discrepancy!" in msg_bad
    assert "missing in shards" in msg_bad


def test_dynamic_ui_journey_discovery():
    discovered = discover_ui_journey_nodes("tests")
    assert len(discovered) >= 11

    # Mandatory Telegram Dispatcher journeys & button contracts
    assert "tests/test_event_admin_telegram.py::test_telegram_event_visible_button_contract_covers_changed_screens" in discovered
    assert "tests/test_event_admin_telegram.py::test_telegram_visible_event_buttons_dispatch_through_router_contract" in discovered
    assert "tests/test_event_admin_telegram.py::test_telegram_visible_event_buttons_execute_real_dispatcher_journeys" in discovered
    assert "tests/test_event_admin_telegram.py::test_telegram_real_paginated_format_save_cancel_journey_preserves_page_filter" in discovered
    assert "tests/test_event_admin_telegram.py::test_telegram_pagination_page_2_edit_menu_back_roundtrip_journey" in discovered
    assert "tests/test_event_admin_telegram.py::test_telegram_premium_emoji_paginated_journey_a_to_ya_to_a" in discovered

    # Mandatory MAX Application journeys & button contracts
    assert "tests/test_event_admin_max.py::test_max_event_visible_button_contract_covers_changed_screens" in discovered
    assert "tests/test_event_admin_max.py::test_max_visible_event_buttons_dispatch_through_application_contract" in discovered
    assert "tests/test_event_admin_max.py::test_max_visible_event_buttons_execute_real_application_journeys" in discovered
    assert "tests/test_event_admin_max.py::test_max_real_handle_update_paginated_format_save_cancel_journey" in discovered
    assert "tests/test_event_admin_max.py::test_max_premium_emoji_paginated_journey_a_to_ya_to_a" in discovered


def test_dynamic_ui_journey_custom_registration(tmp_path: Path):
    custom_dir = tmp_path / "custom_ui"
    custom_dir.mkdir()
    custom_file = custom_dir / "test_custom_admin_flow.py"
    custom_file.write_text(
        "import pytest\n\n"
        "@pytest.mark.ui_journey\n"
        "async def test_new_admin_flow_journey():\n"
        "    pass\n"
    )

    discovered = discover_ui_journey_nodes(str(custom_dir))
    assert len(discovered) == 1
    assert discovered[0] == f"{custom_file.as_posix()}::test_new_admin_flow_journey"


def test_uncontracted_visible_button_raises_error():
    # Telegram uncontracted button
    with pytest.raises(AssertionError, match="Uncontracted event callback: admin_ev_fake_uncontracted_button"):
        _event_callback_contract("admin_ev_fake_uncontracted_button")

    # MAX uncontracted button
    with pytest.raises(AssertionError, match="Uncontracted event payload: admin_ev_fake_uncontracted_button"):
        _max_event_callback_contract("admin_ev_fake_uncontracted_button")


def test_audit_postgres_tests_registration():
    ok, msg, registered = audit_postgres_tests("tests")
    assert ok is True, msg
    assert len(registered) >= 9
    assert "tests/test_event_chains_db.py::test_real_postgresql_init_and_schema_verification" in registered
    assert "tests/test_event_chains_worker.py::test_real_postgresql_pre_send_lock_hierarchy_and_cas" in registered
    assert "tests/test_event_chains_service.py::test_real_postgresql_rejoin_vs_reschedule_deadlock_free" in registered


def test_audit_postgres_tests_detects_unregistered(tmp_path: Path):
    pg_dir = tmp_path / "unregistered_pg"
    pg_dir.mkdir()
    f = pg_dir / "test_pg_unregistered.py"
    f.write_text(
        "import os\n\n"
        "async def test_real_postgresql_unregistered_flow():\n"
        "    url = os.environ.get('POSTGRES_TEST_DATABASE_URL')\n"
        "    assert url is not None\n"
    )

    ok, msg, _ = audit_postgres_tests(str(pg_dir))
    assert ok is False
    assert "Unregistered PostgreSQL tests detected" in msg
    assert "test_real_postgresql_unregistered_flow" in msg


def test_resolve_targeted_tests_ui_journeys_and_boundaries():
    tg_journey_node = "tests/test_event_admin_telegram.py::test_telegram_visible_event_buttons_execute_real_dispatcher_journeys"
    max_journey_node = "tests/test_event_admin_max.py::test_max_visible_event_buttons_execute_real_application_journeys"

    # 1. Non-UI / CI files change only (e.g. .github/workflows/ci.yml or docs)
    ci_tests = resolve_targeted_tests([".github/workflows/ci.yml", "scripts/ci_gate.py"])
    assert "tests/test_max_api_contract.py" in ci_tests
    assert "tests/test_event_content_validator.py" in ci_tests
    assert tg_journey_node in ci_tests
    assert max_journey_node in ci_tests
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
