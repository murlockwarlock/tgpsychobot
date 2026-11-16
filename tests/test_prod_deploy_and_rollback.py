import os
from pathlib import Path
import subprocess
import pytest

from scripts.simulate_pr97_deployment_and_rollback import (
    run_scenario_1_archive_packaging_and_sha_verification,
    run_scenario_2_fail_closed_code_backup,
    run_scenario_3_preflight_rejection_before_code_change,
    run_scenario_4_syntax_error_abort_before_pm2_reload,
    run_scenario_5_partial_deploy_pm2_failure_and_rollback,
    run_scenario_6_path_normalization_and_runtime_preservation,
    run_scenario_7_idempotent_migrations_and_companion_ordering,
    run_scenario_8_mandatory_verification_failure_fails_closed,
)


def test_scenario_1_archive_packaging_and_sha_verification(tmp_path: Path):
    archive_path, commit_sha, short_sha = run_scenario_1_archive_packaging_and_sha_verification(tmp_path)
    assert archive_path.is_file()
    assert len(commit_sha) == 40
    assert len(short_sha) == 7


def test_scenario_2_fail_closed_code_backup(tmp_path: Path):
    prod_dir, backup_file = run_scenario_2_fail_closed_code_backup(tmp_path)
    assert prod_dir.is_dir()
    assert backup_file.is_file()


def test_scenario_3_preflight_rejection_before_code_change(tmp_path: Path):
    prod_dir, backup_file = run_scenario_2_fail_closed_code_backup(tmp_path)
    run_scenario_3_preflight_rejection_before_code_change(prod_dir)


def test_scenario_4_syntax_error_abort_before_pm2_reload(tmp_path: Path):
    prod_dir, backup_file = run_scenario_2_fail_closed_code_backup(tmp_path)
    run_scenario_4_syntax_error_abort_before_pm2_reload(prod_dir, backup_file)


def test_scenario_5_partial_deploy_pm2_failure_and_rollback(tmp_path: Path):
    prod_dir, backup_file = run_scenario_2_fail_closed_code_backup(tmp_path)
    run_scenario_5_partial_deploy_pm2_failure_and_rollback(prod_dir, backup_file)


def test_scenario_6_path_normalization_and_runtime_preservation(tmp_path: Path):
    prod_dir, backup_file = run_scenario_2_fail_closed_code_backup(tmp_path)
    run_scenario_6_path_normalization_and_runtime_preservation(prod_dir)


def test_scenario_7_idempotent_migrations_and_companion_ordering():
    run_scenario_7_idempotent_migrations_and_companion_ordering()


def test_scenario_8_mandatory_verification_failure_fails_closed(tmp_path: Path):
    run_scenario_8_mandatory_verification_failure_fails_closed(tmp_path)


def test_rollback_script_help():
    res = subprocess.run(
        ["bash", "rollback_prod.sh", "--help"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Usage: ./rollback_prod.sh" in res.stdout
    assert "--archive" in res.stdout


def test_rollback_script_rejects_invalid_pm2_names(tmp_path: Path):
    env = os.environ.copy()
    env["PROD_PM2_NAMES"] = "bad;injected_command"
    env["PROD_HOST"] = "example.invalid"
    res = subprocess.run(
        ["bash", "rollback_prod.sh"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "contains invalid process names" in res.stderr
    assert "ssh" not in res.stderr.lower()


def test_e2e_full_simulation_flow(tmp_path: Path):
    archive_path, commit_sha, short_sha = run_scenario_1_archive_packaging_and_sha_verification(tmp_path)
    prod_dir, backup_file = run_scenario_2_fail_closed_code_backup(tmp_path)
    run_scenario_3_preflight_rejection_before_code_change(prod_dir)
    run_scenario_4_syntax_error_abort_before_pm2_reload(prod_dir, backup_file)
    run_scenario_5_partial_deploy_pm2_failure_and_rollback(prod_dir, backup_file)
    run_scenario_6_path_normalization_and_runtime_preservation(prod_dir)
    run_scenario_7_idempotent_migrations_and_companion_ordering()
    run_scenario_8_mandatory_verification_failure_fails_closed(tmp_path)
