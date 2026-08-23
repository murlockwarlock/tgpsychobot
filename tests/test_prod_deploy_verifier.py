from scripts.verify_prod_runtime import (
    parse_names,
    process_database_url,
    validate_pm2_snapshot,
    validate_pm2_stability,
)


def _process(*, status="online", pid=101, restart_time=4, database_url="postgresql+asyncpg://db"):
    return {
        "name": "psy5d_new",
        "pm2_env": {
            "status": status,
            "pid": pid,
            "restart_time": restart_time,
            "pm_exec_path": "/root/telegram_bots/newbots/main.py",
            "env": {"DATABASE_URL": database_url},
        },
    }


def test_parse_names_trims_and_ignores_empty_values():
    assert parse_names(" a, ,b,, a ") == ["a", "b", "a"]


def test_pm2_snapshot_requires_online_processes_with_pids():
    assert validate_pm2_snapshot({"psy5d_new": _process()}, ["psy5d_new"]) == []
    assert validate_pm2_snapshot(
        {"psy5d_new": _process(status="stopped", pid=0)},
        ["psy5d_new"],
    ) == ["not_online:psy5d_new", "missing_pid:psy5d_new"]


def test_pm2_stability_detects_restart_or_pid_change():
    first = {"psy5d_new": _process()}
    second = {"psy5d_new": _process(pid=202, restart_time=5)}
    assert validate_pm2_stability(first, second, ["psy5d_new"]) == [
        "pid_changed:psy5d_new",
        "restart_count_changed:psy5d_new",
    ]


def test_database_url_is_read_from_process_environment_without_reformatting():
    process = _process(database_url="postgresql+asyncpg://user:password@host/db")
    assert process_database_url(process) == "postgresql+asyncpg://user:password@host/db"
