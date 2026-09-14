import asyncio

import aiohttp
import pytest
from unittest.mock import patch


import monitor_psychobots as monitor


def test_proxy_request_kwargs_hides_credentials_from_url():
    kwargs = monitor.proxy_request_kwargs(
        "http://user%40name:pass%25word@185.70.185.209:3128"
    )

    assert kwargs["proxy"] == "http://185.70.185.209:3128"
    assert kwargs["proxy_auth"] == aiohttp.BasicAuth("user@name", "pass%word")


def test_proxy_request_kwargs_rejects_unsupported_scheme():
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        monitor.proxy_request_kwargs("socks5://127.0.0.1:1080")


@pytest.mark.asyncio
async def test_tcp_check_reports_unavailable_port(monkeypatch):
    async def fail_connection(host, port):
        raise ConnectionRefusedError

    monkeypatch.setattr(asyncio, "open_connection", fail_connection)

    issue = await monitor.check_tcp_port("185.70.185.209", 62050, "Marzban Node")

    assert issue == (
        "Marzban Node: 185.70.185.209:62050 is unavailable "
        "(ConnectionRefusedError)"
    )


@pytest.mark.asyncio
async def test_nl_check_covers_all_required_ports(monkeypatch):
    checked_ports = []

    async def record_port(host, port, label):
        checked_ports.append((host, port, label))
        return None

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class Session:
        def get(self, url, **kwargs):
            assert url == "https://api.telegram.org"
            assert kwargs["proxy"] == "http://185.70.185.209:3128"
            return Response()

    monkeypatch.setattr(monitor, "check_tcp_port", record_port)
    monkeypatch.setattr(monitor, "NL_CHECK_ENABLED", True)
    monkeypatch.setattr(monitor, "NL_XRAY_PORTS", (2053, 2069))
    monkeypatch.setenv("TELEGRAM_PROXY", "http://185.70.185.209:3128")

    issues = await monitor.check_nl_infrastructure(Session())

    assert issues == []
    assert {port for _, port, _ in checked_ports} == {3128, 443, 4430, 62050, 2053, 2069}


@pytest.mark.asyncio
async def test_payment_webhook_check_accepts_expected_bad_request():
    class Response:
        status = 400

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class Session:
        def get(self, url, **kwargs):
            assert url == "https://bots.example:8443/bot_legacy_1/webhooks/robokassa/result"
            assert kwargs == {"allow_redirects": False}
            return Response()

    issues = await monitor.check_payment_webhooks(
        {
            "webhook_base_url": "https://bots.example:8443/",
            "webhook_path_prefix": "/bot_legacy_1",
        },
        Session(),
    )

    assert issues == []


@pytest.mark.asyncio
async def test_payment_webhook_check_reports_unavailable_route():
    class Session:
        def get(self, url, **kwargs):
            raise asyncio.TimeoutError

    issues = await monitor.check_payment_webhooks(
        {
            "webhook_base_url": "https://bots.example:8443",
            "webhook_path_prefix": "bot1",
        },
        Session(),
    )

    assert issues == [
        "Payment webhook is unavailable: TimeoutError: "
        "https://bots.example:8443/bot1/webhooks/robokassa/result"
    ]


@pytest.mark.asyncio
async def test_payment_webhook_check_skips_apps_without_public_route():
    issues = await monitor.check_payment_webhooks({}, object())

    assert issues == []


def test_get_candidate_alert_tokens_prioritizes_preferred_pm2_bot(monkeypatch):
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "tg_autobusbusbot_new")
    monkeypatch.delenv("PSYCHOBOTS_ALERT_BOT_TOKEN", raising=False)

    apps = [
        {"name": "tg_someonelikeyouai04_bot_legacy", "token": "token_someone04"},
        {"name": "tg_autobusbusbot_new", "token": "token_autobus"},
        {"name": "tg_veraveda777_bot_legacy", "token": "token_veraveda"},
    ]

    tokens = monitor.get_candidate_alert_tokens(apps)

    assert tokens == ["token_autobus", "token_someone04", "token_veraveda"]


def test_get_candidate_alert_tokens_supports_direct_token_override(monkeypatch):
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_TOKEN", "direct_alert_token")
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "tg_autobusbusbot_new")

    apps = [
        {"name": "tg_autobusbusbot_new", "token": "token_autobus"},
        {"name": "tg_someonelikeyouai04_bot_legacy", "token": "token_someone04"},
    ]

    tokens = monitor.get_candidate_alert_tokens(apps)

    assert tokens == ["direct_alert_token", "token_autobus", "token_someone04"]


def test_get_candidate_alert_tokens_fallback_when_preferred_not_found(monkeypatch):
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "non_existent_bot")
    monkeypatch.delenv("PSYCHOBOTS_ALERT_BOT_TOKEN", raising=False)

    apps = [
        {"name": "tg_someonelikeyouai04_bot_legacy", "token": "token_someone04"},
        {"name": "tg_veraveda777_bot_legacy", "token": "token_veraveda"},
    ]

    tokens = monitor.get_candidate_alert_tokens(apps)

    assert tokens == ["token_someone04", "token_veraveda"]


# ==============================================================================
# Incident B1-F: Stuck AI Dialogue Watchdog Tests
# ==============================================================================


def test_make_safe_db_key_strips_credentials():
    url = "postgresql+asyncpg://admin_user:secret_pass%40123@db.example.com:5433/bot_prod_db"
    safe_key = monitor.make_safe_db_key(url)
    assert safe_key == "db.example.com:5433/bot_prod_db"
    assert "admin_user" not in safe_key
    assert "secret_pass" not in safe_key


def test_make_safe_db_key_handles_ipv6_with_colons():
    url = "postgresql://user:pass@[2001:db8::1]:5432/bot_db"
    safe_key = monitor.make_safe_db_key(url)
    assert safe_key == "[2001:db8::1]:5432/bot_db"
    assert "user" not in safe_key
    assert "pass" not in safe_key


def test_group_apps_by_db_deduplicates_shared_databases():
    apps = [
        {"name": "tg_app1", "db_url": "postgresql://u1:p1@10.0.0.1:5432/shared_db"},
        {"name": "max_app1", "db_url": "postgresql://u2:p2@10.0.0.1:5432/shared_db"},
        {"name": "tg_app2", "db_url": "postgresql://u3:p3@10.0.0.2:5432/other_db"},
    ]
    groups = monitor.group_apps_by_db(apps)
    assert len(groups) == 2
    assert len(groups["10.0.0.1:5432/shared_db"]) == 2
    assert len(groups["10.0.0.2:5432/other_db"]) == 1


def test_format_age_deterministic():
    assert monitor.format_age(0) == "0м"
    assert monitor.format_age(45) == "0м"
    assert monitor.format_age(120) == "2м"
    assert monitor.format_age(3600) == "1ч"
    assert monitor.format_age(3660) == "1ч1м"
    assert monitor.format_age(86400) == "1д"
    assert monitor.format_age(90060) == "1д1ч1м"


def test_make_incident_key_scoping():
    assert (
        monitor.make_incident_key("host:5432/db", 10, 1, "dialogue", None)
        == "host:5432/db:10:1"
    )
    assert (
        monitor.make_incident_key("host:5432/db", 10, 1, "reset", None)
        == "host:5432/db:10:1"
    )
    assert (
        monitor.make_incident_key("host:5432/db", 10, 1, "global", None)
        == "host:5432/db:10:1"
    )
    assert (
        monitor.make_incident_key("host:5432/db", 10, 1, "topic", 42)
        == "host:5432/db:10:1:topic:42"
    )
    assert (
        monitor.make_incident_key("host:5432/db", 10, 1, "topic", None)
        == "host:5432/db:10:1:topic:null"
    )


def test_format_stuck_alert_contains_no_message_content():
    cand = {
        "user_id": 12345,
        "dialogue_id": 2,
        "incident_scope_kind": "topic",
        "incident_topic_id": 10,
        "unanswered_count": 3,
        "first_unanswered_at": monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
        - monitor.timedelta(minutes=15),
    }
    group_apps = [{"name": "tg_bot_alpha"}, {"name": "max_bot_beta"}]
    text = monitor.format_stuck_alert(cand, group_apps)

    assert "12345" in text
    assert "tg_bot_alpha" in text and "max_bot_beta" in text
    assert "тема 10" in text
    assert "3" in text
    assert "15м назад" in text
    assert "password" not in text
    assert "content" not in text


def test_format_stuck_alert_displays_max_raw_id():
    cand = {
        "user_id": 100_000_123_456,
        "dialogue_id": 1,
        "incident_scope_kind": "dialogue",
        "incident_topic_id": None,
        "unanswered_count": 2,
        "first_unanswered_at": monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
        - monitor.timedelta(minutes=5),
    }
    group_apps = [{"name": "max_bot"}]
    text = monitor.format_stuck_alert(cand, group_apps)

    assert "100000123456" in text
    assert "MAX raw: 123456" in text


def test_single_instance_lock_prevents_concurrent_runs(tmp_path, monkeypatch):
    lock_file = str(tmp_path / "test.lock")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)

    # Dynamic resolution without args picks up monkeypatched env
    lock1 = monitor.SingleInstanceLock()
    assert lock1.lock_path == lock_file
    lock2 = monitor.SingleInstanceLock()

    with lock1 as acq1:
        assert acq1 is True
        with lock2 as acq2:
            assert acq2 is False

    with lock2 as acq2_after:
        assert acq2_after is True


def test_single_instance_lock_propagates_io_and_permission_errors(tmp_path):
    # Non-existent parent directory -> open fails with FileNotFoundError
    invalid_path = str(tmp_path / "non_existent_subdir" / "lock.file")
    lock = monitor.SingleInstanceLock(invalid_path)
    with pytest.raises(FileNotFoundError):
        with lock:
            pass

    # PermissionError simulation
    lock_valid = monitor.SingleInstanceLock(str(tmp_path / "perm.lock"))
    with patch("builtins.open", side_effect=PermissionError("Access denied")):
        with pytest.raises(PermissionError):
            with lock_valid:
                pass


@pytest.mark.asyncio
async def test_get_effective_memory_mode_canonical_id_1():
    class FakeConn:
        def __init__(self, table_exists=True, row_id1=None):
            self.table_exists = table_exists
            self.row_id1 = row_id1

        async def fetchval(self, sql):
            assert "to_regclass" in sql
            return "public.ai_config" if self.table_exists else None

        async def fetchrow(self, sql):
            assert "WHERE id = 1" in sql
            return self.row_id1

    # Table missing
    with pytest.raises(RuntimeError, match="Table 'ai_config' is missing"):
        await monitor.get_effective_memory_mode(FakeConn(table_exists=False))

    # id=1 missing
    with pytest.raises(RuntimeError, match="Canonical row AIConfig.*id=1.*is missing"):
        await monitor.get_effective_memory_mode(FakeConn(table_exists=True, row_id1=None))

    # Valid modes
    conn_topic = FakeConn(
        True, {"memory_mode": "topic", "preserve_topic_context": False}
    )
    assert await monitor.get_effective_memory_mode(conn_topic) == "topic"

    conn_global = FakeConn(
        True, {"memory_mode": "global", "preserve_topic_context": False}
    )
    assert await monitor.get_effective_memory_mode(conn_global) == "global"

    conn_reset = FakeConn(
        True, {"memory_mode": "reset", "preserve_topic_context": True}
    )
    assert await monitor.get_effective_memory_mode(conn_reset) == "reset"

    # Fallback from invalid/empty mode
    conn_fallback_topic = FakeConn(
        True, {"memory_mode": "custom_invalid", "preserve_topic_context": True}
    )
    assert await monitor.get_effective_memory_mode(conn_fallback_topic) == "topic"

    conn_fallback_reset = FakeConn(
        True, {"memory_mode": None, "preserve_topic_context": False}
    )
    assert await monitor.get_effective_memory_mode(conn_fallback_reset) == "reset"


@pytest.mark.asyncio
async def test_get_suppressed_max_user_ids_fail_closed():
    class FakeConn:
        def __init__(self, table_exists=True, should_raise=False):
            self.table_exists = table_exists
            self.should_raise = should_raise

        async def fetchval(self, sql):
            if self.should_raise:
                raise ConnectionError("DB dropped")
            return "public.max_bot_states" if self.table_exists else None

        async def fetch(self, sql):
            if self.should_raise:
                raise ConnectionError("DB dropped")
            return [{"user_id": 100_000_000_001}, {"user_id": 100_000_000_002}]

    # Table absent -> normal empty set
    res = await monitor.get_suppressed_max_user_ids(FakeConn(table_exists=False))
    assert res == set()

    # Table present -> returns IDs
    res = await monitor.get_suppressed_max_user_ids(FakeConn(table_exists=True))
    assert res == {100_000_000_001, 100_000_000_002}

    # Query fails -> exception propagates (fail closed)
    with pytest.raises(ConnectionError):
        await monitor.get_suppressed_max_user_ids(FakeConn(should_raise=True))


@pytest.mark.asyncio
async def test_fetch_stuck_dialogue_episodes_sql_generation():
    recorded_sql = {}
    recorded_params = {}

    class FakeConn:
        async def fetch(self, sql, *params):
            recorded_sql["query"] = sql
            recorded_params["params"] = params
            return []

    # 1. Topic mode query verification
    await monitor.fetch_stuck_dialogue_episodes(
        FakeConn(),
        effective_memory_mode="topic",
        max_age_hours=24,
        grace_seconds=180,
    )

    query = recorded_sql["query"]
    params = recorded_params["params"]

    # PostgreSQL specific syntax assertions
    assert "IS DISTINCT FROM 'user'" in query
    assert "t.tail_role = 'user'" in query
    assert "s.unanswered_count >= 2" in query
    assert "DISTINCT ON (m.user_id, m.dialogue_id)" in query

    # Scope condition matching build_history_scope in all 3 CTEs
    assert "m.topic_id IS NOT DISTINCT FROM u.current_topic_id" in query
    assert query.count("m.topic_id IS NOT DISTINCT FROM u.current_topic_id") == 3

    # Structural query does NOT filter by max_age_hours! Only grace cutoff is in params
    assert "AND m_last.timestamp >= $" not in query
    assert len(params) == 1

    # 2. Reset / Global mode query verification (no topic clause)
    await monitor.fetch_stuck_dialogue_episodes(
        FakeConn(),
        effective_memory_mode="reset",
        max_age_hours=0,
        grace_seconds=180,
    )
    query_reset = recorded_sql["query"]
    params_reset = recorded_params["params"]
    assert "m.topic_id IS NOT DISTINCT FROM u.current_topic_id" not in query_reset
    assert "AND m_last.timestamp >= $" not in query_reset
    assert len(params_reset) == 1


# ==============================================================================
# End-to-End Watchdog Lifecycle & State Machine Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_check_stuck_dialogues_active_stuck_alert_and_cooldown(monkeypatch):
    state = {"log_offsets": {"/path.log": 123}, "alerts": {}, "stuck_dialogues": {}}
    apps = [
        {
            "name": "tg_bot",
            "db_url": "postgresql://u:p@localhost:5432/test_db",
            "token": "tok1",
            "owner_ids": [999],
        }
    ]

    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 2,
            "first_unanswered_id": 10,
            "last_unanswered_id": 11,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=4),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        async def close(self):
            pass

    async def fake_connect(dsn, timeout):
        return FakeConn()

    async def fake_get_mode(conn):
        return "reset"

    async def fake_get_episodes(conn, mode, max_age_hours, grace_seconds):
        return mock_episodes

    async def fake_get_max_users(conn):
        return set()

    sent_alerts = []

    async def fake_send_alert(alert_apps, text):
        sent_alerts.append((alert_apps, text))
        return True

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_get_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_get_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_get_max_users)
    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    all_issues = []
    # Run 1: Should alert and commit state
    alerts_sent = await monitor.check_stuck_dialogues(apps, state, all_issues)
    assert alerts_sent == 1
    assert len(sent_alerts) == 1
    assert "101" in sent_alerts[0][1]
    assert all_issues == []

    inc_key = "localhost:5432/test_db:101:1"
    assert inc_key in state["stuck_dialogues"]
    entry = state["stuck_dialogues"][inc_key]
    assert entry["user_id"] == 101
    assert entry["last_count"] == 2
    assert state["log_offsets"] == {"/path.log": 123}

    # Run 2: Immediately running again within 6h cooldown -> no duplicate alert
    sent_alerts.clear()
    alerts_sent_2 = await monitor.check_stuck_dialogues(apps, state, all_issues)
    assert alerts_sent_2 == 0
    assert len(sent_alerts) == 0
    assert inc_key in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_fresh_retry_retains_cooldown_and_no_alert(monkeypatch):
    now_ts = int(monitor.time.time())
    inc_key = "localhost:5432/test_db:101:1"
    state = {
        "stuck_dialogues": {
            inc_key: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 1,
                "incident_scope_kind": "reset",
                "incident_topic_id": None,
                "first_alerted_at": now_ts - 600,
                "last_alerted_at": now_ts - 600,
                "last_count": 2,
                "first_id": 10,
                "last_id": 11,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    # User sends fresh retry 20 seconds ago:
    # In Stage A it is returned (suffix count = 3)
    # But is_grace_elapsed is False!
    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 3,
            "first_unanswered_id": 10,
            "last_unanswered_id": 12,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=10),
            "last_unanswered_at": now_utc - monitor.timedelta(seconds=20),
            "is_grace_elapsed": False,  # Fresh retry!
        }
    ]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout):
        return FakeConn()

    async def fake_mode(c):
        return "reset"

    async def fake_episodes(conn, mode, max_age_hours=0, grace_seconds=180):
        return mock_episodes

    async def fake_max(c):
        return set()

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)

    sent_alerts = []

    async def spy_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor, "send_alert", spy_send)

    # Run: fresh retry must NOT alert, and MUST NOT delete cooldown entry!
    alerts = await monitor.check_stuck_dialogues(apps, state, [])
    assert alerts == 0
    assert len(sent_alerts) == 0
    assert inc_key in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_recovery_removes_state_when_episode_resolved(monkeypatch):
    inc_key = "localhost:5432/test_db:101:1"
    state = {
        "stuck_dialogues": {
            inc_key: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 1,
                "incident_scope_kind": "reset",
                "incident_topic_id": None,
                "first_alerted_at": 1000,
                "last_alerted_at": 1000,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout):
        return FakeConn()

    async def fake_mode(c):
        return "reset"

    async def fake_empty_episodes(*args, **kwargs):
        return []

    async def fake_empty_max(c):
        return set()

    # Episode resolved: assistant responded -> Stage A returns empty list!
    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_empty_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_empty_max)

    await monitor.check_stuck_dialogues(apps, state, [])
    assert inc_key not in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_delivery_failure_does_not_commit_cooldown(monkeypatch):
    state = {"stuck_dialogues": {}}
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 2,
            "first_unanswered_id": 1,
            "last_unanswered_id": 2,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=4),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout):
        return FakeConn()

    async def fake_mode(c):
        return "reset"

    async def fake_get_episodes(*args, **kwargs):
        return mock_episodes

    async def fake_empty_max(c):
        return set()

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_get_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_empty_max)

    # Delivery fails:
    async def fake_failing_send(a, t):
        return False

    monkeypatch.setattr(monitor, "send_alert", fake_failing_send)

    alerts = await monitor.check_stuck_dialogues(apps, state, [])
    assert alerts == 0
    inc_key = "localhost:5432/test_db:101:1"
    assert inc_key not in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_db_query_failure_preserves_state_and_emits_issue(monkeypatch):
    inc_key = "localhost:5432/test_db:101:1"
    state = {
        "stuck_dialogues": {
            inc_key: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 1,
                "incident_scope_kind": "reset",
                "incident_topic_id": None,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    async def failing_connect(d, timeout):
        raise ConnectionRefusedError("PostgreSQL down")

    monkeypatch.setattr(monitor.asyncpg, "connect", failing_connect)

    all_issues = []
    await monitor.check_stuck_dialogues(apps, state, all_issues)

    # State key must remain preserved!
    assert inc_key in state["stuck_dialogues"]
    assert len(all_issues) == 1
    assert "stuck dialogue check failed" in all_issues[0][1][0]


@pytest.mark.asyncio
async def test_max_state_query_failure_fails_closed(monkeypatch):
    inc_key = "localhost:5432/test_db:101:1"
    state = {
        "stuck_dialogues": {
            inc_key: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 1,
                "incident_scope_kind": "reset",
                "incident_topic_id": None,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout):
        return FakeConn()

    async def fake_mode(c):
        return "reset"

    async def fake_empty_episodes(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_empty_episodes)

    # MAX state query crashes:
    async def failing_max_check(conn):
        raise RuntimeError("Corrupted max_bot_states")

    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", failing_max_check)

    all_issues = []
    alerts = await monitor.check_stuck_dialogues(apps, state, all_issues)
    assert alerts == 0
    # State key preserved on error
    assert inc_key in state["stuck_dialogues"]
    assert len(all_issues) == 1
    assert "Corrupted max_bot_states" in all_issues[0][1][0]


@pytest.mark.asyncio
async def test_max_state_active_suppresses_stage_b_but_preserves_stage_a(monkeypatch):
    state = {"stuck_dialogues": {}}
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 2,
            "first_unanswered_id": 1,
            "last_unanswered_id": 2,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=4),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout):
        return FakeConn()

    async def fake_mode(c):
        return "reset"

    async def fake_get_episodes(*args, **kwargs):
        return mock_episodes

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_get_episodes)

    # User 101 is actively in a test flow:
    async def fake_max_suppressed(c):
        return {101}

    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max_suppressed)

    sent_alerts = []

    async def spy_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor, "send_alert", spy_send)

    alerts = await monitor.check_stuck_dialogues(apps, state, [])
    assert alerts == 0
    assert len(sent_alerts) == 0


@pytest.mark.asyncio
async def test_topic_same_dialogue_id_distinct_incidents(monkeypatch):
    # Two topics sharing dialogue_id = 5 must have distinct keys and not suppress each other
    now_ts = int(monitor.time.time())
    key_topic_10 = "localhost:5432/test_db:101:5:topic:10"
    state = {
        "stuck_dialogues": {
            key_topic_10: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 5,
                "incident_scope_kind": "topic",
                "incident_topic_id": 10,
                "first_alerted_at": now_ts - 100,
                "last_alerted_at": now_ts - 100,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    # User switched to topic 20 (also dialogue_id 5) and is stuck there:
    mock_episodes = [
        {
            "user_id": 101,
            "dialogue_id": 5,
            "current_topic_id": 20,
            "incident_scope_kind": "topic",
            "incident_topic_id": 20,
            "unanswered_count": 2,
            "first_unanswered_id": 50,
            "last_unanswered_id": 51,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=4),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout):
        return FakeConn()

    async def fake_mode(c):
        return "topic"

    async def fake_get_episodes(*args, **kwargs):
        return mock_episodes

    async def fake_empty_max(c):
        return set()

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_get_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_empty_max)

    sent_alerts = []

    async def spy_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor, "send_alert", spy_send)

    alerts = await monitor.check_stuck_dialogues(apps, state, [])
    # Topic 20 must alert as a NEW incident!
    assert alerts == 1
    key_topic_20 = "localhost:5432/test_db:101:5:topic:20"
    assert key_topic_20 in state["stuck_dialogues"]
    # Topic 10 was cleaned up by recovery since it is no longer the active Stage A episode!
    assert key_topic_10 not in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_alert_routing_uses_all_apps_and_labels_group_apps(monkeypatch):
    apps = [
        {
            "name": "tg_affected_bot",
            "db_url": "postgresql://u:p@localhost:5432/db1",
            "token": "tok_affected",
            "owner_ids": [100],
        },
        {
            "name": "tg_alert_router_bot",
            "db_url": "postgresql://u:p@localhost:5432/db2",
            "token": "tok_preferred",
            "owner_ids": [200],
        },
    ]

    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "tg_alert_router_bot")

    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes_db1 = [
        {
            "user_id": 555,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 2,
            "first_unanswered_id": 1,
            "last_unanswered_id": 2,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=4),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        def __init__(self, db_url): self.db_url = db_url
        async def close(self): pass

    async def fake_connect(dsn, timeout):
        return FakeConn(dsn)

    async def fake_mode(c):
        return "reset"

    async def mock_fetch(conn, mode, max_age_hours, grace_seconds):
        if "db1" in conn.db_url:
            return mock_episodes_db1
        return []

    async def fake_empty_max(c):
        return set()

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", mock_fetch)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_empty_max)

    sent_calls = []

    async def spy_send_alert(delivered_apps, text):
        sent_calls.append((delivered_apps, text))
        return True

    monkeypatch.setattr(monitor, "send_alert", spy_send_alert)

    state = {}
    await monitor.check_stuck_dialogues(apps, state, [])

    assert len(sent_calls) == 1
    delivered_apps, text = sent_calls[0]

    # Delivery routing uses all apps (so preferred bot is available)
    assert len(delivered_apps) == 2
    assert {a["name"] for a in delivered_apps} == {"tg_affected_bot", "tg_alert_router_bot"}

    # Incident text labels only affected group app
    assert "tg_affected_bot" in text
    assert "tg_alert_router_bot" not in text


@pytest.mark.asyncio
async def test_run_check_exit_codes(monkeypatch, tmp_path):
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    async def fake_nl(s):
        return []

    monkeypatch.setattr(monitor, "get_bot_apps", lambda: [])
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_nl)

    # Clean run -> exit 0
    res = await monitor.run_check(False)
    assert res == 0

    # Overlapping run -> exit 0 immediately
    with monitor.SingleInstanceLock(lock_file):
        res_overlap = await monitor.run_check(False)
        assert res_overlap == 0


@pytest.mark.asyncio
async def test_db_a_failure_does_not_stop_db_b(monkeypatch):
    apps = [
        {"name": "bot_a", "db_url": "postgresql://u:p@10.0.0.1:5432/db_a", "token": "t_a", "owner_ids": [1]},
        {"name": "bot_b", "db_url": "postgresql://u:p@10.0.0.2:5432/db_b", "token": "t_b", "owner_ids": [2]},
    ]
    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes_b = [
        {
            "user_id": 202,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 2,
            "first_unanswered_id": 1,
            "last_unanswered_id": 2,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=4),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        def __init__(self, dsn):
            self.dsn = dsn
        async def close(self):
            pass

    async def fake_connect(dsn, timeout):
        return FakeConn(dsn)

    async def fake_mode(c):
        return "reset"

    async def fake_episodes(conn, mode, *args, **kwargs):
        if "db_a" in conn.dsn:
            raise ConnectionError("db_a failed query")
        return mock_episodes_b

    async def fake_max(conn):
        return set()

    sent_alerts = []
    async def fake_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)
    monkeypatch.setattr(monitor, "send_alert", fake_send)

    all_issues = []
    state = {}
    alerts = await monitor.check_stuck_dialogues(apps, state, all_issues)

    # db_a failed -> emitted to all_issues
    assert len(all_issues) == 1
    assert "db_a failed query" in all_issues[0][1][0]
    # db_b succeeded -> alert sent
    assert alerts == 1
    assert len(sent_alerts) == 1
    assert "202" in sent_alerts[0]


@pytest.mark.asyncio
async def test_cooldown_not_bypassed_by_count_or_age_change(monkeypatch):
    now_ts = int(monitor.time.time())
    inc_key = "localhost:5432/test_db:101:1"
    state = {
        "stuck_dialogues": {
            inc_key: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 1,
                "incident_scope_kind": "reset",
                "incident_topic_id": None,
                "first_alerted_at": now_ts - 1000,
                "last_alerted_at": now_ts - 1000,
                "last_count": 2,
                "first_id": 1,
                "last_id": 2,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    # User now has 4 unanswered messages (count and age changed), but 6h cooldown has not passed
    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 4,
            "first_unanswered_id": 1,
            "last_unanswered_id": 4,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=30),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout): return FakeConn()
    async def fake_mode(c): return "reset"
    async def fake_episodes(*args, **kwargs): return mock_episodes
    async def fake_max(c): return set()

    sent_alerts = []
    async def fake_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)
    monkeypatch.setattr(monitor, "send_alert", fake_send)

    alerts = await monitor.check_stuck_dialogues(apps, state, [])
    assert alerts == 0
    assert len(sent_alerts) == 0
    # State remains intact
    assert inc_key in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_new_episode_alerts_immediately_after_genuine_recovery(monkeypatch):
    inc_key = "localhost:5432/test_db:101:1"
    now_ts = int(monitor.time.time())
    state = {
        "stuck_dialogues": {
            inc_key: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 1,
                "incident_scope_kind": "reset",
                "incident_topic_id": None,
                "first_alerted_at": now_ts - 300,
                "last_alerted_at": now_ts - 300,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout): return FakeConn()
    async def fake_mode(c): return "reset"
    async def fake_max(c): return set()

    sent_alerts = []
    async def fake_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)
    monkeypatch.setattr(monitor, "send_alert", fake_send)

    # Step 1: Episode is resolved (Stage A returns empty)
    async def fake_empty(*args, **kwargs): return []
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_empty)

    await monitor.check_stuck_dialogues(apps, state, [])
    assert inc_key not in state["stuck_dialogues"]

    # Step 2: Later, a brand new stuck episode occurs for the same user
    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_new_episode = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 2,
            "first_unanswered_id": 20,
            "last_unanswered_id": 21,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=10),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "is_grace_elapsed": True,
        }
    ]
    async def fake_new_ep(*args, **kwargs): return mock_new_episode
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_new_ep)

    alerts = await monitor.check_stuck_dialogues(apps, state, [])
    # Must alert immediately because previous episode genuinely recovered and state was cleared!
    assert alerts == 1
    assert len(sent_alerts) == 1
    assert inc_key in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_shared_db_scanned_once_and_single_alert(monkeypatch):
    apps = [
        {"name": "tg_app1", "db_url": "postgresql://u:p@10.0.0.1:5432/shared_db", "token": "t1", "owner_ids": [1]},
        {"name": "max_app1", "db_url": "postgresql://u:p@10.0.0.1:5432/shared_db", "token": "t2", "owner_ids": [1]},
    ]
    scan_count = 0
    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes = [
        {
            "user_id": 303,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "reset",
            "incident_topic_id": None,
            "unanswered_count": 2,
            "first_unanswered_id": 1,
            "last_unanswered_id": 2,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=4),
            "is_grace_elapsed": True,
        }
    ]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout): return FakeConn()
    async def fake_mode(c): return "reset"
    async def fake_max(c): return set()

    async def fake_episodes(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return mock_episodes

    sent_alerts = []
    async def fake_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)
    monkeypatch.setattr(monitor, "send_alert", fake_send)

    state = {}
    alerts = await monitor.check_stuck_dialogues(apps, state, [])

    # Shared DB scanned exactly ONCE
    assert scan_count == 1
    # Exactly ONE alert sent
    assert alerts == 1
    assert len(sent_alerts) == 1
    # Alert text lists both bots sharing the DB
    assert "tg_app1" in sent_alerts[0]
    assert "max_app1" in sent_alerts[0]


@pytest.mark.asyncio
async def test_run_check_exit_code_isolation_scenarios(monkeypatch, tmp_path):
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    async def fake_nl(s): return []
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_nl)

    # Scenario 1: Only stuck dialogue -> exit 0
    apps = [{"name": "bot1", "status": "online", "db_url": "postgresql://u:p@localhost:5432/db", "token": "t", "owner_ids": [1]}]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)
    async def fake_empty_async(*args, **kwargs): return []
    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    async def mock_check_stuck_alert(a, s, all_issues):
        # Simulate stuck alert sent, but NO infra issues
        return 1

    monkeypatch.setattr(monitor, "check_stuck_dialogues", mock_check_stuck_alert)
    exit_code_clean = await monitor.run_check(False)
    assert exit_code_clean == 0

    # Scenario 2: Stuck dialogue + infra issue -> exit 1
    async def mock_check_stuck_with_infra(a, s, all_issues):
        all_issues.append(("bot1", ["DB: schema missing column"]))
        return 1

    monkeypatch.setattr(monitor, "check_stuck_dialogues", mock_check_stuck_with_infra)
    sent_infra_alerts = []
    async def fake_send(a, t):
        sent_infra_alerts.append(t)
        return True
    monkeypatch.setattr(monitor, "send_alert", fake_send)

    exit_code_infra = await monitor.run_check(False)
    assert exit_code_infra == 1
    assert len(sent_infra_alerts) == 1
    assert "Psychobots monitor detected issues" in sent_infra_alerts[0]


def test_backward_compatible_state_loading_and_saving(tmp_path):
    state_file = tmp_path / "monitor_state.json"
    legacy_content = '{"log_offsets": {"/var/log/app.log": 9999}, "alerts": {"key1": 123456}}'
    state_file.write_text(legacy_content)

    loaded = monitor.load_state(str(state_file))
    assert loaded["log_offsets"] == {"/var/log/app.log": 9999}
    assert loaded["alerts"] == {"key1": 123456}
    assert loaded["stuck_dialogues"] == {}

    # Mutate stuck_dialogues
    loaded["stuck_dialogues"]["db:1:1"] = {"user_id": 1}
    monitor.save_state(loaded, str(state_file))

    reloaded = monitor.load_state(str(state_file))
    assert reloaded["log_offsets"] == {"/var/log/app.log": 9999}
    assert reloaded["alerts"] == {"key1": 123456}
    assert reloaded["stuck_dialogues"]["db:1:1"] == {"user_id": 1}


@pytest.mark.asyncio
async def test_watchdog_is_strictly_read_only():
    # Verify that all SQL queries in the watchdog are SELECT only
    captured_queries = []

    class FakeConn:
        async def fetch(self, sql, *args):
            captured_queries.append(sql)
            return []
        async def fetchval(self, sql, *args):
            captured_queries.append(sql)
            return "public.max_bot_states"
        async def fetchrow(self, sql, *args):
            captured_queries.append(sql)
            return {"memory_mode": "reset", "preserve_topic_context": False}

    conn = FakeConn()
    await monitor.get_effective_memory_mode(conn)
    await monitor.get_suppressed_max_user_ids(conn)
    await monitor.fetch_stuck_dialogue_episodes(conn, "topic", 24, 180)
    await monitor.fetch_stuck_dialogue_episodes(conn, "reset", 0, 180)

    forbidden_mutations = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE "]
    for q in captured_queries:
        for verb in forbidden_mutations:
            assert verb not in q.upper(), f"Watchdog SQL must not contain mutation verb {verb}"


@pytest.mark.asyncio
async def test_stage_a_includes_over_horizon_episodes_and_recovery_preserves_them(monkeypatch):
    monkeypatch.setattr(monitor, "STUCK_MAX_AGE_HOURS", 24)
    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    inc_key = "localhost:5432/test_db:101:1"
    now_ts = int(monitor.time.time())

    state = {
        "stuck_dialogues": {
            inc_key: {
                "db_key": "localhost:5432/test_db",
                "user_id": 101,
                "dialogue_id": 1,
                "incident_scope_kind": "dialogue",
                "incident_topic_id": None,
                "first_alerted_at": now_ts - 100000,
                "last_alerted_at": now_ts - 100000,
                "last_count": 2,
                "first_id": 1,
                "last_id": 2,
            }
        }
    }
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    # Episode has last unanswered message 48 hours ago (over 24h horizon)
    class FakeConn:
        async def close(self): pass
        async def fetch(self, sql, *args):
            return [
                {
                    "user_id": 101,
                    "dialogue_id": 1,
                    "current_topic_id": None,
                    "unanswered_count": 2,
                    "first_unanswered_id": 1,
                    "last_unanswered_id": 2,
                    "first_unanswered_at": now_utc - monitor.timedelta(hours=50),
                    "last_unanswered_at": now_utc - monitor.timedelta(hours=48),
                    "is_grace_elapsed": True,
                }
            ]

    async def fake_connect(d, timeout): return FakeConn()
    async def fake_mode(c): return "reset"
    async def fake_max(c): return set()

    sent_alerts = []
    async def fake_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)
    monkeypatch.setattr(monitor, "send_alert", fake_send)

    # Run check
    alerts = await monitor.check_stuck_dialogues(apps, state, [])

    # 1. Stage B alert is suppressed due to max_age_hours=24
    assert alerts == 0
    assert len(sent_alerts) == 0

    # 2. Existing state entry is PRESERVED because episode is active in Stage A
    assert inc_key in state["stuck_dialogues"]


@pytest.mark.asyncio
async def test_reset_and_global_share_incident_identity_and_cooldown(monkeypatch):
    inc_key = "localhost:5432/test_db:101:1"
    now_ts = int(monitor.time.time())
    apps = [{"name": "bot", "db_url": "postgresql://u:p@localhost:5432/test_db", "token": "t", "owner_ids": [1]}]

    now_utc = monitor.datetime.now(monitor.timezone.utc).replace(tzinfo=None)
    mock_episodes_reset = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "dialogue",
            "incident_topic_id": None,
            "memory_mode": "reset",
            "unanswered_count": 2,
            "first_unanswered_id": 1,
            "last_unanswered_id": 2,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=10),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "is_grace_elapsed": True,
            "is_within_max_age": True,
        }
    ]

    class FakeConn:
        async def close(self): pass

    async def fake_connect(d, timeout): return FakeConn()
    async def fake_max(c): return set()

    sent_alerts = []
    async def fake_send(a, t):
        sent_alerts.append(t)
        return True

    monkeypatch.setattr(monitor.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)
    monkeypatch.setattr(monitor, "send_alert", fake_send)

    # Run 1: Mode is 'reset' -> alerts and stores cooldown with incident_scope_kind='dialogue'
    async def fake_mode_reset(c): return "reset"
    async def fake_ep_reset(*args, **kwargs): return mock_episodes_reset
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode_reset)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_ep_reset)

    state = {}
    alerts1 = await monitor.check_stuck_dialogues(apps, state, [])
    assert alerts1 == 1
    assert inc_key in state["stuck_dialogues"]
    assert state["stuck_dialogues"][inc_key]["incident_scope_kind"] == "dialogue"

    # Run 2: Memory mode switches to 'global', same dialogue remains stuck
    sent_alerts.clear()
    mock_episodes_global = [
        {
            "user_id": 101,
            "dialogue_id": 1,
            "current_topic_id": None,
            "incident_scope_kind": "dialogue",
            "incident_topic_id": None,
            "memory_mode": "global",
            "unanswered_count": 2,
            "first_unanswered_id": 1,
            "last_unanswered_id": 2,
            "first_unanswered_at": now_utc - monitor.timedelta(minutes=10),
            "last_unanswered_at": now_utc - monitor.timedelta(minutes=5),
            "is_grace_elapsed": True,
            "is_within_max_age": True,
        }
    ]
    async def fake_mode_global(c): return "global"
    async def fake_ep_global(*args, **kwargs): return mock_episodes_global
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode_global)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_ep_global)

    alerts2 = await monitor.check_stuck_dialogues(apps, state, [])
    # Same incident -> cooldown retained, duplicate alert NOT sent
    assert alerts2 == 0
    assert len(sent_alerts) == 0
    # State is NOT deleted by recovery
    assert inc_key in state["stuck_dialogues"]

    # Run 3: Switch back from global to reset -> cooldown still active and state preserved
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode_reset)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_ep_reset)
    alerts3 = await monitor.check_stuck_dialogues(apps, state, [])
    assert alerts3 == 0
    assert inc_key in state["stuck_dialogues"]


# =====================================================================
# B1-F.1: PostgreSQL-only Backend Filtering, Classifier & Sanitization Tests
# =====================================================================


def test_classify_and_parse_db_url_supported_schemes():
    # 1. postgresql://
    status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(
        "postgresql://myuser:mypass@localhost:5432/mydb"
    )
    assert status == "supported"
    assert safe_key == "localhost:5432/mydb"
    assert dsn == "postgresql://myuser:mypass@localhost:5432/mydb"
    assert reason is None

    # 2. postgres://
    status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(
        "postgres://myuser:mypass@localhost:5432/mydb"
    )
    assert status == "supported"
    assert safe_key == "localhost:5432/mydb"
    assert dsn == "postgres://myuser:mypass@localhost:5432/mydb"
    assert reason is None

    # 3. postgresql+asyncpg:// -> normalized to postgresql://
    status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(
        "postgresql+asyncpg://myuser:mypass@localhost:5432/mydb"
    )
    assert status == "supported"
    assert safe_key == "localhost:5432/mydb"
    assert dsn == "postgresql://myuser:mypass@localhost:5432/mydb"
    assert reason is None

    # 4. mixed-case POSTGRESQL+ASYNCPG:// -> normalized to postgresql://
    status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(
        "POSTGRESQL+ASYNCPG://myuser:mypass@localhost:5432/mydb"
    )
    assert status == "supported"
    assert safe_key == "localhost:5432/mydb"
    assert dsn == "postgresql://myuser:mypass@localhost:5432/mydb"
    assert reason is None

    # 5. mixed-case PostgreSQL:// -> normalized
    status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(
        "PostgreSQL://myuser:mypass@localhost:5432/mydb"
    )
    assert status == "supported"
    assert safe_key == "localhost:5432/mydb"
    assert dsn == "postgresql://myuser:mypass@localhost:5432/mydb"
    assert reason is None


def test_classify_and_parse_db_url_unsupported_pg_driver():
    for scheme in ("postgresql+psycopg", "postgresql+psycopg2", "postgresql+unknown", "POSTGRESQL+PSYCOPG"):
        url = f"{scheme}://user:pass@localhost:5432/mydb"
        status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(url)
        assert status == "malformed_or_unsupported_pg"
        assert safe_key is None
        assert dsn is None
        assert reason == "unsupported PostgreSQL driver scheme"


def test_classify_and_parse_db_url_non_postgres_backends():
    non_pg_urls = [
        "sqlite:///path/to/db.sqlite",
        "sqlite+aiosqlite:///./darimiru_bot.db",
        "mysql://user:pass@localhost/mydb",
        "mysql+aiomysql://user:pass@localhost/mydb",
        "",
        None,
    ]
    for url in non_pg_urls:
        status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(url)
        assert status == "unsupported_backend"
        assert safe_key is None
        assert dsn is None
        assert reason is None


def test_classify_and_parse_db_url_malformed_postgres():
    # missing host
    for url in ("postgresql:///mydb", "postgresql://:5432/mydb"):
        status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(url)
        assert status == "malformed_or_unsupported_pg"
        assert dsn is None
        assert reason == "missing host"

    # invalid nonnumeric port
    status, _, dsn, reason = monitor.classify_and_parse_db_url("postgresql://localhost:not_a_port/mydb")
    assert status == "malformed_or_unsupported_pg"
    assert dsn is None
    assert reason == "invalid port"

    # port out of range (0 or >65535)
    status, _, dsn, reason = monitor.classify_and_parse_db_url("postgresql://localhost:0/mydb")
    assert status == "malformed_or_unsupported_pg"
    assert reason == "invalid port"

    status, _, dsn, reason = monitor.classify_and_parse_db_url("postgresql://localhost:70000/mydb")
    assert status == "malformed_or_unsupported_pg"
    assert reason == "invalid port"

    # missing database name
    for url in ("postgresql://localhost:5432/", "postgresql://localhost:5432"):
        status, _, dsn, reason = monitor.classify_and_parse_db_url(url)
        assert status == "malformed_or_unsupported_pg"
        assert dsn is None
        assert reason == "missing database name"

    # invalid URL syntax (e.g. unclosed bracket)
    status, _, dsn, reason = monitor.classify_and_parse_db_url("postgresql://[invalid_ipv6/mydb")
    assert status == "malformed_or_unsupported_pg"
    assert dsn is None
    assert reason == "invalid URL syntax"

    # obvious malformed PostgreSQL prefix
    for url in ("postgresql:", "postgres://", "postgresql+broken"):
        status, _, dsn, reason = monitor.classify_and_parse_db_url(url)
        assert status == "malformed_or_unsupported_pg"
        assert dsn is None
        assert reason in ("missing host", "invalid URL syntax", "unsupported PostgreSQL driver scheme")

    # obvious malformed PostgreSQL prefix without colon (e.g. postgresql//host/db)
    for url in ("postgresql//host/db", "postgres//host/db"):
        status, safe_key, dsn, reason = monitor.classify_and_parse_db_url(url)
        assert status == "malformed_or_unsupported_pg"
        assert safe_key is None
        assert dsn is None
        assert reason == "invalid URL syntax"

        # Verify when passed through grouping it emits sanitized infrastructure issue and does not group
        issues = []
        groups = monitor.group_apps_by_db([{"name": "bad_bot", "db_url": url}], malformed_issues=issues)
        assert len(groups) == 0
        assert any(name == "bad_bot" and "invalid URL syntax" in msgs[0] for name, msgs in issues)


def test_security_credential_bearing_malformed_url_sanitization():
    url = "postgresql://secret_user:super_secret_pwd@localhost:invalid_port/secret_db"
    issues = []
    groups = monitor.group_apps_by_db(
        [{"name": "broken_bot", "db_url": url}],
        malformed_issues=issues,
    )
    assert len(groups) == 0
    assert len(issues) == 1
    app_name, issue_list = issues[0]
    assert app_name == "broken_bot"
    assert issue_list == ["DB: malformed PostgreSQL DATABASE_URL: invalid port"]

    # Strict check: credentials and raw URL are completely absent from issue output
    issue_blob = str(issues)
    assert "secret_user" not in issue_blob
    assert "super_secret_pwd" not in issue_blob
    assert "secret_db" not in issue_blob
    assert url not in issue_blob


def test_make_safe_db_key_hardening():
    # Valid PostgreSQL and IPv6
    key = monitor.make_safe_db_key("postgresql://user:pass@[2001:db8::1]:5432/my_db")
    assert key == "[2001:db8::1]:5432/my_db"

    # SQLite must raise ValueError, NEVER produce fake localhost:5432 key
    with pytest.raises(ValueError) as exc_info:
        monitor.make_safe_db_key("sqlite+aiosqlite:///./darimiru_bot.db")
    assert "Cannot generate safe DB key" in str(exc_info.value)
    assert "localhost:5432" not in str(exc_info.value)

    # Malformed PostgreSQL must raise ValueError
    with pytest.raises(ValueError) as exc_info:
        monitor.make_safe_db_key("postgresql://localhost:invalid/my_db")
    assert "invalid port" in str(exc_info.value)


def test_group_apps_by_db_filtering_and_shared_db():
    apps = [
        {"name": "tg_bot", "db_url": "postgresql://u:p@localhost:5432/shared_db"},
        {"name": "max_bot", "db_url": "postgresql+asyncpg://u:p@localhost:5432/shared_db"},
        {"name": "darimiru_bot", "db_url": "sqlite+aiosqlite:///./darimiru_bot.db"},
        {"name": "broken_bot", "db_url": "postgresql+psycopg://u:p@localhost:5432/other_db"},
    ]
    issues = []
    groups = monitor.group_apps_by_db(apps, malformed_issues=issues)

    # Only shared_db is grouped
    assert len(groups) == 1
    assert "localhost:5432/shared_db" in groups
    assert len(groups["localhost:5432/shared_db"]) == 2
    assert groups["localhost:5432/shared_db"][0]["name"] == "tg_bot"
    assert groups["localhost:5432/shared_db"][1]["name"] == "max_bot"

    # Normalized DSN was attached to the apps in the group
    assert groups["localhost:5432/shared_db"][0]["normalized_db_url"] == "postgresql://u:p@localhost:5432/shared_db"
    assert groups["localhost:5432/shared_db"][1]["normalized_db_url"] == "postgresql://u:p@localhost:5432/shared_db"

    # darimiru_bot produces no issue
    assert not any(app == "darimiru_bot" for app, _ in issues)

    # broken_bot produces sanitized issue
    assert any(app == "broken_bot" and "unsupported PostgreSQL driver scheme" in msg[0] for app, msg in issues)


@pytest.mark.asyncio
async def test_mixed_execution_in_check_stuck_dialogues(monkeypatch):
    connected_dsns = []

    class FakeConn:
        async def close(self): pass

    async def fake_connect(dsn, timeout):
        connected_dsns.append(dsn)
        return FakeConn()

    async def fake_mode(c): return "reset"
    async def fake_episodes(*args, **kwargs): return []
    async def fake_max(c): return set()

    monkeypatch.setattr("asyncpg.connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", fake_mode)
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", fake_episodes)
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", fake_max)

    apps = [
        {"name": "tg_valid", "db_url": "postgresql+asyncpg://u:p@localhost:5432/valid_db", "token": "t", "owner_ids": [1]},
        {"name": "darimiru_bot", "db_url": "sqlite+aiosqlite:///./darimiru_bot.db", "token": "t", "owner_ids": [1]},
        {"name": "broken_bot", "db_url": "postgresql+psycopg://u:p@localhost:5432/broken_db", "token": "t", "owner_ids": [1]},
    ]

    all_issues = []
    state = {}
    alerts = await monitor.check_stuck_dialogues(apps, state, all_issues)
    assert alerts == 0

    # Prove only valid PostgreSQL called asyncpg.connect
    assert len(connected_dsns) == 1
    assert connected_dsns[0] == "postgresql://u:p@localhost:5432/valid_db"

    # Prove darimiru_bot created no B1-F issue and no asyncpg call
    assert not any(name == "darimiru_bot" for name, _ in all_issues)

    # Prove broken_bot created exactly sanitized issue
    broken_issues = [msgs for name, msgs in all_issues if name == "broken_bot"]
    assert len(broken_issues) == 1
    assert "unsupported PostgreSQL driver scheme" in broken_issues[0][0]

    # Prove no incident entries created for darimiru_bot or broken_bot
    assert len(state["stuck_dialogues"]) == 0


@pytest.mark.asyncio
async def test_asyncpg_connect_receives_normalized_dsn_actual_flow(monkeypatch):
    connected_dsns = []

    class FakeConn:
        async def close(self): pass

    async def fake_connect(dsn, timeout):
        connected_dsns.append(dsn)
        return FakeConn()

    monkeypatch.setattr("asyncpg.connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", lambda c: asyncio.sleep(0, result="reset"))
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", lambda *args, **kwargs: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", lambda c: asyncio.sleep(0, result=set()))

    # 1. postgresql+asyncpg://
    apps1 = [{"name": "bot1", "db_url": "postgresql+asyncpg://usr:pwd@localhost:5432/db1", "token": "t", "owner_ids": [1]}]
    await monitor.check_stuck_dialogues(apps1, {}, [])
    assert len(connected_dsns) == 1
    assert connected_dsns[0] == "postgresql://usr:pwd@localhost:5432/db1"

    # 2. mixed-case POSTGRESQL+ASYNCPG://
    apps2 = [{"name": "bot2", "db_url": "POSTGRESQL+ASYNCPG://usr:pwd@localhost:5432/db2", "token": "t", "owner_ids": [1]}]
    await monitor.check_stuck_dialogues(apps2, {}, [])
    assert len(connected_dsns) == 2
    assert connected_dsns[1] == "postgresql://usr:pwd@localhost:5432/db2"


@pytest.mark.asyncio
async def test_unsupported_pg_driver_never_reaches_asyncpg(monkeypatch):
    connected = False

    async def fake_connect(dsn, timeout):
        nonlocal connected
        connected = True
        return None

    monkeypatch.setattr("asyncpg.connect", fake_connect)

    apps = [{"name": "bot_psycopg", "db_url": "postgresql+psycopg://usr:pwd@localhost:5432/db", "token": "t", "owner_ids": [1]}]
    all_issues = []
    state = {}
    await monitor.check_stuck_dialogues(apps, state, all_issues)

    assert connected is False
    assert any(name == "bot_psycopg" and "unsupported PostgreSQL driver scheme" in msgs[0] for name, msgs in all_issues)


@pytest.mark.asyncio
async def test_b1f_connection_path_does_not_call_make_dsn(monkeypatch):
    def boom(url):
        raise RuntimeError("make_dsn should not be called by valid B1-F connection path!")

    monkeypatch.setattr(monitor, "make_dsn", boom)

    connected_dsns = []

    class FakeConn:
        async def close(self): pass

    async def fake_connect(dsn, timeout):
        connected_dsns.append(dsn)
        return FakeConn()

    monkeypatch.setattr("asyncpg.connect", fake_connect)
    monkeypatch.setattr(monitor, "get_effective_memory_mode", lambda c: asyncio.sleep(0, result="reset"))
    monkeypatch.setattr(monitor, "fetch_stuck_dialogue_episodes", lambda *args, **kwargs: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(monitor, "get_suppressed_max_user_ids", lambda c: asyncio.sleep(0, result=set()))

    apps = [{"name": "bot1", "db_url": "postgresql+asyncpg://usr:pwd@localhost:5432/db1", "token": "t", "owner_ids": [1]}]
    await monitor.check_stuck_dialogues(apps, {}, [])
    assert len(connected_dsns) == 1
    assert connected_dsns[0] == "postgresql://usr:pwd@localhost:5432/db1"






