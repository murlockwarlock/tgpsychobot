import asyncio

import aiohttp
import pytest


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
