from unittest.mock import Mock, patch

import telegram_client


def test_get_telegram_proxy_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("TELEGRAM_PROXY", raising=False)

    assert telegram_client.get_telegram_proxy() is None


def test_get_telegram_proxy_normalizes_wrapping_quotes(monkeypatch):
    monkeypatch.setenv("TELEGRAM_PROXY", " 'http://relay.example:3128' ")

    assert telegram_client.get_telegram_proxy() == "http://relay.example:3128"


def test_create_telegram_bot_uses_configured_proxy(monkeypatch):
    monkeypatch.setenv("TELEGRAM_PROXY", "http://relay.example:3128")
    session = Mock()
    bot = Mock()

    with patch.object(telegram_client, "AiohttpSession", return_value=session) as session_factory:
        with patch.object(telegram_client, "Bot", return_value=bot) as bot_factory:
            result = telegram_client.create_telegram_bot("1234567890:test-token")

    session_factory.assert_called_once_with(proxy="http://relay.example:3128")
    bot_factory.assert_called_once_with(token="1234567890:test-token", session=session, default=None)
    assert result is bot
