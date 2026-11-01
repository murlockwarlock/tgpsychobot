import os
import re
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test_bot_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from max_messenger_bot.api import MaxApiError


def test_ecosystem_config_max_bot_token_wiring():
    repo_root = Path(__file__).parent.parent
    ecosystem_path = repo_root / "ecosystem.config.js"
    assert ecosystem_path.exists(), "ecosystem.config.js must exist"

    content = ecosystem_path.read_text(encoding="utf-8")

    # Verify tg_veraveda777_bot_legacy receives MAX_SE13639182_BOT_TOKEN
    assert 'name: "tg_veraveda777_bot_legacy"' in content
    # Find block for tg_veraveda777_bot_legacy
    legacy_block_match = re.search(
        r'name:\s*"tg_veraveda777_bot_legacy".*?"MAX_BOT_TOKEN":\s*process\.env\.MAX_SE13639182_BOT_TOKEN',
        content,
        re.DOTALL,
    )
    assert legacy_block_match is not None, "tg_veraveda777_bot_legacy must wire MAX_SE13639182_BOT_TOKEN"

    # Verify tg_yourself_way_bot_new receives MAX_ID519010411655_BOT_TOKEN
    assert 'name: "tg_yourself_way_bot_new"' in content
    yourself_block_match = re.search(
        r'name:\s*"tg_yourself_way_bot_new".*?"MAX_BOT_TOKEN":\s*process\.env\.MAX_ID519010411655_BOT_TOKEN',
        content,
        re.DOTALL,
    )
    assert yourself_block_match is not None, "tg_yourself_way_bot_new must wire MAX_ID519010411655_BOT_TOKEN"

    # Zero hardcoded tokens in ecosystem.config.js
    # Ensure no raw secret tokens are accidentally embedded
    assert not re.search(r'MAX_BOT_TOKEN":\s*"[0-9a-zA-Z_-]{20,}"', content)


@pytest.mark.asyncio
async def test_scheduler_patch_bot_send_message_missing_max_token_fails_closed():
    # When MAX_BOT_TOKEN is empty/missing, scheduler must fail closed (log warning, do not send)
    import scheduler

    class DummyBot:
        async def send_message(self, chat_id, text, **kwargs):
            return "tg_sent"

    bot = DummyBot()
    scheduler.patch_bot_send_message(bot)

    max_user_id = 100_000_000_001  # user_id >= 100_000_000_000 is MAX user

    with patch.dict(os.environ, {"MAX_BOT_TOKEN": ""}, clear=False):
        res = await bot.send_message(max_user_id, "Hello MAX")
        # Returns None, did not crash, did not send to wrong bot
        assert res is None


def _max_error_client(error):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.send_message.side_effect = error
    return client


@pytest.mark.asyncio
async def test_scheduler_treats_suspended_max_recipient_as_terminal_without_traceback(caplog):
    import scheduler

    class DummyBot:
        async def send_message(self, chat_id, text, **kwargs):
            return "tg_sent"

    bot = DummyBot()
    scheduler.patch_bot_send_message(bot)
    error = MaxApiError(
        "POST /messages failed: HTTP 403",
        status=403,
        error_code="chat.denied",
        payload={"code": "chat.denied", "message": "Key: error.dialog.suspended"},
    )
    with patch.dict(os.environ, {"MAX_BOT_TOKEN": "test-token"}, clear=False), \
         patch("max_messenger_bot.api.MaxApiClient", return_value=_max_error_client(error)), \
         caplog.at_level(logging.WARNING, logger="scheduler"):
        result = await bot.send_message(100_000_000_001, "Hello MAX")

    assert result is None
    assert any("terminal" in record.message.lower() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.asyncio
async def test_scheduler_does_not_suppress_other_max_forbidden_response(caplog):
    import scheduler

    class DummyBot:
        async def send_message(self, chat_id, text, **kwargs):
            return "tg_sent"

    bot = DummyBot()
    scheduler.patch_bot_send_message(bot)
    error = MaxApiError(
        "POST /messages failed: HTTP 403",
        status=403,
        error_code="auth.forbidden",
        payload={"code": "auth.forbidden", "message": "forbidden"},
    )
    with patch.dict(os.environ, {"MAX_BOT_TOKEN": "test-token"}, clear=False), \
         patch("max_messenger_bot.api.MaxApiClient", return_value=_max_error_client(error)), \
         caplog.at_level(logging.ERROR, logger="scheduler"):
        result = await bot.send_message(100_000_000_002, "Hello MAX")

    assert result is None
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.asyncio
async def test_scheduler_does_not_suppress_unknown_max_exception(caplog):
    import scheduler

    class DummyBot:
        async def send_message(self, chat_id, text, **kwargs):
            return "tg_sent"

    bot = DummyBot()
    scheduler.patch_bot_send_message(bot)
    with patch.dict(os.environ, {"MAX_BOT_TOKEN": "test-token"}, clear=False), \
         patch("max_messenger_bot.api.MaxApiClient", return_value=_max_error_client(RuntimeError("unexpected"))), \
         caplog.at_level(logging.ERROR, logger="scheduler"):
        result = await bot.send_message(100_000_000_003, "Hello MAX")

    assert result is None
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.asyncio
async def test_webhooks_send_msg_universal_missing_max_token_fails_closed():
    # In webhooks.py, send_msg_universal for user_id >= 100_000_000_000 must fail closed
    from webhooks import send_msg_universal

    dummy_bot = AsyncMock()
    max_user_id = 100_000_000_001

    with patch.dict(os.environ, {"MAX_BOT_TOKEN": ""}, clear=False):
        delivered = await send_msg_universal(dummy_bot, max_user_id, "Test")
        assert delivered is False
        assert dummy_bot.send_message.await_count == 0
