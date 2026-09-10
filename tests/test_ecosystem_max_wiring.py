import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test_bot_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest


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
