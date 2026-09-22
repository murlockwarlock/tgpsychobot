#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BOT_TOKEN=123456:test
export DATABASE_URL=sqlite+aiosqlite:///:memory:
export BASE_WEBHOOK_URL=https://example.invalid
export SERVER_IP=127.0.0.1
export PYTHONPATH="${AUTHORING_TEST_DEPS:-$PWD/.test-deps}:$PWD"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec /root/telegram_bots/venv/bin/python scripts/server_pytest.py -p pytest_asyncio.plugin -p pytest_timeout -q --asyncio-mode=auto "$@"
