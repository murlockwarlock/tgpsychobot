import asyncio
import os
import sys

os.environ.setdefault("BOT_TOKEN", "ci-test-token")
os.environ.setdefault("BOT_INSTANCE_ID", "tg_autobusbusbot_new")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("TESTING", "true")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from database import Base


@pytest_asyncio.fixture
async def test_session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield maker
    await engine.dispose()


@pytest.fixture(autouse=True)
def reset_aiogram_routers():
    for mod_name in ("handlers", "automation_admin", "admin_content_authoring", "event_admin_telegram"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "router"):
            mod.router._parent_router = None
    yield
    for mod_name in ("handlers", "automation_admin", "admin_content_authoring", "event_admin_telegram"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "router"):
            mod.router._parent_router = None


@pytest_asyncio.fixture(autouse=True)
async def cancel_leaked_async_tasks():
    yield
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

