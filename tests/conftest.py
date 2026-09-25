import asyncio

import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def cancel_leaked_async_tasks():
    yield
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
