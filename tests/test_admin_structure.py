import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import keyboards
from database import Base, Content


def callback_values(markup):
    return {
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    }


class AdminStructureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()

    def test_main_admin_has_general_settings_and_automations(self):
        callbacks = callback_values(keyboards.admin_panel_keyboard())
        self.assertIn("admin_general_settings", callbacks)
        self.assertIn("automation_menu", callbacks)

    def test_profile_settings_are_not_in_test_menu(self):
        config = SimpleNamespace(
            is_enabled=True,
            show_progress=True,
            secret_test_enabled=True,
            formulas_enabled=False,
            interpretation_input_mode="all",
            separate_result_prompt_enabled=False,
            result_prompt_is_final=False,
        )
        callbacks = callback_values(keyboards.admin_test_menu_keyboard(config))

        self.assertFalse(any(value.startswith("admin_test_toggle_profile_") for value in callbacks))
        self.assertIn("edit_content_test_intro", callbacks)
        self.assertIn("edit_content_test_results", callbacks)
        self.assertIn("edit_content_secret_test_outro", callbacks)

    async def test_test_content_is_hidden_from_general_content_section(self):
        async with self.sessions() as session:
            session.add_all([
                Content(key="start_message", text_content="start", is_visible=True),
                Content(key="disclaimer", text_content="rules", is_visible=True),
                Content(key="about", button_title="О проекте", text_content="about", is_visible=True),
                Content(key="test_button", button_title="Пройти тест", text_content="test", is_visible=True),
                Content(key="test_intro", button_title="Приветствие теста", text_content="intro", is_visible=True),
                Content(key="test_results", button_title="Результаты теста", text_content="result", is_visible=True),
                Content(key="secret_test_outro", button_title="Финал теста", text_content="final", is_visible=True),
            ])
            await session.commit()

        with patch.object(keyboards, "async_session_maker", self.sessions):
            markup = await keyboards.content_management_keyboard()
        callbacks = callback_values(markup)

        self.assertIn("edit_content_about", callbacks)
        self.assertNotIn("edit_content_test_button", callbacks)
        self.assertNotIn("edit_content_test_intro", callbacks)
        self.assertNotIn("edit_content_test_results", callbacks)
        self.assertNotIn("edit_content_secret_test_outro", callbacks)
