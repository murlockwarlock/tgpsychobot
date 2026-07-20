import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from keyboards import should_show_test_button
from metadata_export import metadata_export_entry
from profile_onboarding import configured_profile_fields, missing_profile_fields


class TestButtonVisibilityTests(unittest.TestCase):
    def test_disabled_test_is_hidden_for_every_user_role(self):
        config = SimpleNamespace(is_enabled=False)

        self.assertFalse(should_show_test_button(config))

    def test_enabled_test_is_visible(self):
        self.assertTrue(should_show_test_button(SimpleNamespace(is_enabled=True)))


class ProfileOnboardingTests(unittest.TestCase):
    def test_configured_fields_follow_independent_flags(self):
        config = SimpleNamespace(
            profile_collect_name=True,
            profile_collect_gender=False,
            profile_collect_age=True,
        )

        self.assertEqual(configured_profile_fields(config), ["name", "age"])

    def test_only_missing_enabled_fields_are_requested_in_order(self):
        config = SimpleNamespace(
            profile_collect_name=True,
            profile_collect_gender=True,
            profile_collect_age=True,
        )
        user = SimpleNamespace(name="Анна", gender=None, age=None)

        self.assertEqual(missing_profile_fields(config, user), ["gender", "age"])

    def test_all_fields_can_be_disabled(self):
        config = SimpleNamespace(
            profile_collect_name=False,
            profile_collect_gender=False,
            profile_collect_age=False,
        )
        user = SimpleNamespace(name=None, gender=None, age=None)

        self.assertEqual(missing_profile_fields(config, user), [])


class MetadataExportTests(unittest.TestCase):
    def setUp(self):
        self.user = SimpleNamespace(
            id=123456,
            username="telegram_login",
            name="Иван",
            first_name="Иван Telegram",
        )
        self.metadata = [{"saved_at": "2026-07-20T10:00:00+00:00", "data": {"score": 5}}]

    def test_regular_export_contains_telegram_identity(self):
        exported = metadata_export_entry(self.user, self.metadata, anonymize=False)

        self.assertEqual(exported["user_info"]["id"], 123456)
        self.assertEqual(exported["user_info"]["username"], "telegram_login")
        self.assertEqual(exported["user_info"]["name"], "Иван")

    def test_anonymized_export_omits_identity_keys_entirely(self):
        exported = metadata_export_entry(self.user, self.metadata, anonymize=True, anonymous_index=7)

        self.assertEqual(exported["user_info"], {"label": "user_7"})
        serialized_keys = set(exported["user_info"])
        self.assertTrue({"id", "username", "name"}.isdisjoint(serialized_keys))
        self.assertEqual(exported["metadata"], self.metadata)


if __name__ == "__main__":
    unittest.main()
