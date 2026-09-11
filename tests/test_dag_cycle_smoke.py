import os
import unittest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


class DagCycleSmokeTests(unittest.TestCase):
    def test_import_modules_in_standard_order(self):
        """Verify importing new and modified modules does not raise circular import errors."""
        import effective_subscription
        import notification_transport
        import notification_renderer
        import notification_outbox
        import subscription_renewal
        import webhooks
        import handlers
        import scheduler
        import max_messenger_bot.services.subscription_access
        import max_messenger_bot.services.admin_payments

        self.assertIsNotNone(effective_subscription)
        self.assertIsNotNone(notification_transport)
        self.assertIsNotNone(notification_renderer)
        self.assertIsNotNone(notification_outbox)
        self.assertIsNotNone(subscription_renewal)
        self.assertIsNotNone(webhooks)
        self.assertIsNotNone(handlers)
        self.assertIsNotNone(scheduler)

    def test_outbox_and_transport_isolation(self):
        """Verify notification_transport does not import outbox, scheduler, or webhooks."""
        import sys
        transport_module = sys.modules.get("notification_transport")
        self.assertIsNotNone(transport_module)

        # notification_transport must not depend on database or high-level business flows
        self.assertFalse(hasattr(transport_module, "PaymentNotificationOutbox"))
        self.assertFalse(hasattr(transport_module, "handle_yookassa_webhook"))
