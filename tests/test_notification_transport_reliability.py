import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from max_messenger_bot.api import MaxApiError
from notification_transport import send_notification_transport, clean_html_for_max, build_subscribe_keyboard


class NotificationTransportReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_delivery_success(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=True)

        res = await send_notification_transport(bot, 123456789, "Hello Telegram")
        self.assertTrue(res)
        bot.send_message.assert_awaited_once()

    async def test_telegram_delivery_failure_returns_false(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("Telegram Network error"))

        res = await send_notification_transport(bot, 123456789, "Hello Telegram")
        self.assertFalse(res)

    async def test_max_delivery_missing_token_fail_closed(self):
        bot = MagicMock()
        max_user_id = 100_000_000_555

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": ""}, clear=False):
            res = await send_notification_transport(bot, max_user_id, "Hello MAX")
            self.assertFalse(res)

    async def test_max_delivery_success(self):
        bot = MagicMock()
        max_user_id = 100_000_000_555

        mock_client_instance = MagicMock()
        mock_client_instance.send_message = AsyncMock(return_value=True)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": "valid_max_token"}, clear=False), \
             patch("max_messenger_bot.api.MaxApiClient", return_value=mock_client_instance):
            res = await send_notification_transport(bot, max_user_id, "<b>Hello</b> MAX")
            self.assertTrue(res)
            mock_client_instance.send_message.assert_awaited_once()
            # Clean text verification
            call_kwargs = mock_client_instance.send_message.call_args[1]
            self.assertEqual(call_kwargs["text"], "Hello MAX")
            self.assertEqual(call_kwargs["user_id"], 555)

    async def test_max_delivery_failure_returns_false(self):
        bot = MagicMock()
        max_user_id = 100_000_000_555

        mock_client_instance = MagicMock()
        mock_client_instance.send_message = AsyncMock(side_effect=Exception("MAX HTTP 500"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": "valid_max_token"}, clear=False), \
             patch("max_messenger_bot.api.MaxApiClient", return_value=mock_client_instance):
            res = await send_notification_transport(bot, max_user_id, "Hello MAX")
            self.assertFalse(res)

    async def test_suspended_max_recipient_is_terminal_delivery_failure(self):
        error = MaxApiError(
            "POST /messages failed: HTTP 403",
            status=403,
            error_code="chat.denied",
            payload={"code": "chat.denied", "message": "Key: error.dialog.suspended"},
        )
        self.assertTrue(error.is_suspended_recipient)

    async def test_suspended_max_delivery_is_not_retried_by_scheduler_outbox(self):
        from datetime import datetime

        import database
        from scheduler import _send_deduplicated_notification

        await database.init_db()
        max_user_id = 100_000_000_777
        error = MaxApiError(
            "POST /messages failed: HTTP 403",
            status=403,
            error_code="chat.denied",
            payload={"code": "chat.denied", "message": "Key: error.dialog.suspended"},
        )
        mock_client_instance = MagicMock()
        mock_client_instance.send_message = AsyncMock(side_effect=error)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=True)

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": "valid_max_token"}, clear=False), \
             patch("max_messenger_bot.api.MaxApiClient", return_value=mock_client_instance):
            first = await _send_deduplicated_notification(
                bot,
                max_user_id,
                "Reminder",
                "terminal-max-reminder",
                datetime.utcnow(),
            )
            second = await _send_deduplicated_notification(
                bot,
                max_user_id,
                "Reminder",
                "terminal-max-reminder",
                datetime.utcnow(),
            )

        self.assertFalse(first)
        self.assertFalse(second)
        mock_client_instance.send_message.assert_awaited_once()
        async with database.async_session_maker() as session:
            from sqlalchemy import select
            from database import PaymentNotificationOutbox

            row = await session.scalar(
                select(PaymentNotificationOutbox).where(
                    PaymentNotificationOutbox.unique_key == (
                        f"scheduler:notification:{max_user_id}:terminal-max-reminder"
                    )
                )
            )
        self.assertIsNotNone(row)
        self.assertEqual(row.status, "failed_terminal")
        self.assertEqual(row.last_error, "max_chat_suspended")

    async def test_transient_notification_failure_is_backed_off(self):
        from datetime import datetime

        import database
        from scheduler import _send_deduplicated_notification

        await database.init_db()
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("temporary network failure"))
        user_id = 778

        first = await _send_deduplicated_notification(
            bot,
            user_id,
            "Reminder",
            "transient-reminder",
            datetime.utcnow(),
        )
        second = await _send_deduplicated_notification(
            bot,
            user_id,
            "Reminder",
            "transient-reminder",
            datetime.utcnow(),
        )

        self.assertFalse(first)
        self.assertFalse(second)
        bot.send_message.assert_awaited_once()

    async def test_other_max_forbidden_response_is_not_suspended_recipient(self):
        error = MaxApiError(
            "POST /messages failed: HTTP 403",
            status=403,
            error_code="auth.forbidden",
            payload={"code": "auth.forbidden", "message": "forbidden"},
        )
        self.assertFalse(error.is_suspended_recipient)

    async def test_false_notify_sent_regression_in_scheduler(self):
        """On transport failure, _send_deduplicated_notification does not log NOTIFY_SENT and returns False."""
        from scheduler import _send_deduplicated_notification
        from types import SimpleNamespace

        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("Network failure"))
        sub = SimpleNamespace(id=42, user_id=123456)

        with patch("scheduler.plog") as mock_log:
            from datetime import datetime
            res = await _send_deduplicated_notification(bot, sub.user_id, "Test message", "test_key", datetime.utcnow(), logger=mock_log)
            self.assertFalse(res)
            # Ensure NOTIFY_SENT is never logged on failure
            for call in mock_log.info.call_args_list:
                msg = str(call)
                self.assertNotIn("NOTIFY_SENT", msg)
