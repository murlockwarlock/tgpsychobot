import asyncio
import hashlib
import json
import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy.ext.asyncio as sqlalchemy_asyncio
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramForbiddenError
from aiogram.methods import GetMe, SendMessage, TelegramMethod
from aiogram.types import Chat, Message, User as TgUser

os.environ.setdefault("BOT_TOKEN", "8346799080:AAF_TEST_AUTOBUS_BOT_TOKEN")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

original_create_async_engine = sqlalchemy_asyncio.create_async_engine


def _sqlite_compatible_engine(*args, **kwargs):
    kwargs.pop("pool_recycle", None)
    kwargs.pop("pool_use_lifo", None)
    return original_create_async_engine(*args, **kwargs)


with patch.object(sqlalchemy_asyncio, "create_async_engine", _sqlite_compatible_engine):
    import database
    from database import (
        BotGeneralConfig,
        PaymentNotificationOutbox,
        SubscriptionConfig,
        User,
        UserSubscription,
        async_session_maker,
        init_db,
    )
    import monitor_psychobots as monitor
    from notification_outbox import (
        _send_admin_terminal_failure_alert,
        dispatch_outbox_by_key,
        enqueue_outbox_event,
        process_payment_notification_outbox,
    )
    import notification_routing as nr
    from notification_routing import EventCategory, send_admin_notification
    from notification_transport import (
        NotificationDeliveryResult,
        is_bot_blocked_reason,
        send_notification_transport_result,
    )


class RecordingSession(BaseSession):
    """
    Aiogram 3 recording session that performs real Pydantic validation on methods
    and captures all outbound requests at the HTTP transport boundary.
    """

    def __init__(self, fail_with: Exception | None = None):
        super().__init__()
        self.requests: list[tuple[str, int, SendMessage]] = []
        self.fail_with = fail_with

    async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None):
        if self.fail_with is not None:
            raise self.fail_with
        if isinstance(method, GetMe):
            from notification_routing import load_bot_registry
            registry = load_bot_registry()
            username = "TestBot"
            for inst in registry.get("instances", []):
                if inst.get("platform_id") == bot.id:
                    username = inst.get("username", "TestBot")
                    break
            return TgUser(id=bot.id, is_bot=True, first_name="TestBot", username=username)
        if isinstance(method, SendMessage):
            self.requests.append((bot.token, bot.id, method))
            return Message(
                message_id=9901,
                date=datetime.now(),
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=TgUser(id=bot.id, is_bot=True, first_name="TestBot"),
                text=method.text,
            )
        return None

    async def stream_content(self, url: str, timeout: int | None = None, chunk_size: int = 65536):
        yield b""

    async def close(self) -> None:
        pass


class OutboxTerminalAndBlockedUserAlertsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        async with async_session_maker() as session:
            from sqlalchemy import delete
            await session.execute(delete(PaymentNotificationOutbox))
            config = await session.get(SubscriptionConfig, 1)
            if not config:
                config = SubscriptionConfig(id=1, notifications_enabled=True)
                session.add(config)
            else:
                config.notifications_enabled = True
            await session.commit()

        self.autobus_token = "8346799080:AAF_VALID_AUTOBUS_MONITOR_TOKEN"
        self.recording_session = RecordingSession()

    def _make_bot(self, token: str | None = None):
        t = token or self.autobus_token
        return Bot(token=t, session=self.recording_session)

    # -------------------------------------------------------------------------
    # 1. User blocked Telegram bot -> error registered -> NO admin / autobus alerts
    # -------------------------------------------------------------------------
    async def test_01_user_blocked_bot_registers_failure_with_no_admin_alerts(self):
        """
        1. Пользователь заблокировал Telegram-бота → ошибка зарегистрирована
           → никаких админских уведомлений ни напрямую, ни через autobus.
           Не менять состояние подписок, платежей и других бизнес-объектов.
        """
        user_id = 111001
        key = f"scheduler:notification:{user_id}:block_test_01"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            session.add(User(id=user_id, first_name="Blocked User"))
            sub = UserSubscription(
                id=8001,
                user_id=user_id,
                plan_id=1,
                payment_provider="scheduler",
                start_date=now - timedelta(days=10),
                end_date=now + timedelta(days=20),
                auto_renewal=True,
            )
            session.add(sub)
            await session.commit()

            await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Hello user"},
            )
            await session.commit()

        mock_bot = self._make_bot()
        # Bot raises TelegramForbiddenError when attempting delivery
        mock_bot.send_message = AsyncMock(
            side_effect=TelegramForbiddenError(
                method=MagicMock(),
                message="Forbidden: bot was blocked by the user",
            )
        )

        with patch.dict(os.environ, {
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            delivered = await dispatch_outbox_by_key(mock_bot, key)

        self.assertFalse(delivered)

        # Check DB state
        async with async_session_maker() as session:
            from sqlalchemy import select
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertIsNotNone(row)
            self.assertEqual(row.status, "failed_terminal")
            self.assertEqual(row.last_error, "telegram_bot_blocked")
            self.assertEqual(row.attempts, 1)

            # Business subscription must remain untouched
            sub_after = await session.get(UserSubscription, 8001)
            self.assertTrue(sub_after.auto_renewal)
            self.assertEqual(sub_after.plan_id, 1)

        # Zero alert requests captured by the recording session
        self.assertEqual(len(self.recording_session.requests), 0)

    # -------------------------------------------------------------------------
    # 2. Block error in PM2 stderr traceback -> monitor filters it out
    # -------------------------------------------------------------------------
    def test_02_monitor_filters_user_blocked_python_traceback_in_stderr(self, tmp_path=None):
        """
        2. Ошибка блокировки попала в PM2 stderr в виде полного Python traceback
           → монитор правильно её отфильтровал.
        """
        traceback_chunk = (
            "Traceback (most recent call last):\n"
            '  File "/root/telegram_bots/notification_transport.py", line 82, in send_notification_transport_result\n'
            "    await bot.send_message(recipient_id, text)\n"
            '  File "/root/telegram_bots/.venv/lib/python3.12/site-packages/aiogram/client/bot.py", line 450, in send_message\n'
            "    return await self(SendMessage(chat_id=chat_id, text=text))\n"
            "aiogram.exceptions.TelegramForbiddenError: Telegram server says - Forbidden: bot was blocked by the user\n"
        )
        summaries = monitor.extract_log_error_summaries(traceback_chunk)
        self.assertEqual(summaries, [])

        # Also test with temp error log file
        import tempfile
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(traceback_chunk)
            f.flush()
            temp_path = f.name

        try:
            app = {"name": "tg_someonelikeyouai04_bot_legacy", "error_log": temp_path}
            state = {}
            issues = monitor.read_new_log_errors(app, state, include_existing=True)
            self.assertEqual(issues, [])
        finally:
            os.unlink(temp_path)

    # -------------------------------------------------------------------------
    # 3. User block AND real crash present simultaneously -> only block suppressed
    # -------------------------------------------------------------------------
    def test_03_monitor_suppresses_only_user_block_when_real_crash_present(self):
        """
        3. Одновременно присутствуют блокировка пользователя и настоящее падение приложения
           → подавлена только блокировка, авария доставлена.
        """
        chunk = (
            "Traceback (most recent call last):\n"
            '  File "main.py", line 100, in handle_msg\n'
            "    await bot.send_message(uid, 'Hi')\n"
            "aiogram.exceptions.TelegramForbiddenError: Telegram server says - Forbidden: bot was blocked by the user\n"
            "\n"
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 45, in run_loop\n'
            "    raise RuntimeError('Database connection pool exhausted')\n"
            "RuntimeError: Database connection pool exhausted\n"
        )
        summaries = monitor.extract_log_error_summaries(chunk)
        self.assertEqual(len(summaries), 1)
        self.assertIn("RuntimeError: Database connection pool exhausted", summaries[0])
        self.assertNotIn("TelegramForbiddenError", summaries[0])
        self.assertNotIn("blocked by the user", summaries[0])

    # -------------------------------------------------------------------------
    # 4. Outbox 1/7 to 6/7 -> no alerts, retry backed off normally
    # -------------------------------------------------------------------------
    async def test_04_outbox_attempts_1_to_6_no_alerts_and_retry_scheduled(self):
        """
        4. Outbox 1/7–6/7 → уведомлений нет, retry работает штатно.
        """
        user_id = 111004
        key = f"scheduler:notification:{user_id}:transient_retry_04"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Reminder"},
            )
            await session.commit()

        # Mock delivery that fails with a retryable transient error
        transient_fail_mock = AsyncMock(
            return_value=NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason="transport_delivery_timeout",
            )
        )

        mock_bot = self._make_bot()

        for attempt in range(1, 7):
            with patch.dict(os.environ, {
                "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
                "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
            }):
                res = await dispatch_outbox_by_key(mock_bot, key, deliver_func=transient_fail_mock)
                self.assertFalse(res)

            async with async_session_maker() as session:
                from sqlalchemy import select
                row = await session.scalar(
                    select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
                )
                self.assertEqual(row.status, "pending")
                self.assertEqual(row.attempts, attempt)
                self.assertGreater(row.next_retry_at, now)
                # Reset next_retry_at so next iteration can claim it
                row.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
                await session.commit()

            # Confirm 0 admin alerts sent through recording session
            self.assertEqual(len(self.recording_session.requests), 0)

    # -------------------------------------------------------------------------
    # 5. Outbox 7/7 -> exactly ONE service notification routed through Autobus
    # -------------------------------------------------------------------------
    async def test_05_outbox_7_of_7_sends_exactly_one_autobus_alert(self):
        """
        5. Outbox 7/7 → ровно одно служебное уведомление через autobus.
           Тестировать не только вызовы функций, но и конечную маршрутизацию
           уведомлений с проверкой сформированных Telegram methods.
        """
        user_id = 111005
        key = f"scheduler:notification:{user_id}:terminal_final_05"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            row = await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Final reminder"},
            )
            row.attempts = 6  # 7th attempt will be terminal
            row.next_retry_at = now - timedelta(seconds=5)
            await session.commit()

        transient_fail_mock = AsyncMock(
            return_value=NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason="network_timeout_504",
            )
        )

        mock_bot = self._make_bot()

        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            res = await dispatch_outbox_by_key(mock_bot, key, deliver_func=transient_fail_mock)
            self.assertFalse(res)

        async with async_session_maker() as session:
            from sqlalchemy import select
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertEqual(row.status, "failed_terminal")
            self.assertEqual(row.attempts, 7)
            self.assertEqual(row.max_attempts, 7)
            payload_dict = json.loads(row.event_payload_json)
            self.assertTrue(payload_dict.get("terminal_alert_sent"))

        # Verify real Telegram SendMessage method was dispatched strictly to Autobus
        self.assertGreater(len(self.recording_session.requests), 0)
        token, bot_id, method = self.recording_session.requests[0]
        self.assertEqual(bot_id, 8346799080)  # Must be Autobus bot platform_id
        self.assertIn("Outbox Terminal Failure", method.text)
        self.assertIn("7/7", method.text)
        self.assertIn("failed_terminal", method.text)

    # -------------------------------------------------------------------------
    # 6. Outbox failed_terminal on 1/7 due to permanent error -> NO alert
    # -------------------------------------------------------------------------
    async def test_06_outbox_failed_terminal_on_attempt_1_sends_no_alert(self):
        """
        6. Outbox failed_terminal на 1/7 из-за постоянной ошибки
           → не отправляется аварийное уведомление, состояние БД корректное.
        """
        user_id = 111006
        key = f"scheduler:notification:{user_id}:perm_err_06"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            row = await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Permanent fail"},
            )
            await session.commit()

        # Non-retryable error on attempt 1
        permanent_fail_mock = AsyncMock(
            return_value=NotificationDeliveryResult(
                delivered=False,
                retryable=False,
                reason="permanent_invalid_payload",
            )
        )

        mock_bot = self._make_bot()

        with patch.dict(os.environ, {
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            res = await dispatch_outbox_by_key(mock_bot, key, deliver_func=permanent_fail_mock)
            self.assertFalse(res)

        async with async_session_maker() as session:
            from sqlalchemy import select
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertEqual(row.status, "failed_terminal")
            self.assertEqual(row.attempts, 1)
            self.assertEqual(row.last_error, "permanent_invalid_payload")

        # Zero alerts dispatched because attempts (1) < max_attempts (7)
        self.assertEqual(len(self.recording_session.requests), 0)

    # -------------------------------------------------------------------------
    # 7. User blocked bot reaching terminal -> NO alert even at max attempts
    # -------------------------------------------------------------------------
    async def test_07_user_blocked_bot_terminal_sends_no_alert_even_at_max_attempts(self):
        """
        7. Пользователь заблокировал бота и доставка стала terminal
           → нет уведомления даже при максимальном числе попыток.
        """
        user_id = 111007
        key = f"scheduler:notification:{user_id}:blocked_max_07"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            row = await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Blocked final reminder"},
            )
            row.attempts = 6  # Reaching 7/7
            row.next_retry_at = now - timedelta(seconds=5)
            await session.commit()

        # Fails with user blocked reason
        blocked_fail_mock = AsyncMock(
            return_value=NotificationDeliveryResult(
                delivered=False,
                retryable=False,
                reason="telegram_bot_blocked",
            )
        )

        mock_bot = self._make_bot()

        with patch.dict(os.environ, {
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            res = await dispatch_outbox_by_key(mock_bot, key, deliver_func=blocked_fail_mock)
            self.assertFalse(res)

        async with async_session_maker() as session:
            from sqlalchemy import select
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertEqual(row.status, "failed_terminal")
            self.assertEqual(row.attempts, 7)
            self.assertEqual(row.last_error, "telegram_bot_blocked")

        # Must NOT send any alerts when reason is bot blocked, even at 7/7
        self.assertEqual(len(self.recording_session.requests), 0)

    # -------------------------------------------------------------------------
    # 8. Repeated worker execution creates NO duplicates
    # -------------------------------------------------------------------------
    async def test_08_repeated_outbox_execution_does_not_create_duplicate_alerts(self):
        """
        8. Повторное выполнение worker/monitor не создаёт дубли.
        """
        user_id = 111008
        key = f"scheduler:notification:{user_id}:dedup_test_08"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            row = await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Duplicate guard test"},
            )
            row.attempts = 6
            row.next_retry_at = now - timedelta(seconds=5)
            await session.commit()

        transient_fail_mock = AsyncMock(
            return_value=NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason="upstream_gateway_502",
            )
        )

        mock_bot = self._make_bot()

        # Run 1: triggers final failure alert
        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            res1 = await dispatch_outbox_by_key(mock_bot, key, deliver_func=transient_fail_mock)
            self.assertFalse(res1)

        first_count = len(self.recording_session.requests)
        self.assertGreater(first_count, 0)

        # Run 2: repeat execution via periodic outbox scanner
        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            # Outbox worker scan
            delivered_cnt = await process_payment_notification_outbox(mock_bot, deliver_func=transient_fail_mock)
            self.assertEqual(delivered_cnt, 0)

            # Direct dispatch attempt on same key
            res2 = await dispatch_outbox_by_key(mock_bot, key, deliver_func=transient_fail_mock)
            self.assertFalse(res2)

        # Zero additional alerts dispatched
        self.assertEqual(len(self.recording_session.requests), first_count)

    # -------------------------------------------------------------------------
    # 9. Real infrastructure errors are NOT suppressed by the filter
    # -------------------------------------------------------------------------
    def test_09_real_infrastructure_errors_are_not_suppressed(self):
        """
        9. Настоящие инфраструктурные ошибки не подавляются новым фильтром:
           - падение или остановка работающего бота;
           - недоступность критической БД;
           - серьёзные проблемы Telegram/MAX API и инфраструктуры;
           - сбой резервного копирования VPS.
        """
        real_errors = [
            "PM2: status is stopped",
            "PM2: status is errored",
            "DB: connection/query failed: Cannot connect to host localhost:5432",
            "DB: users missing columns: is_admin",
            "Telegram getMe failed: HTTP 502 Bad Gateway",
            "Telegram getMe failed: ClientConnectorError: Connection refused",
            "Proxy: Telegram check returned HTTP 502",
            "aiogram.exceptions.TelegramForbiddenError: Telegram server says - Forbidden: bot was kicked from the supergroup",
            "aiogram.exceptions.TelegramForbiddenError: Telegram server says - Forbidden: not enough rights to send text messages",
            "aiogram.exceptions.TelegramBadRequest: chat not found",
            "VPS backup: dump failed on /var/backups",
            "YooKassa Circuit Breaker: HTTP 401 shop authentication failed",
        ]

        for err in real_errors:
            self.assertFalse(
                monitor.is_bot_blocked_or_suspended_log(err),
                f"Infrastructure error should NOT be suppressed: {err}",
            )
            self.assertFalse(
                is_bot_blocked_reason(err),
                f"Infrastructure reason should NOT be suppressed: {err}",
            )

        # Confirm bot blocked indicators ARE correctly identified
        blocked_samples = [
            "aiogram.exceptions.TelegramForbiddenError: Telegram server says - Forbidden: bot was blocked by the user",
            "Forbidden: bot was blocked by the user",
            "telegram_bot_blocked",
            "max_chat_suspended",
            "MaxApiError: POST /messages failed: HTTP 403: chat.denied error.dialog.suspended",
            "chat.denied error.dialog.suspended",
        ]
        for b in blocked_samples:
            self.assertTrue(
                monitor.is_bot_blocked_or_suspended_log(b),
                f"Blocked indicator MUST be detected: {b}",
            )
            self.assertTrue(
                is_bot_blocked_reason(b),
                f"Blocked reason MUST be detected: {b}",
            )
