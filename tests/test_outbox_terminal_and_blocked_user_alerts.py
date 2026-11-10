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
            self.assertEqual(payload_dict.get("terminal_alert_status"), "sent")

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

    # -------------------------------------------------------------------------
    # 10. Outbox 7/7 -> Autobus down -> flag NOT set -> worker runs when up
    #     -> alert delivered -> flag set -> NO re-delivery to recipient user!
    # -------------------------------------------------------------------------
    async def test_10_terminal_alert_retried_when_autobus_initially_unavailable(self):
        """
        10. Разделение состояний «запланировано» и «доставлено»:
            - На 7/7 попытке Autobus недоступен -> флаг terminal_alert_sent НЕ выставляется в True.
            - terminal_alert_status сохраняется как 'pending'.
            - Worker при восстановлении Autobus доставляет служебное уведомление.
            - Пользователю сообщение повторно НЕ отправляется!
            - Повторный запуск worker не создает дублей.
        """
        user_id = 111010
        key = f"scheduler:notification:{user_id}:retry_autobus_down_10"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            row = await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Autobus outage test message"},
            )
            row.attempts = 6  # 7th attempt will reach terminal failure
            row.next_retry_at = now - timedelta(seconds=5)
            await session.commit()

        user_deliver_mock = AsyncMock(
            return_value=NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason="upstream_503_unavailable",
            )
        )

        mock_bot = self._make_bot()

        # Step 1: Attempt 7 occurs while Autobus / admin notification route is DOWN
        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)), \
           patch("notification_routing.send_admin_notification", AsyncMock(return_value=False)):
            res = await dispatch_outbox_by_key(mock_bot, key, deliver_func=user_deliver_mock)
            self.assertFalse(res)

        # Confirm user delivery mock was called exactly once for attempt 7
        self.assertEqual(user_deliver_mock.call_count, 1)

        # Inspect DB: must be failed_terminal, but terminal_alert_sent MUST NOT be True
        async with async_session_maker() as session:
            from sqlalchemy import select
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertEqual(row.status, "failed_terminal")
            self.assertEqual(row.attempts, 7)
            self.assertEqual(row.max_attempts, 7)
            payload_dict = json.loads(row.event_payload_json)
            # CRITICAL REQUIREMENT: Not considered sent because autobus failed!
            self.assertFalse(payload_dict.get("terminal_alert_sent", False))
            self.assertEqual(payload_dict.get("terminal_alert_status"), "pending")

            # Reset next_retry_at into the past so worker will process the retry immediately
            row.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
            await session.commit()

        # Confirm 0 admin alerts were dispatched through recording session so far
        self.assertEqual(len(self.recording_session.requests), 0)

        # Reset user mock to prove worker NEVER calls user delivery func
        user_deliver_mock.reset_mock()

        # Step 2: Autobus is now UP and worker runs
        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            delivered_cnt = await process_payment_notification_outbox(mock_bot, deliver_func=user_deliver_mock)
            self.assertEqual(delivered_cnt, 0)

        # CRITICAL REQUIREMENT: User delivery function was NEVER called!
        user_deliver_mock.assert_not_called()

        # CRITICAL REQUIREMENT: Exactly 1 alert broadcast was now delivered via Autobus (to its 3 registered admins)
        self.assertEqual(len(self.recording_session.requests), 3)
        for token, bot_id, method in self.recording_session.requests:
            self.assertEqual(bot_id, 8346799080)  # Autobus platform_id
            self.assertIn("Outbox Terminal Failure", method.text)
            self.assertIn("7/7", method.text)

        # Inspect DB: NOW terminal_alert_sent MUST be True and terminal_alert_status == 'sent'
        async with async_session_maker() as session:
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            payload_dict = json.loads(row.event_payload_json)
            self.assertTrue(payload_dict.get("terminal_alert_sent"))
            self.assertEqual(payload_dict.get("terminal_alert_status"), "sent")
            self.assertIsNotNone(payload_dict.get("terminal_alert_delivered_at"))

        # Step 3: Run worker again to ensure idempotency and no duplicate alerts
        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            await process_payment_notification_outbox(mock_bot, deliver_func=user_deliver_mock)

        user_deliver_mock.assert_not_called()
        self.assertEqual(len(self.recording_session.requests), 3)

    # -------------------------------------------------------------------------
    # 11. Concurrent workers atomic CAS prevents duplicate terminal failure alerts
    # -------------------------------------------------------------------------
    async def test_11_concurrent_workers_atomic_cas_prevents_duplicate_terminal_alerts(self):
        """
        11. Конкурентные worker'ы при повторной отправке terminal alert:
            CAS на базе claim_token исключает параллельные дубли.
            Ровно одно оповещение отправляется в Autobus.
        """
        user_id = 111011
        key = f"scheduler:notification:{user_id}:concurrency_alert_11"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            row = await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Concurrent alert test"},
            )
            row.status = "failed_terminal"
            row.attempts = 7
            row.max_attempts = 7
            row.last_error = "network_504_gateway_timeout"
            row.next_retry_at = now - timedelta(seconds=1)
            row.event_payload_json = json.dumps({
                "text": "Concurrent alert test",
                "terminal_alert_status": "pending",
                "terminal_alert_sent": False,
            })
            await session.commit()

        mock_bot = self._make_bot()
        user_mock = AsyncMock()

        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            # Run two workers concurrently
            res1, res2 = await asyncio.gather(
                process_payment_notification_outbox(mock_bot, deliver_func=user_mock),
                process_payment_notification_outbox(mock_bot, deliver_func=user_mock),
            )

        user_mock.assert_not_called()
        # Exactly 1 alert broadcast (to Autobus's 3 admins) captured, confirming atomic CAS prevented duplication
        self.assertEqual(len(self.recording_session.requests), 3)

        async with async_session_maker() as session:
            from sqlalchemy import select
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            payload_dict = json.loads(row.event_payload_json)
            self.assertTrue(payload_dict.get("terminal_alert_sent"))
            self.assertEqual(payload_dict.get("terminal_alert_status"), "sent")

    # -------------------------------------------------------------------------
    # 12. User blocked bot is NEVER alerted even on background worker scans
    # -------------------------------------------------------------------------
    async def test_12_user_blocked_bot_never_alerted_on_worker_scan(self):
        """
        12. Если пользователь заблокировал бота, даже при запуске периодического
            worker'а по записям failed_terminal уведомление никогда не отправляется.
        """
        user_id = 111012
        key = f"scheduler:notification:{user_id}:blocked_worker_scan_12"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            row = await enqueue_outbox_event(
                session,
                key,
                "scheduler",
                user_id,
                "subscription_reminder",
                {"text": "Blocked worker scan test"},
            )
            row.status = "failed_terminal"
            row.attempts = 7
            row.max_attempts = 7
            row.last_error = "telegram_bot_blocked"
            row.next_retry_at = now - timedelta(seconds=1)
            row.event_payload_json = json.dumps({
                "text": "Blocked worker scan test",
                "terminal_alert_status": "suppressed_blocked",
                "terminal_alert_sent": False,
            })
            await session.commit()

        mock_bot = self._make_bot()
        user_mock = AsyncMock()

        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": "tg_someonelikeyouai04_bot_legacy",
            "PM2_APP_NAME": "tg_someonelikeyouai04_bot_legacy",
            "TELEGRAM_AUTOBUSBUSBOT_TOKEN": self.autobus_token,
            "AUTOBUS_TELEGRAM_BOT_TOKEN": self.autobus_token,
        }), patch("telegram_client.create_telegram_bot", lambda token: self._make_bot(token)):
            await process_payment_notification_outbox(mock_bot, deliver_func=user_mock)

        user_mock.assert_not_called()
        self.assertEqual(len(self.recording_session.requests), 0)
