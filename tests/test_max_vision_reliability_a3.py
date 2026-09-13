import asyncio
from datetime import datetime, timedelta
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ["BOT_TOKEN"] = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from alert_cooldown import AlertCooldown
from database import AIConfig, Base, Message as DBMessage, SubscriptionConfig, Topic, User
from max_messenger_bot import ai as max_ai
from max_messenger_bot.ai import (
    AIServiceError,
    InsufficientBalanceError,
    KIE_VISION_TRANSIENT_CLASSES,
    MaxVisionServiceError,
    _extract_status_and_provider_error_code,
    _validate_kie_json_response,
)
from max_messenger_bot.services import common as max_common


class TestMaxVisionReliabilityA3(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._orig_max_ai_sessions = max_ai.async_session_maker
        self._orig_max_common_sessions = max_common.async_session_maker
        max_ai.async_session_maker = self.sessions
        max_common.async_session_maker = self.sessions

        # Clear process-local cooldown dictionary
        max_common._VISION_ALERT_COOLDOWNS.clear()

        async with self.sessions() as session:
            self.topic = Topic(
                id=1,
                name="Психология",
                is_active=True,
                system_prompt="Тестовый системный промпт.",
            )
            self.user = User(
                id=7001,
                first_name="Тест",
                gender="male",
                age=30,
                current_dialogue_id=1,
                current_topic_id=1,
                metadata_json="{}",
            )
            self.ai_config = AIConfig(
                id=1,
                provider="KIE",
                vision_provider="KIE",
                vision_model="gemini-3-flash",
                kie_api_key="sk-kie-test-key",
                system_prompt="Системный промпт.",
                memory_mode="global",
            )
            self.sub_config = SubscriptionConfig(
                id=1,
                notifications_enabled=True,
            )
            session.add_all([self.topic, self.user, self.ai_config, self.sub_config])
            await session.commit()

    async def asyncTearDown(self):
        max_ai.async_session_maker = self._orig_max_ai_sessions
        max_common.async_session_maker = self._orig_max_common_sessions
        max_common._VISION_ALERT_COOLDOWNS.clear()
        await self.engine.dispose()

    # 1. configured preferred KIE vision model is attempted first
    async def test_01_configured_preferred_kie_model_attempted_first(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_model = "gemini-2.5-flash"
            await session.commit()

        models_called = []

        async def fake_analyze_kie(*args, **kwargs):
            # 4th arg is model
            model = args[3]
            models_called.append(model)
            return "Анализ успешен"

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            res = await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertEqual(res, "Анализ успешен")
        self.assertEqual(models_called, ["gemini-2.5-flash"])

    # 2. candidate list is de-duplicated
    async def test_02_candidate_list_is_deduplicated(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_model = "gemini-3-flash"
            await session.commit()

        models_called = []

        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            models_called.append(model)
            raise TimeoutError(f"Timeout on {model}")

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            with self.assertRaises(MaxVisionServiceError) as ctx:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        # Preferred gemini-3-flash followed by gemini-2.5-flash (no duplicate gemini-3-flash)
        self.assertEqual(models_called, ["gemini-3-flash", "gemini-2.5-flash"])
        attempt_models = [a["model"] for a in ctx.exception.attempts]
        self.assertEqual(attempt_models, ["gemini-3-flash", "gemini-2.5-flash"])
        self.assertEqual(len(attempt_models), len(set(attempt_models)))

    # 3. preferred gemini-3-flash transient failure -> gemini-2.5-flash attempted next
    async def test_03_preferred_gemini_3_flash_transient_failure_falls_back_to_gemini_2_5_flash(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_model = "gemini-3-flash"
            await session.commit()

        models_called = []

        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            models_called.append(model)
            if model == "gemini-3-flash":
                raise TimeoutError("KIE gemini-3-flash read timed out")
            return "Ответ от gemini-2.5-flash"

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            res = await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertEqual(models_called, ["gemini-3-flash", "gemini-2.5-flash"])
        self.assertEqual(res, "Ответ от gemini-2.5-flash")

    # 4. preferred gemini-2.5-flash transient failure -> gemini-3-flash attempted next
    async def test_04_preferred_gemini_2_5_flash_transient_failure_falls_back_to_gemini_3_flash(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_model = "gemini-2.5-flash"
            await session.commit()

        models_called = []

        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            models_called.append(model)
            if model == "gemini-2.5-flash":
                raise ConnectionResetError("KIE connection reset by peer")
            return "Ответ от gemini-3-flash"

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            res = await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertEqual(models_called, ["gemini-2.5-flash", "gemini-3-flash"])
        self.assertEqual(res, "Ответ от gemini-3-flash")

    # 5. alternate succeeds -> result returned -> zero admin alerts
    async def test_05_alternate_succeeds_returns_result_and_zero_admin_alerts(self):
        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            if model == "gemini-3-flash":
                raise TimeoutError("Transient timeout")
            return "Успешный ответ от резервной модели"

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie), \
             patch("error_reporting.notify_admins_about_error") as mock_notify:
            res = await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertEqual(res, "Успешный ответ от резервной модели")
        mock_notify.assert_not_called()

    # 6. HTTP 400 -> provider_rejection -> no alternate
    async def test_06_http_400_provider_rejection_no_alternate(self):
        calls = []

        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            calls.append(model)
            exc = AIServiceError("KIE multimodal request rejected: status=400 message=Bad Request")
            exc.status_code = 400
            exc.provider_error_code = 400
            raise exc

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            with self.assertRaises(MaxVisionServiceError) as ctx:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertEqual(len(calls), 1)
        self.assertEqual(ctx.exception.classification, "provider_rejection")
        self.assertFalse(ctx.exception.is_transient)
        self.assertEqual(len(ctx.exception.attempts), 1)

    # 7. HTTP 401 -> auth -> no alternate
    async def test_07_http_401_auth_no_alternate(self):
        calls = []

        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            calls.append(model)
            exc = AIServiceError("KIE API Error: Unauthorized")
            exc.status_code = 401
            exc.provider_error_code = 401
            raise exc

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            with self.assertRaises(MaxVisionServiceError) as ctx:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertEqual(len(calls), 1)
        self.assertEqual(ctx.exception.classification, "auth")
        self.assertFalse(ctx.exception.is_transient)
        self.assertEqual(len(ctx.exception.attempts), 1)

    # 8. HTTP 402 -> insufficient_balance_quota -> no alternate
    async def test_08_http_402_insufficient_balance_quota_no_alternate(self):
        calls = []

        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            calls.append(model)
            exc = InsufficientBalanceError("KIE API Error: Insufficient credits")
            exc.status_code = 402
            exc.provider_error_code = 402
            raise exc

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            with self.assertRaises(MaxVisionServiceError) as ctx:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertEqual(len(calls), 1)
        self.assertEqual(ctx.exception.classification, "insufficient_balance_quota")
        self.assertFalse(ctx.exception.is_transient)
        self.assertEqual(len(ctx.exception.attempts), 1)

    # 9. structured MaxVisionServiceError preserves required fields
    async def test_09_structured_max_vision_service_error_preserves_fields(self):
        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            if model == "gemini-3-flash":
                raise TimeoutError("Connection timed out")
            # Alternate fails with 502
            exc = AIServiceError("KIE gateway bad response: status=502 message=Bad Gateway")
            exc.status_code = 502
            exc.provider_error_code = "KIE_502"
            raise exc

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            with self.assertRaises(MaxVisionServiceError) as ctx:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        err = ctx.exception
        self.assertEqual(err.provider, "KIE")
        self.assertEqual(err.model, "gemini-2.5-flash")
        self.assertEqual(err.status_code, 502)
        self.assertEqual(err.provider_error_code, "KIE_502")
        self.assertEqual(err.classification, "provider_5xx")
        self.assertTrue(err.is_transient)
        self.assertEqual(len(err.attempts), 2)

        # Verify attempt 1
        att1 = err.attempts[0]
        self.assertEqual(att1["provider"], "KIE")
        self.assertEqual(att1["model"], "gemini-3-flash")
        self.assertEqual(att1["status"], "FAILED")
        self.assertEqual(att1["classification"], "timeout")
        self.assertEqual(att1["exception_class"], "TimeoutError")
        self.assertIn("Connection timed out", att1["error"])

        # Verify attempt 2
        att2 = err.attempts[1]
        self.assertEqual(att2["provider"], "KIE")
        self.assertEqual(att2["model"], "gemini-2.5-flash")
        self.assertEqual(att2["status"], "FAILED")
        self.assertEqual(att2["classification"], "provider_5xx")
        self.assertEqual(att2["exception_class"], "AIServiceError")
        self.assertIn("502", att2["error"])

    # 10. MaxVisionServiceError uses: raise ... from exc and __cause__ is preserved
    async def test_10_max_vision_service_error_uses_raise_from_exc_cause_preserved(self):
        terminal_exc = RuntimeError("Low level connection broke")

        async def fake_analyze_kie(*args, **kwargs):
            model = args[3]
            if model == "gemini-3-flash":
                raise TimeoutError("timeout")
            raise terminal_exc

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            with self.assertRaises(MaxVisionServiceError) as ctx:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        self.assertIs(ctx.exception.__cause__, terminal_exc)

    # 11. ai.py emits ZERO admin alerts
    async def test_11_ai_py_emits_zero_admin_alerts(self):
        async def fake_analyze_kie(*args, **kwargs):
            raise TimeoutError("Permanent or transient failure")

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie), \
             patch("error_reporting.notify_admins_about_error") as mock_notify:
            try:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")
            except MaxVisionServiceError:
                pass

        mock_notify.assert_not_called()

    # 12. common.py terminal failure: exactly one friendly MAX terminal error and one Telegram notifier invocation
    async def test_12_common_py_terminal_failure_one_user_error_and_one_notifier(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "think_mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        terminal_err = MaxVisionServiceError(
            "Terminal failure",
            provider="KIE",
            model="gemini-2.5-flash",
            classification="timeout",
            is_transient=True,
            attempts=[
                {"provider": "KIE", "model": "gemini-3-flash", "status": "FAILED", "classification": "timeout", "exception_class": "TimeoutError", "error": "timeout"},
                {"provider": "KIE", "model": "gemini-2.5-flash", "status": "FAILED", "classification": "timeout", "exception_class": "TimeoutError", "error": "timeout"},
            ],
            status_code=None,
            provider_error_code=None,
        )

        with patch("max_messenger_bot.ai.analyze_image", side_effect=terminal_err), \
             patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            await max_common.run_ai_dialogue_with_image(
                client=mock_client,
                chat_id=999,
                user_id=7001,
                image_bytes=b"fake_image",
                caption="Разбери фото",
            )

        # Friendly user error edited exactly once
        mock_client.edit_message.assert_called_once_with(
            "think_mid_123",
            text="Сервис анализа изображений временно недоступен.",
        )
        # Notifier invoked exactly once
        mock_notify.assert_called_once()

    # 13. notifier call parameters
    async def test_13_notifier_call_parameters(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        attempts = [
            {"provider": "KIE", "model": "gemini-3-flash", "status": "FAILED", "classification": "provider_5xx", "exception_class": "AIServiceError", "error": "500 error"},
        ]
        terminal_err = MaxVisionServiceError(
            "KIE 500 error",
            provider="KIE",
            model="gemini-3-flash",
            classification="provider_5xx",
            is_transient=True,
            attempts=attempts,
            status_code=500,
            provider_error_code=None,
        )

        with patch("max_messenger_bot.ai.analyze_image", side_effect=terminal_err), \
             patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            await max_common.run_ai_dialogue_with_image(
                client=mock_client,
                chat_id=8888,
                user_id=7001,
                image_bytes=b"fake_image",
                caption="Разбери",
            )

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        self.assertEqual(kwargs["stage"], "vision_analysis")
        self.assertEqual(kwargs["provider"], "KIE")
        self.assertEqual(kwargs["model"], "gemini-3-flash")
        self.assertEqual(kwargs["classification_override"], "provider_5xx")
        self.assertEqual(kwargs["provider_attempts"], attempts)
        self.assertIs(kwargs["exception"], terminal_err)
        self.assertIsNone(kwargs["user_id"])
        self.assertEqual(kwargs["extra"], {"max_user_id": 7001, "max_chat_id": 8888})

    # 14. same keyed alert immediately repeated -> second admin alert suppressed
    async def test_14_same_keyed_alert_immediately_repeated_suppressed(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        terminal_err = MaxVisionServiceError(
            "Terminal failure",
            provider="KIE",
            model="gemini-2.5-flash",
            classification="timeout",
            is_transient=True,
            attempts=[],
        )

        with patch("max_messenger_bot.ai.analyze_image", side_effect=terminal_err), \
             patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            # 1st failure
            await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")
            # 2nd failure immediately following
            await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")

        # Notifier called only once due to cooldown!
        self.assertEqual(mock_notify.call_count, 1)
        # But user message was edited both times
        self.assertEqual(mock_client.edit_message.call_count, 2)

    # 15. different classification -> not suppressed by previous key
    async def test_15_different_classification_not_suppressed(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        err_timeout = MaxVisionServiceError(
            "Timeout",
            provider="KIE",
            model="gemini-3-flash",
            classification="timeout",
            is_transient=True,
            attempts=[],
        )
        err_5xx = MaxVisionServiceError(
            "5xx",
            provider="KIE",
            model="gemini-3-flash",
            classification="provider_5xx",
            is_transient=True,
            attempts=[],
        )

        with patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            with patch("max_messenger_bot.ai.analyze_image", side_effect=err_timeout):
                await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")
            with patch("max_messenger_bot.ai.analyze_image", side_effect=err_5xx):
                await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")

        self.assertEqual(mock_notify.call_count, 2)

    # 16. different provider -> not suppressed by previous key
    async def test_16_different_provider_not_suppressed(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        err_kie = MaxVisionServiceError(
            "KIE timeout",
            provider="KIE",
            model="gemini-3-flash",
            classification="timeout",
            is_transient=True,
            attempts=[],
        )
        err_other = MaxVisionServiceError(
            "Other timeout",
            provider="OtherProvider",
            model="other-model",
            classification="timeout",
            is_transient=True,
            attempts=[],
        )

        with patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            with patch("max_messenger_bot.ai.analyze_image", side_effect=err_kie):
                await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")
            with patch("max_messenger_bot.ai.analyze_image", side_effect=err_other):
                await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")

        self.assertEqual(mock_notify.call_count, 2)

    # 17. cooldown duration is exactly 60 minutes
    async def test_17_cooldown_duration_is_exactly_60_minutes(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        terminal_err = MaxVisionServiceError(
            "Timeout",
            provider="KIE",
            model="gemini-3-flash",
            classification="timeout",
            is_transient=True,
            attempts=[],
        )

        with patch("max_messenger_bot.ai.analyze_image", side_effect=terminal_err), \
             patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")

        key = ("vision_analysis", "KIE", "timeout")
        self.assertIn(key, max_common._VISION_ALERT_COOLDOWNS)
        cd = max_common._VISION_ALERT_COOLDOWNS[key]
        self.assertEqual(cd.duration, timedelta(hours=1))
        self.assertEqual(cd.duration, timedelta(minutes=60))
        self.assertEqual(cd.duration.total_seconds(), 3600)

        # 59 minutes later -> still in cooldown
        now_59 = cd.last_sent_at + timedelta(minutes=59)
        self.assertFalse(cd.should_send(now=now_59))

        # 61 minutes later -> cooldown expired
        now_61 = cd.last_sent_at + timedelta(minutes=61)
        self.assertTrue(cd.should_send(now=now_61))

    # 18. generic non-KIE AIServiceError behavior in run_ai_dialogue_with_image remains unchanged
    async def test_18_generic_non_kie_ai_service_error_remains_unchanged(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        generic_err = AIServiceError("OpenAI generic vision error")

        with patch("max_messenger_bot.ai.analyze_image", side_effect=generic_err), \
             patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")

        # Friendly user error is sent
        mock_client.edit_message.assert_called_once_with(
            "mid_123",
            text="Сервис анализа изображений временно недоступен.",
        )
        # Generic AIServiceError does NOT trigger telegram admin notification
        mock_notify.assert_not_called()

    # 19. existing successful vision flow still works
    async def test_19_existing_successful_vision_flow_still_works(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})
        mock_client.delete_message = AsyncMock(return_value={"ok": True})

        async def fake_analyze_kie(*args, **kwargs):
            return "На изображении изображен красивый пейзаж."

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie), \
             patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            await max_common.run_ai_dialogue_with_image(
                client=mock_client,
                chat_id=999,
                user_id=7001,
                image_bytes=b"fake_image_bytes",
                caption="Опиши пейзаж",
            )

        # AI response was saved in DB
        async with self.sessions() as session:
            messages = (await session.scalars(Base.metadata.tables["messages"].select().where(
                Base.metadata.tables["messages"].c.user_id == 7001
            ))).all()
            self.assertTrue(len(messages) >= 2)

        # Thinking message was deleted or edited
        self.assertTrue(mock_client.edit_message.called or mock_client.delete_message.called)
        # Zero admin errors
        mock_notify.assert_not_called()

    # Additional check: invalid preferred model fails permanently without trying alternate
    async def test_20_unsupported_preferred_model_fails_permanently_without_alternate(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_model = "unsupported-model-xyz"
            await session.commit()

        calls = []

        async def fake_analyze_kie(*args, **kwargs):
            calls.append(args[3])
            return "OK"

        with patch.object(max_ai, "_analyze_kie", side_effect=fake_analyze_kie):
            with self.assertRaises(MaxVisionServiceError) as ctx:
                await max_ai.analyze_image(7001, b"fake_bytes", "Промпт")

        # Never called _analyze_kie because configuration validation failed before trying models
        self.assertEqual(len(calls), 0)
        self.assertEqual(ctx.exception.classification, "configuration")
        self.assertFalse(ctx.exception.is_transient)
        self.assertEqual(len(ctx.exception.attempts), 1)
        self.assertEqual(ctx.exception.attempts[0]["model"], "unsupported-model-xyz")

    # Additional check: notifications disabled prevents admin alert without consuming cooldown
    async def test_21_notifications_disabled_does_not_alert_nor_consume_cooldown(self):
        async with self.sessions() as session:
            sub = await session.get(SubscriptionConfig, 1)
            sub.notifications_enabled = False
            await session.commit()

        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})

        terminal_err = MaxVisionServiceError(
            "Timeout",
            provider="KIE",
            model="gemini-3-flash",
            classification="timeout",
            is_transient=True,
            attempts=[],
        )

        with patch("max_messenger_bot.ai.analyze_image", side_effect=terminal_err), \
             patch("max_messenger_bot.services.common.notify_admins_about_error", new_callable=AsyncMock) as mock_notify:
            await max_common.run_ai_dialogue_with_image(mock_client, 999, 7001, b"img", "cap")

        mock_notify.assert_not_called()
        key = ("vision_analysis", "KIE", "timeout")
        # Cooldown entry was not consumed
        self.assertNotIn(key, max_common._VISION_ALERT_COOLDOWNS)
