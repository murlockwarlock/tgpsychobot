import os
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
import scheduler


class _Session:
    def __init__(self, values, *, commit_error=None):
        self.values = values
        self.added = []
        self.commit_error = commit_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, model, object_id, *args, **kwargs):
        return self.values.get(model)

    async def execute(self, _statement):
        return SimpleNamespace(all=lambda: [])

    async def scalar(self, _statement):
        return self.values.get(handlers.MediaLibrary)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        if self.commit_error is not None:
            raise self.commit_error
        return None


class _SessionFactory:
    def __init__(self, *sessions):
        self.sessions = list(sessions)

    def __call__(self):
        return self.sessions.pop(0)


class _Message:
    def __init__(self, events):
        self.events = events
        self.edits = []

    async def edit_text(self, *args, **kwargs):
        self.events.append("edit")
        self.edits.append((args, kwargs))


class _Callback:
    def __init__(self, events, data="pay_yookassa_7"):
        self.data = data
        self.events = events
        self.answers = []
        self.from_user = SimpleNamespace(
            id=42,
            username="buyer",
            full_name="Buyer",
            first_name="Buyer",
        )
        self.message = _Message(events)
        self.bot = SimpleNamespace(
            get_me=AsyncMock(return_value=SimpleNamespace(username="test_bot")),
        )

    async def answer(self, *args, **kwargs):
        self.events.append("answered")
        self.answers.append((args, kwargs))


def _yookassa_config():
    return SimpleNamespace(
        yookassa_shop_id="shop-id",
        yookassa_secret_key="secret-key",
        privacy_policy_url=None,
        offer_agreement_url=None,
    )


def _plan():
    return SimpleNamespace(
        id=7,
        name="Тариф",
        price=10.0,
        is_trial=False,
        allow_auto_renewal=True,
    )


def _user():
    return SimpleNamespace(subscription=None, promo_codes=[])


def _payment():
    return SimpleNamespace(
        id="payment-1",
        status="pending",
        confirmation=SimpleNamespace(confirmation_url="https://yookassa.example/pay/1"),
        payment_method=None,
    )


async def _run_yookassa_callback(
    callback,
    provider_create,
    events,
    *,
    persistence_session=None,
):
    first = _Session({
        handlers.SubscriptionConfig: _yookassa_config(),
        handlers.SubscriptionPlan: _plan(),
        handlers.User: _user(),
    })
    second = persistence_session or _Session({})
    sessions = _SessionFactory(first, second)

    async def fake_to_thread(function, *args):
        events.append("provider_start")
        result = function(*args)
        events.append("provider_end")
        return result

    fake_payment = SimpleNamespace(create=provider_create)
    fake_configuration = SimpleNamespace(account_id=None, secret_key=None)
    with (
        patch.object(handlers, "async_session_maker", sessions),
        patch.object(handlers.asyncio, "to_thread", side_effect=fake_to_thread),
        patch.object(handlers, "Configuration", fake_configuration),
        patch.object(handlers, "Payment", fake_payment),
    ):
        await handlers.create_yookassa_invoice(callback, MagicMock())


async def _test_yookassa_callback_acknowledges_before_provider_io_and_preserves_link():
    events = []
    callback = _Callback(events)

    await _run_yookassa_callback(callback, lambda payload, key: _payment(), events)

    assert callback.answers == [((), {})]
    assert events.index("answered") < events.index("provider_start")
    keyboard = callback.message.edits[-1][1]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].url == "https://yookassa.example/pay/1"


async def _test_yookassa_ssl_failure_is_safe_for_user_and_reports_original_chain():
    events = []
    callback = _Callback(events)

    class FakeSSLError(Exception):
        pass

    ssl_error = FakeSSLError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF")
    sdk_error = AttributeError("'NoneType' object has no attribute 'status_code'")
    sdk_error.__cause__ = ssl_error

    with patch.object(
        handlers,
        "notify_admins_about_error",
        new_callable=AsyncMock,
        side_effect=RuntimeError("admin delivery unavailable"),
    ) as notify:
        await _run_yookassa_callback(callback, lambda payload, key: (_ for _ in ()).throw(sdk_error), events)

    assert callback.answers == [((), {})]
    assert events.index("answered") < events.index("provider_start")
    assert "платёжным сервисом" in callback.message.edits[-1][0][0]
    assert "NoneType" not in callback.message.edits[-1][0][0]
    assert notify.await_count == 1
    assert notify.await_args.kwargs["exception"] is sdk_error
    assert notify.await_args.kwargs["stage"] == "create_payment"


async def _test_yookassa_persistence_failure_keeps_created_payment_and_is_local_error():
    events = []
    callback = _Callback(events)
    provider_create = MagicMock(return_value=_payment())
    persistence_error = RuntimeError("database commit failed")

    with patch.object(handlers, "_safe_notify_admins_about_error", AsyncMock()) as notify:
        await _run_yookassa_callback(
            callback,
            provider_create,
            events,
            persistence_session=_Session({}, commit_error=persistence_error),
        )

    assert provider_create.call_count == 1
    assert "платёжным сервисом" not in " ".join(
        str(args[0]) for args, _ in callback.message.edits if args
    )
    keyboard = callback.message.edits[-1][1]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].url == "https://yookassa.example/pay/1"
    assert notify.await_count == 1
    alert = notify.await_args.kwargs
    assert alert["provider"] == "YooKassa"
    assert alert["stage"] == "persist_payment"
    assert alert["classification_override"] == "application_internal"
    assert alert["extra"]["payment_id"] == "payment-1"
    assert alert["extra"]["provider_status"] == "SUCCESS"


async def _test_yookassa_presentation_failure_reuses_same_payment_link():
    events = []

    class PresentationMessage(_Message):
        def __init__(self, message_events):
            super().__init__(message_events)
            self.fallbacks = []

        async def edit_text(self, *args, **kwargs):
            self.events.append("edit_failed")
            raise RuntimeError("Telegram edit failed")

        async def answer(self, *args, **kwargs):
            self.events.append("fallback_answer")
            self.fallbacks.append((args, kwargs))

    callback = _Callback(events)
    callback.message = PresentationMessage(events)
    provider_create = MagicMock(return_value=_payment())

    with patch.object(handlers, "_safe_notify_admins_about_error", AsyncMock()) as notify:
        await _run_yookassa_callback(callback, provider_create, events)

    assert provider_create.call_count == 1
    assert callback.message.fallbacks
    fallback_keyboard = callback.message.fallbacks[0][1]["reply_markup"]
    assert fallback_keyboard.inline_keyboard[0][0].url == "https://yookassa.example/pay/1"
    alert = notify.await_args.kwargs
    assert alert["stage"] == "present_payment"
    assert alert["classification_override"] == "application_internal"
    assert alert["extra"]["provider_status"] == "SUCCESS"
    assert alert["extra"]["presentation_fallback"] == "SUCCESS"


async def _test_yookassa_provider_call_keeps_single_fixed_idempotency_key():
    events = []
    callback = _Callback(events)
    provider_create = MagicMock(return_value=_payment())

    with patch.object(handlers, "uuid4", return_value="fixed-idempotency-key"):
        await _run_yookassa_callback(callback, provider_create, events)

    assert provider_create.call_count == 1
    payload, idempotency_key = provider_create.call_args.args
    assert idempotency_key == "fixed-idempotency-key"
    assert payload["metadata"] == {"user_id": 42, "plan_id": 7}


async def _test_yookassa_recurring_provider_failure_does_not_consume_attempt_and_success_stays_success():
    sub = SimpleNamespace(
        id=8,
        user_id=42,
        payment_attempt_count=2,
        payment_method_id="pm-1",
    )
    plan = SimpleNamespace(id=7, name="Тариф")
    config = SimpleNamespace(
        yookassa_shop_id="shop-id",
        yookassa_secret_key="secret-key",
        notifications_enabled=False,
    )
    started_at = scheduler.datetime(2026, 8, 26, 12, 0, 0)
    fake_configuration = SimpleNamespace(account_id=None, secret_key=None)
    fake_bad_request = type("FakeBadRequestError", (Exception,), {})
    fake_forbidden = type("FakeForbiddenError", (Exception,), {})
    fake_internal = type("FakeInternalServerError", (Exception,), {})
    fake_rate_limit = type("FakeTooManyRequestsError", (Exception,), {})
    fake_unauthorized = type("FakeUnauthorizedError", (Exception,), {})
    fake_payment = SimpleNamespace(create=MagicMock())

    async def fail_to_thread(function, *args):
        raise ConnectionError("provider temporarily unavailable")

    with (
        patch.object(scheduler, "Configuration", fake_configuration),
        patch.object(scheduler, "Payment", fake_payment),
        patch.object(scheduler, "BadRequestError", fake_bad_request),
        patch.object(scheduler, "ForbiddenError", fake_forbidden),
        patch.object(scheduler, "InternalServerError", fake_internal),
        patch.object(scheduler, "TooManyRequestsError", fake_rate_limit),
        patch.object(scheduler, "UnauthorizedError", fake_unauthorized),
        patch.object(scheduler.asyncio, "to_thread", side_effect=fail_to_thread),
    ):
        failed = await scheduler.process_recurring_payment(
            MagicMock(), sub, plan, 10.0, config, started_at
        )

    assert failed[0] == "provider_error"
    assert sub.payment_attempt_count == 2

    async def succeed_to_thread(function, *args):
        return SimpleNamespace(id="payment-2", status="succeeded")

    with (
        patch.object(scheduler, "Configuration", fake_configuration),
        patch.object(scheduler, "Payment", fake_payment),
        patch.object(scheduler.asyncio, "to_thread", side_effect=succeed_to_thread),
    ):
        succeeded = await scheduler.process_recurring_payment(
            MagicMock(), sub, plan, 10.0, config, started_at
        )

    assert succeeded == (True, "payment-2", "succeeded", None)
    assert sub.payment_attempt_count == 2


async def _test_robokassa_recurring_request_uses_documented_fields_and_signature():
    captured = {}

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def text(self):
            return "OK42"

    class _HttpSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, data):
            captured["url"] = url
            captured["data"] = data
            return _Response()

    config = SimpleNamespace(
        robokassa_merchant_login="demo",
        robokassa_password_1="pass1",
    )
    plan = SimpleNamespace(name="Тариф")

    with patch.object(scheduler.aiohttp, "ClientSession", return_value=_HttpSession()):
        result = await scheduler.process_recurring_robokassa_payment(
            config, plan, 10.0, "41", 42
        )

    assert result is True
    assert captured["url"].endswith("/Merchant/Recurring")
    assert captured["data"]["MerchantLogin"] == "demo"
    assert captured["data"]["OutSum"] == "10.00"
    assert captured["data"]["InvoiceID"] == "42"
    assert captured["data"]["PreviousInvoiceID"] == "41"
    assert "InvId" not in captured["data"]
    assert captured["data"]["SignatureValue"] == "70aa371c7594b731aeda96ded889a048"


def test_yookassa_callback_acknowledges_before_provider_io_and_preserves_link():
    asyncio.run(_test_yookassa_callback_acknowledges_before_provider_io_and_preserves_link())


def test_yookassa_ssl_failure_is_safe_for_user_and_reports_original_chain():
    asyncio.run(_test_yookassa_ssl_failure_is_safe_for_user_and_reports_original_chain())


def test_yookassa_persistence_failure_keeps_created_payment_and_is_local_error():
    asyncio.run(_test_yookassa_persistence_failure_keeps_created_payment_and_is_local_error())


def test_yookassa_presentation_failure_reuses_same_payment_link():
    asyncio.run(_test_yookassa_presentation_failure_reuses_same_payment_link())


def test_yookassa_provider_call_keeps_single_fixed_idempotency_key():
    asyncio.run(_test_yookassa_provider_call_keeps_single_fixed_idempotency_key())


def test_yookassa_recurring_provider_failure_does_not_consume_attempt_and_success_stays_success():
    asyncio.run(_test_yookassa_recurring_provider_failure_does_not_consume_attempt_and_success_stays_success())


def test_robokassa_callback_failure_is_acknowledged_and_has_safe_user_response():
    async def scenario():
        events = []
        callback = _Callback(events, data="pay_robokassa_7")
        with (
            patch.object(
                handlers,
                "_create_robokassa_invoice_impl",
                AsyncMock(side_effect=RuntimeError("local invoice build failed")),
            ),
            patch.object(handlers, "_safe_notify_admins_about_error", AsyncMock()) as notify,
        ):
            await handlers.create_robokassa_invoice(callback, MagicMock())

        assert callback.answers == [((), {})]
        assert notify.await_count == 1
        assert "Не удалось подготовить ссылку" in callback.message.edits[-1][0][0]

    asyncio.run(scenario())


def test_card_ai_callback_is_acknowledged_before_provider_io_and_reports_safe_failure():
    async def scenario():
        events = []

        class CardMessage(_Message):
            async def delete(self):
                events.append("deleted")

            async def answer_photo(self, *args, **kwargs):
                events.append("photo")

            async def answer(self, *args, **kwargs):
                events.append(("user_message", args[0] if args else kwargs.get("text")))

        callback = _Callback(events, data="card_select_7")
        callback.message = CardMessage(events)
        fake_bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock())
        config_session = _Session({
            handlers.AIConfig: SimpleNamespace(provider="KIE", kie_model="gemini-2.5-flash"),
        })
        media_session = _Session({
            handlers.MediaLibrary: SimpleNamespace(
                file_name="card.jpg",
                description="Описание",
                file_id="file-7",
            )
        })

        async def provider_failure(*args, **kwargs):
            assert "answered" in events
            raise handlers.AIServiceError("provider timeout")

        with (
            patch.object(handlers, "async_session_maker", _SessionFactory(config_session, media_session)),
            patch.object(handlers, "_get_card_spread_state", AsyncMock(return_value={"pending_card_ids": [7]})),
            patch.object(handlers, "_advance_card_spread_after_selection", AsyncMock(return_value=[])),
            patch.object(handlers.ai_integration, "generate_response", side_effect=provider_failure),
            patch.object(handlers, "_report_ai_failure", AsyncMock()) as report,
        ):
            await handlers.process_card_selection(
                callback,
                fake_bot,
            )

        assert callback.answers == [((), {})]
        assert events.index("answered") < events.index("photo")
        assert "Не удалось получить интерпретацию" in fake_bot.send_message.await_args.kwargs["text"]
        assert report.await_count == 1

    asyncio.run(scenario())


def test_robokassa_recurring_request_uses_documented_fields_and_signature():
    asyncio.run(_test_robokassa_recurring_request_uses_documented_fields_and_signature())
