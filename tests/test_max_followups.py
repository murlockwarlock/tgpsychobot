from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import followups
import automation_admin
import admin_content_authoring
from database import Base, FollowupCampaign, FollowupDelivery, FollowupDeliveryAttempt, FollowupRun, FollowupStep, Topic, User
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot import app as max_app_module
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.services import admin_followups as max_admin_followups
from max_messenger_bot import storage as max_storage
from max_messenger_bot.storage import StorageBase
from max_messenger_bot.models import MAX_ID_OFFSET, IncomingMessage, Sender
from followup_admin_contract import classify_followup_callback


class FakeMaxClient:
    def __init__(self):
        self.sent = []
        self.answered = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return {"message": {"body": {"mid": f"max-{len(self.sent)}"}}}

    async def answer_callback(self, callback_id, **kwargs):
        self.answered.append((callback_id, kwargs))
        return {}


class _BoundaryResponse:
    status = 200
    content_type = "application/json"

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return json.dumps(self.payload)

    async def json(self):
        return self.payload


class _BoundarySession:
    def __init__(self, validate_body=None):
        self.requests = []
        self.closed = False
        self.validate_body = validate_body

    def request(self, method, url, *, params=None, json=None):
        path = url.rsplit("/", 1)[-1]
        request = {"method": method, "path": f"/{path}", "params": params or {}, "body": json}
        if path == "messages" and self.validate_body is not None:
            self.validate_body(json)
        if path == "answers" and self.validate_body is not None and json is not None and json.get("message") is not None:
            self.validate_body(json["message"])
        self.requests.append(request)
        if path == "messages":
            mid = f"max-boundary-{len(self.requests)}"
            return _BoundaryResponse({"message": {"body": {"mid": mid}}})
        return _BoundaryResponse({})


class ValidatingMaxClient(MaxApiClient):
    def __init__(self):
        super().__init__("test-token", "https://max.test")
        self.transport = _BoundarySession(self.validate_message_body)
        self._session = self.transport
        self.visible_followup_callbacks = {}

    @property
    def messages(self):
        return [request for request in self.transport.requests if request["path"] == "/messages"]

    def validate_message_body(self, body):
        assert isinstance(body, dict)
        assert isinstance(body.get("text"), str)
        attachments = body.get("attachments", [])
        assert isinstance(attachments, list)
        for attachment in attachments:
            assert isinstance(attachment, dict)
            assert attachment.get("type") == "inline_keyboard"
            payload = attachment.get("payload")
            assert isinstance(payload, dict)
            buttons = payload.get("buttons")
            assert isinstance(buttons, list)
            for row in buttons:
                assert isinstance(row, list)
                for button in row:
                    assert isinstance(button, dict)
                    assert isinstance(button.get("type"), str)
                    assert isinstance(button.get("text"), str)
                    if button["type"] == "callback":
                        assert isinstance(button.get("payload"), str)
                        assert button["payload"]

    def validate_latest_message(self):
        request = self.messages[-1]
        self.validate_message_body(request["body"])
        return request


def _keyboard_buttons(request):
    attachments = request["body"].get("attachments") or []
    return [
        button
        for attachment in attachments
        if attachment.get("type") == "inline_keyboard"
        for row in attachment.get("payload", {}).get("buttons", [])
        for button in row
    ]


def _latest_screen(client: ValidatingMaxClient):
    request = client.messages[-1]
    client.validate_message_body(request["body"])
    buttons = _keyboard_buttons(request)
    for button in buttons:
        if button.get("type") == "callback" and button.get("payload", "").startswith("admin_fu"):
            classification = classify_followup_callback(button["payload"])
            client.visible_followup_callbacks[button["payload"]] = classification
            assert classification is not None
    return request, buttons


async def _press_visible_button(app, client, raw_user_id, update_number, label_or_match):
    request, buttons = _latest_screen(client)
    if callable(label_or_match):
        assert any(label_or_match(button.get("text", "")) for button in buttons), [button.get("text") for button in buttons]
        button = next(button for button in buttons if label_or_match(button.get("text", "")))
    else:
        button = next(button for button in buttons if button.get("text") == label_or_match)
    assert button["type"] == "callback"
    await app.handle_update(
        {
            "update_type": "message_callback",
            "update_id": f"journey-{update_number}",
            "callback": {
                "callback_id": f"callback-{update_number}",
                "payload": button["payload"],
                "sender": {"user_id": raw_user_id, "name": "Admin"},
            },
            "message": {
                "recipient": {"chat_id": raw_user_id},
                "body": {"attachments": request["body"].get("attachments", [])},
            },
        }
    )
    assert not client.messages[-1]["body"].get("text", "").startswith("Произошла внутренняя ошибка")
    return client.messages[-1]


async def _send_admin_text(app, raw_user_id, update_number, text):
    await app.handle_update(
        {
            "update_type": "message_created",
            "update_id": f"journey-{update_number}",
            "message": {
                "recipient": {"chat_id": raw_user_id},
                "sender": {"user_id": raw_user_id, "name": "Admin"},
                "body": {"text": text},
            },
        }
    )


async def _send_user_text(app, raw_user_id, update_number, text):
    await app.handle_update(
        {
            "update_type": "message_created",
            "update_id": f"journey-{update_number}",
            "message": {
                "recipient": {"chat_id": raw_user_id},
                "sender": {"user_id": raw_user_id, "name": "User"},
                "body": {"text": text},
            },
        }
    )


@pytest_asyncio.fixture
async def followup_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)
    yield sessions
    await engine.dispose()


@pytest.mark.asyncio
async def test_max_static_followup_uses_max_keyboard_and_delivery_metadata(followup_db):
    user_id = MAX_ID_OFFSET + 101
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=None)
        campaign = FollowupCampaign(
            name="MAX chain",
            is_active=True,
            include_main_dialogue=True,
            quiet_start_minute=0,
            quiet_end_minute=0,
            jitter_min_seconds=0,
            jitter_max_seconds=0,
        )
        campaign.steps.append(
            FollowupStep(
                sort_order=0,
                delay_minutes=1,
                message_type="static",
                message_text="Напоминание\n[Продолжить](btn:continue)",
            )
        )
        session.add_all([user, campaign])
        await session.flush()
        session.add(
            FollowupRun(
                campaign_id=campaign.id,
                user_id=user_id,
                dialogue_id=1,
                topic_id=0,
                due_at=datetime.utcnow() - timedelta(minutes=2),
            )
        )
        await session.commit()

    client = ValidatingMaxClient()
    with patch.object(followups, "async_session_maker", followup_db):
        assert await followups.process_due_followups(
            followups.FollowupTransportRegistry(max_client=client)
        ) == 1

    assert len(client.messages) == 1
    payload = client.messages[0]["body"]
    assert client.messages[0]["params"]["user_id"] == 101
    assert payload["text"] == "Напоминание"
    assert payload["attachments"][0]["type"] == "inline_keyboard"
    assert payload["attachments"][0]["payload"]["buttons"][0][0]["text"] == "Продолжить"
    async with followup_db() as session:
        delivery = await session.scalar(select(FollowupDelivery))
    assert delivery.platform == "max"
    assert delivery.external_message_id == "max-boundary-1"
    assert delivery.telegram_message_id is None


@pytest.mark.asyncio
async def test_max_static_followup_button_uses_real_callback_route(followup_db, monkeypatch):
    user_id = MAX_ID_OFFSET + 111
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=None)
        campaign = FollowupCampaign(
            name="MAX buttons",
            is_active=True,
            include_main_dialogue=True,
            quiet_start_minute=0,
            quiet_end_minute=0,
            jitter_min_seconds=0,
            jitter_max_seconds=0,
        )
        campaign.steps.append(
            FollowupStep(
                sort_order=0,
                delay_minutes=1,
                message_type="static",
                message_text="Напоминание\n[Меню](btn:svc:menu)",
            )
        )
        session.add_all([user, campaign])
        await session.flush()
        session.add(
            FollowupRun(
                campaign_id=campaign.id,
                user_id=user_id,
                dialogue_id=1,
                topic_id=0,
                due_at=datetime.utcnow() - timedelta(minutes=2),
            )
        )
        await session.commit()

    client = ValidatingMaxClient()
    with patch.object(followups, "async_session_maker", followup_db):
        assert await followups.process_due_followups(
            followups.FollowupTransportRegistry(max_client=client)
        ) == 1

    button_payload = client.messages[0]["body"]["attachments"][0]["payload"]["buttons"][0][0]["payload"]
    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _false_async())
    monkeypatch.setattr(followups, "begin_user_activity", AsyncMock(return_value=None))
    monkeypatch.setattr(followups, "finalize_user_activity", AsyncMock())
    show_menu = AsyncMock()
    monkeypatch.setattr(max_app_module.common, "show_menu", show_menu)
    app = MaxBotApplication(client)
    await app.handle_update(
        {
            "update_type": "message_callback",
            "update_id": "max-static-button",
            "callback": {
                "callback_id": "cb-static-button",
                "payload": button_payload,
                "sender": {"user_id": 111, "name": "MAX"},
            },
            "message": {"recipient": {"chat_id": 111}, "body": {"attachments": client.messages[0]["body"]["attachments"]}},
        }
    )

    show_menu.assert_awaited_once_with(client, 111, user_id=user_id)
    assert client.transport.requests[-1]["path"] == "/answers"


@pytest.mark.asyncio
async def test_max_and_telegram_due_runs_use_only_their_platform_transport(followup_db):
    max_id = MAX_ID_OFFSET + 202
    async with followup_db() as session:
        campaign = FollowupCampaign(
            name="Shared",
            is_active=True,
            include_main_dialogue=True,
            quiet_start_minute=0,
            quiet_end_minute=0,
            jitter_min_seconds=0,
            jitter_max_seconds=0,
        )
        campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Hi"))
        session.add(campaign)
        await session.flush()
        for user_id in (303, max_id):
            session.add(User(id=user_id, first_name="User", current_dialogue_id=1, current_topic_id=None))
            session.add(FollowupRun(campaign_id=campaign.id, user_id=user_id, dialogue_id=1, topic_id=0, due_at=datetime.utcnow() - timedelta(minutes=2)))
        await session.commit()

    class TelegramClient:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text, **kwargs):
            self.sent.append((chat_id, text))
            return SimpleNamespace(message_id=7)

    telegram = TelegramClient()
    maximum = ValidatingMaxClient()
    with patch.object(followups, "async_session_maker", followup_db):
        assert await followups.process_due_followups(
            followups.FollowupTransportRegistry(telegram=telegram, max_client=maximum)
        ) == 2
    assert telegram.sent == [(303, "Hi")]
    assert [item["params"]["user_id"] for item in maximum.messages] == [202]


@pytest.mark.asyncio
async def test_max_ai_followup_keeps_buttons_and_followup_request_context(followup_db):
    user_id = MAX_ID_OFFSET + 606
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=7, current_topic_id=None)
        campaign = FollowupCampaign(name="AI", is_active=True, include_main_dialogue=True, quiet_start_minute=0, quiet_end_minute=0, jitter_min_seconds=0, jitter_max_seconds=0)
        campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="ai", ai_instruction="Верни пользователя в диалог."))
        session.add_all([user, campaign])
        await session.flush()
        session.add(FollowupRun(campaign_id=campaign.id, user_id=user_id, dialogue_id=7, topic_id=0, due_at=datetime.utcnow() - timedelta(minutes=1)))
        await session.commit()
    client = ValidatingMaxClient()
    with patch.object(followups, "async_session_maker", followup_db), patch(
        "ai_integration.get_ai_response",
        new=AsyncMock(return_value="Ответ\n[Открыть](btn:continue)"),
    ) as response:
        assert await followups.process_due_followups(followups.FollowupTransportRegistry(max_client=client)) == 1
    response.assert_awaited_once()
    assert response.await_args.kwargs["request_type"] == "followup"
    assert response.await_args.kwargs["track_user_activity"] is False
    assert client.messages[0]["body"]["text"] == "Ответ"
    assert client.messages[0]["body"]["attachments"][0]["payload"]["buttons"][0][0]["text"] == "Открыть"


@pytest.mark.asyncio
async def test_concurrent_due_workers_send_one_max_message(followup_db):
    user_id = MAX_ID_OFFSET + 707
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=None)
        campaign = FollowupCampaign(name="Concurrent", is_active=True, include_main_dialogue=True, quiet_start_minute=0, quiet_end_minute=0, jitter_min_seconds=0, jitter_max_seconds=0)
        campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Once"))
        session.add_all([user, campaign])
        await session.flush()
        session.add(FollowupRun(campaign_id=campaign.id, user_id=user_id, dialogue_id=1, topic_id=0, due_at=datetime.utcnow() - timedelta(minutes=1)))
        await session.commit()

    class BlockingClient(FakeMaxClient):
        def __init__(self):
            super().__init__()
            import asyncio
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def send_message(self, **kwargs):
            self.started.set()
            await self.release.wait()
            return await super().send_message(**kwargs)

    client = BlockingClient()
    with patch.object(followups, "async_session_maker", followup_db):
        first = __import__("asyncio").create_task(followups.process_due_followups(followups.FollowupTransportRegistry(max_client=client)))
        await client.started.wait()
        second = await followups.process_due_followups(followups.FollowupTransportRegistry(max_client=client))
        client.release.set()
        assert await first == 1
    assert second == 0
    assert len(client.sent) == 1


@pytest.mark.asyncio
async def test_max_activity_restarts_shared_chain_without_counting_followup_as_activity(followup_db):
    user_id = MAX_ID_OFFSET + 808
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=None)
        campaign = FollowupCampaign(name="Reset", is_active=True, include_main_dialogue=True, quiet_start_minute=0, quiet_end_minute=0, jitter_min_seconds=0, jitter_max_seconds=0)
        campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Later"))
        session.add_all([user, campaign])
        await session.commit()
    with patch.object(followups, "async_session_maker", followup_db):
        await followups.record_user_activity(user_id, dialogue_id=1, topic_id=None, activity_at=datetime.utcnow() - timedelta(minutes=2))
        ingress = await followups.begin_user_activity(user_id, dialogue_id=1, topic_id=None, activity_at=datetime.utcnow())
        await followups.finalize_user_activity(user_id, ingress, dialogue_id=1, topic_id=0)
    assert ingress.run_generations
    async with followup_db() as session:
        runs = (await session.scalars(select(FollowupRun).where(FollowupRun.user_id == user_id).order_by(FollowupRun.generation))).all()
    assert len(runs) == 1
    assert runs[0].generation == 2
    assert runs[0].status == "active"


@pytest.mark.asyncio
async def test_max_external_failure_keeps_shared_uncertain_semantics(followup_db):
    user_id = MAX_ID_OFFSET + 909
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=None)
        campaign = FollowupCampaign(name="Failure", is_active=True, include_main_dialogue=True, quiet_start_minute=0, quiet_end_minute=0, jitter_min_seconds=0, jitter_max_seconds=0)
        campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Fail"))
        session.add_all([user, campaign])
        await session.flush()
        session.add(FollowupRun(campaign_id=campaign.id, user_id=user_id, dialogue_id=1, topic_id=0, due_at=datetime.utcnow() - timedelta(minutes=1)))
        await session.commit()

    class FailingClient(FakeMaxClient):
        async def send_message(self, **kwargs):
            raise RuntimeError("temporary MAX transport failure")

    with patch.object(followups, "async_session_maker", followup_db):
        assert await followups.process_due_followups(followups.FollowupTransportRegistry(max_client=FailingClient())) == 0
    async with followup_db() as session:
        attempt = await session.scalar(select(FollowupDeliveryAttempt))
        run = await session.scalar(select(FollowupRun))
    assert attempt.status == "uncertain"
    assert run.status == "uncertain"


@pytest.mark.asyncio
async def test_max_activity_wrapper_calls_shared_ingress_and_finalization():
    client = SimpleNamespace()
    app = MaxBotApplication(client)
    app._handle_message_impl = AsyncMock()
    ingress = object()
    message = IncomingMessage(
        raw={},
        message_id="m1",
        chat_id=404,
        sender=Sender(user_id=MAX_ID_OFFSET + 404, username=None, first_name="User", last_name=None),
        text="hello",
    )
    with patch.object(app, "_begin_max_followup_activity", new=AsyncMock(return_value=ingress)) as begin, patch.object(
        app, "_finalize_max_followup_activity", new=AsyncMock()
    ) as finalize, patch(
        "max_messenger_bot.app.common.ensure_user", new=AsyncMock()
    ):
        await app.handle_message(message)
    begin.assert_awaited_once_with(MAX_ID_OFFSET + 404)
    finalize.assert_awaited_once_with(MAX_ID_OFFSET + 404, ingress)
    app._handle_message_impl.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_activity_ingress_runs_through_handle_update_boundary():
    client = SimpleNamespace()
    app = MaxBotApplication(client)
    app._handle_message_impl = AsyncMock()
    ingress = object()
    with patch.object(app, "_begin_max_followup_activity", new=AsyncMock(return_value=ingress)) as begin, patch.object(
        app, "_finalize_max_followup_activity", new=AsyncMock()
    ) as finalize, patch(
        "max_messenger_bot.app.common.ensure_user", new=AsyncMock()
    ):
        await app.handle_update(
            {
                "update_type": "message_created",
                "update_id": "max-activity-boundary",
                "message": {
                    "body": {"text": "Сообщение пользователя"},
                    "recipient": {"chat_id": 909},
                    "sender": {"user_id": 909, "name": "MAX"},
                },
            }
        )
    begin.assert_awaited_once_with(MAX_ID_OFFSET + 909)
    finalize.assert_awaited_once_with(MAX_ID_OFFSET + 909, ingress)
    app._handle_message_impl.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_topic_switch_finalizes_after_committed_scope(followup_db, monkeypatch):
    raw_user_id = 701
    user_id = MAX_ID_OFFSET + raw_user_id
    topic = Topic(name="New scope", is_active=True, show_in_list=True)
    campaign = FollowupCampaign(
        name="Topic race",
        is_active=True,
        all_topics=True,
        include_main_dialogue=True,
        quiet_start_minute=0,
        quiet_end_minute=0,
        jitter_min_seconds=0,
        jitter_max_seconds=0,
    )
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Later"))
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=None)
        session.add_all([user, topic, campaign])
        await session.flush()
        session.add(
            FollowupRun(
                campaign_id=campaign.id,
                user_id=user_id,
                dialogue_id=1,
                topic_id=0,
                due_at=datetime.utcnow(),
                status="active",
            )
        )
        await session.commit()

    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.topics_service, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(followups, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _false_async())

    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_select(client, chat_id, selected_user_id, selected_topic_id, states):
        started.set()
        await release.wait()
        async with followup_db() as session:
            selected_user = await session.get(User, selected_user_id)
            selected_user.current_topic_id = selected_topic_id
            await session.commit()
        await client.send_message(chat_id=chat_id, text="✅ Тема сохранена.")

    monkeypatch.setattr(max_app_module.topics_service, "select_topic", delayed_select)
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)

    await _send_user_text(app, raw_user_id, 1, "/topics")
    _latest_screen(client)
    switch = asyncio.create_task(_press_visible_button(app, client, raw_user_id, 2, "New scope"))
    await started.wait()
    assert not switch.done()
    release.set()
    await switch

    async with followup_db() as session:
        saved_user = await session.get(User, user_id)
        runs = (
            await session.scalars(
                select(FollowupRun)
                .where(FollowupRun.user_id == user_id)
                .order_by(FollowupRun.dialogue_id, FollowupRun.topic_id)
            )
        ).all()
    assert saved_user.current_topic_id == topic.id
    assert {(run.dialogue_id, run.topic_id, run.status) for run in runs} == {
        (1, 0, "cancelled"),
        (1, topic.id, "active"),
    }


@pytest.mark.asyncio
async def test_max_reset_topic_finalizes_after_committed_scope(followup_db, monkeypatch):
    raw_user_id = 712
    user_id = MAX_ID_OFFSET + raw_user_id
    topic = Topic(name="Current scope", is_active=True, show_in_list=True)
    campaign = FollowupCampaign(
        name="Reset topic race",
        is_active=True,
        all_topics=True,
        include_main_dialogue=True,
        quiet_start_minute=0,
        quiet_end_minute=0,
        jitter_min_seconds=0,
        jitter_max_seconds=0,
    )
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Later"))
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=7)
        session.add_all([user, topic, campaign])
        await session.flush()
        session.add(
            FollowupRun(
                campaign_id=campaign.id,
                user_id=user_id,
                dialogue_id=1,
                topic_id=7,
                due_at=datetime.utcnow(),
                status="active",
            )
        )
        await session.commit()

    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.topics_service, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(followups, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _false_async())

    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_reset(client, chat_id, selected_user_id, states):
        started.set()
        await release.wait()
        async with followup_db() as session:
            selected_user = await session.get(User, selected_user_id)
            selected_user.current_topic_id = None
            await session.commit()
        await client.send_message(chat_id=chat_id, text="✅ Основной диалог сохранён.")

    monkeypatch.setattr(max_app_module.topics_service, "reset_topic", delayed_reset)
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)

    await _send_user_text(app, raw_user_id, 1, "/topics")
    _latest_screen(client)
    reset = asyncio.create_task(_press_visible_button(app, client, raw_user_id, 2, "🏠 Перейти в основной диалог"))
    await started.wait()
    assert not reset.done()
    release.set()
    await reset

    async with followup_db() as session:
        saved_user = await session.get(User, user_id)
        runs = (
            await session.scalars(
                select(FollowupRun)
                .where(FollowupRun.user_id == user_id)
                .order_by(FollowupRun.dialogue_id, FollowupRun.topic_id)
            )
        ).all()
    assert saved_user.current_topic_id is None
    assert {(run.dialogue_id, run.topic_id, run.status) for run in runs} == {
        (1, 0, "active"),
        (1, 7, "cancelled"),
    }


@pytest.mark.asyncio
async def test_max_dialogue_reset_finalizes_after_committed_scope(followup_db, monkeypatch):
    raw_user_id = 702
    user_id = MAX_ID_OFFSET + raw_user_id
    topic = Topic(name="Reset scope", is_active=True, show_in_list=True)
    campaign = FollowupCampaign(
        name="Reset race",
        is_active=True,
        all_topics=True,
        include_main_dialogue=True,
        quiet_start_minute=0,
        quiet_end_minute=0,
        jitter_min_seconds=0,
        jitter_max_seconds=0,
    )
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Later"))
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=7)
        session.add_all([user, topic, campaign])
        await session.flush()
        session.add(
            FollowupRun(
                campaign_id=campaign.id,
                user_id=user_id,
                dialogue_id=1,
                topic_id=7,
                due_at=datetime.utcnow(),
                status="active",
            )
        )
        await session.commit()

    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(followups, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _false_async())

    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_reset(client, states, chat_id, selected_user_id, token, expected_dialogue_id, expected_topic_id):
        await states.clear(selected_user_id)
        started.set()
        await release.wait()
        async with followup_db() as session:
            selected_user = await session.get(User, selected_user_id)
            selected_user.current_dialogue_id += 1
            await session.commit()
        await client.send_message(chat_id=chat_id, text="✅ Диалог сброшен.")

    monkeypatch.setattr(max_app_module.common, "execute_dialogue_reset", delayed_reset)
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)

    await _send_user_text(app, raw_user_id, 1, "🗑️ Сбросить диалог")
    _latest_screen(client)
    reset = asyncio.create_task(_press_visible_button(app, client, raw_user_id, 2, "🗑️ Да, сбросить диалог"))
    await started.wait()
    assert not reset.done()
    release.set()
    await reset

    async with followup_db() as session:
        saved_user = await session.get(User, user_id)
        runs = (
            await session.scalars(
                select(FollowupRun)
                .where(FollowupRun.user_id == user_id)
                .order_by(FollowupRun.dialogue_id)
            )
        ).all()
    assert saved_user.current_dialogue_id == 2
    assert {(run.dialogue_id, run.topic_id, run.status) for run in runs} == {
        (1, 7, "cancelled"),
        (2, 7, "active"),
    }


@pytest.mark.asyncio
async def test_max_static_editor_back_cancels_input_and_reopen_preserves_text(followup_db, monkeypatch):
    raw_admin_id = 703
    admin_id = MAX_ID_OFFSET + raw_admin_id
    campaign = FollowupCampaign(name="Editor", include_main_dialogue=True)
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=5, message_type="static", message_text="Original"))
    async with followup_db() as session:
        session.add_all([User(id=admin_id, name="Admin", first_name="Admin", is_admin=True), campaign])
        await session.commit()

    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_app_module.common, "maybe_require_disclaimer", lambda *args, **kwargs: _false_async())
    monkeypatch.setattr(max_app_module.common, "ensure_access_before_chat", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_app_module.common, "run_ai_dialogue", AsyncMock())
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)
    update_number = 0

    async def send(text):
        nonlocal update_number
        update_number += 1
        await _send_admin_text(app, raw_admin_id, update_number, text)
        return _latest_screen(client)

    async def press(label):
        nonlocal update_number
        update_number += 1
        return await _press_visible_button(app, client, raw_admin_id, update_number, label)

    await send("/admin")
    await press("💬 Догоняющие сообщения")
    await press(lambda text: text.endswith("Editor"))
    await press(lambda text: text.startswith("🪜 Шаги"))
    await press(lambda text: text.startswith("1. через 5 мин."))
    await press("Текст сообщения")
    await press("Изменить: Сообщение")
    state = await app.states.get(admin_id)
    assert state is not None and state.state == "max_followup_step_text"
    await press("⬅️ Назад")
    assert await app.states.get(admin_id) is None

    await send("unrelated ordinary text")
    await asyncio.sleep(0)
    async with followup_db() as session:
        saved_step = await session.scalar(select(FollowupStep))
    assert saved_step.message_text == "Original"

    await press("⬅️ Назад")
    await press("Текст сообщения")
    assert "Original" in client.messages[-1]["body"]["text"]
    await press("Изменить: Сообщение")
    await send("Changed")
    await press("⬅️ Назад")
    await press("⬅️ Назад")
    await press(lambda text: text.startswith("1. через 5 мин."))
    await press("Текст сообщения")
    assert "Changed" in client.messages[-1]["body"]["text"]


@pytest.mark.asyncio
async def test_max_step_edit_rejects_cross_campaign_step(followup_db, monkeypatch):
    raw_admin_id = 704
    admin_id = MAX_ID_OFFSET + raw_admin_id
    campaign_a = FollowupCampaign(name="Campaign A")
    campaign_b = FollowupCampaign(name="Campaign B")
    step_b = FollowupStep(sort_order=0, delay_minutes=5, message_type="static", message_text="Protected")
    campaign_b.steps.append(step_b)
    async with followup_db() as session:
        session.add_all([User(id=admin_id, name="Admin", is_admin=True), campaign_a, campaign_b])
        await session.commit()
        campaign_a_id = campaign_a.id
        step_b_id = step_b.id

    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)

    await app.handle_update(
        {
            "update_type": "message_callback",
            "update_id": "cross-campaign-open",
            "callback": {
                "callback_id": "cross-campaign-open-callback",
                "payload": f"admin_fu_step_edit_{campaign_a_id}_{step_b_id}",
                "sender": {"user_id": raw_admin_id, "name": "Admin"},
            },
            "message": {"recipient": {"chat_id": raw_admin_id}, "body": {"attachments": []}},
        }
    )
    assert "Шаг не найден" in client.messages[-1]["body"]["text"]
    assert await app.states.get(admin_id) is None

    await app.states.set(
        admin_id,
        raw_admin_id,
        "max_followup_step_edit",
        {"campaign_id": campaign_a_id, "step_id": step_b_id},
    )
    await _send_admin_text(app, raw_admin_id, 705, "99\nHacked")
    async with followup_db() as session:
        saved_step = await session.get(FollowupStep, step_b_id)
    assert saved_step.delay_minutes == 5
    assert saved_step.message_text == "Protected"
    assert await app.states.get(admin_id) is None


@pytest.mark.asyncio
async def test_max_step_delete_reindexes_remaining_steps_through_visible_button(followup_db, monkeypatch):
    raw_admin_id = 706
    admin_id = MAX_ID_OFFSET + raw_admin_id
    campaign = FollowupCampaign(name="Delete visible", include_main_dialogue=True)
    campaign.steps.extend(
        [
            FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="First"),
            FollowupStep(sort_order=1, delay_minutes=2, message_type="static", message_text="Middle"),
            FollowupStep(sort_order=2, delay_minutes=3, message_type="static", message_text="Last"),
        ]
    )
    async with followup_db() as session:
        session.add_all([User(id=admin_id, name="Admin", is_admin=True), campaign])
        await session.commit()
        middle_id = campaign.steps[1].id

    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)
    update_number = 0

    async def send(text):
        nonlocal update_number
        update_number += 1
        await _send_admin_text(app, raw_admin_id, update_number, text)
        return _latest_screen(client)

    async def press(label):
        nonlocal update_number
        update_number += 1
        return await _press_visible_button(app, client, raw_admin_id, update_number, label)

    await send("/admin")
    await press("💬 Догоняющие сообщения")
    await press(lambda text: text.endswith("Delete visible"))
    await press(lambda text: text.startswith("🪜 Шаги"))
    await press(lambda text: text.startswith("2. через 2 мин."))
    await press("🗑 Удалить")

    async with followup_db() as session:
        remaining = (
            await session.scalars(
                select(FollowupStep).where(FollowupStep.campaign_id == campaign.id).order_by(FollowupStep.sort_order, FollowupStep.id)
            )
        ).all()
    assert [step.sort_order for step in remaining] == [0, 1]
    assert middle_id not in {step.id for step in remaining}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["claimed", "retryable", "uncertain", "delivered", "retry_exhausted"])
async def test_max_step_delete_blocks_all_protected_attempt_statuses(followup_db, status):
    campaign = FollowupCampaign(name=f"Protected {status}")
    step = FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Protected")
    campaign.steps.append(step)
    user = User(id=MAX_ID_OFFSET + 707, first_name="Admin", is_admin=True)
    async with followup_db() as session:
        session.add_all([campaign, user])
        await session.flush()
        run = FollowupRun(campaign_id=campaign.id, user_id=user.id, dialogue_id=1, topic_id=0, due_at=datetime.utcnow())
        session.add(run)
        await session.flush()
        attempt = FollowupDeliveryAttempt(
            run_id=run.id,
            step_id=step.id,
            step_index=0,
            generation=1,
            claim_token=f"token-{status}",
            status=status,
        )
        session.add(attempt)
        await session.commit()

    client = FakeMaxClient()
    with patch.object(max_admin_followups, "async_session_maker", followup_db):
        await max_admin_followups.delete_step(client, 1, campaign.id, step.id)
    async with followup_db() as session:
        assert await session.get(FollowupStep, step.id) is not None
        saved_attempt = await session.scalar(select(FollowupDeliveryAttempt))
    assert saved_attempt.status == status
    assert any("нельзя удалить" in message["text"] for message in client.sent)


@pytest.mark.asyncio
async def test_max_step_delete_blocks_delivery_history_and_preserves_it(followup_db):
    campaign = FollowupCampaign(name="Delivery history")
    step = FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Sent")
    campaign.steps.append(step)
    user = User(id=MAX_ID_OFFSET + 708, first_name="Admin", is_admin=True)
    async with followup_db() as session:
        session.add_all([campaign, user])
        await session.flush()
        run = FollowupRun(campaign_id=campaign.id, user_id=user.id, dialogue_id=1, topic_id=0, due_at=datetime.utcnow())
        session.add(run)
        await session.flush()
        delivery = FollowupDelivery(run_id=run.id, step_id=step.id, platform="max", external_message_id="history")
        session.add(delivery)
        await session.commit()

    client = FakeMaxClient()
    with patch.object(max_admin_followups, "async_session_maker", followup_db):
        await max_admin_followups.delete_step(client, 1, campaign.id, step.id)
    async with followup_db() as session:
        assert await session.get(FollowupStep, step.id) is not None
        saved_delivery = await session.scalar(select(FollowupDelivery))
    assert saved_delivery.external_message_id == "history"


@pytest.mark.asyncio
async def test_max_self_test_concurrent_visible_presses_send_once(followup_db, monkeypatch):
    raw_admin_id = 709
    admin_id = MAX_ID_OFFSET + raw_admin_id
    campaign = FollowupCampaign(
        name="Self test",
        is_active=True,
        include_main_dialogue=True,
        quiet_start_minute=0,
        quiet_end_minute=0,
        jitter_min_seconds=0,
        jitter_max_seconds=0,
    )
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="SELF_TEST_ONCE"))
    async with followup_db() as session:
        session.add_all([User(id=admin_id, name="Admin", is_admin=True, current_dialogue_id=1), campaign])
        await session.commit()

    for module in (max_app_module, max_storage, max_admin_followups):
        monkeypatch.setattr(module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)
    update_number = 0

    async def send(text):
        nonlocal update_number
        update_number += 1
        await _send_admin_text(app, raw_admin_id, update_number, text)
        return _latest_screen(client)

    async def press(label):
        nonlocal update_number
        update_number += 1
        return await _press_visible_button(app, client, raw_admin_id, update_number, label)

    await send("/admin")
    await press("💬 Догоняющие сообщения")
    await press(lambda text: text.endswith("Self test"))
    await press("🧪 Проверить на себе")
    request, buttons = _latest_screen(client)
    send_button = next(button for button in buttons if button.get("text", "").startswith("▶️"))
    payload = send_button["payload"]
    update_base = {
        "update_type": "message_callback",
        "callback": {
            "callback_id": "self-test-concurrent",
            "payload": payload,
            "sender": {"user_id": raw_admin_id, "name": "Admin"},
        },
        "message": {"recipient": {"chat_id": raw_admin_id}, "body": {"attachments": request["body"].get("attachments", [])}},
    }
    first = {**update_base, "update_id": "self-test-first"}
    second = {**update_base, "update_id": "self-test-second", "callback": {**update_base["callback"], "callback_id": "self-test-second-callback"}}
    await asyncio.gather(app.handle_update(first), app.handle_update(second))

    external = [message for message in client.messages if message["body"].get("text") == "SELF_TEST_ONCE"]
    assert len(external) == 1
    state = await app.states.get(admin_id)
    assert state is not None
    assert state.data["step_index"] == 1
    assert "Ручная проверка завершена" in client.messages[-1]["body"]["text"]


@pytest.mark.asyncio
async def test_max_empty_and_nonempty_stop_event_screens_match_telegram(followup_db, monkeypatch):
    raw_admin_id = 710
    admin_id = MAX_ID_OFFSET + raw_admin_id
    campaign = FollowupCampaign(name="Stop parity", include_main_dialogue=True)
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text="Text"))
    async with followup_db() as session:
        session.add_all([User(id=admin_id, name="Admin", is_admin=True), campaign])
        await session.commit()

    for module in (max_app_module, max_storage, max_admin_followups, automation_admin):
        monkeypatch.setattr(module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)
    update_number = 0

    async def send(text):
        nonlocal update_number
        update_number += 1
        await _send_admin_text(app, raw_admin_id, update_number, text)
        return _latest_screen(client)

    async def press(label):
        nonlocal update_number
        update_number += 1
        return await _press_visible_button(app, client, raw_admin_id, update_number, label)

    async def expected_telegram_labels():
        target = SimpleNamespace(
            data=f"followup_stop_events_edit_{campaign.id}",
            message=SimpleNamespace(edit_text=AsyncMock()),
        )
        state = SimpleNamespace(
            get_data=AsyncMock(return_value={}),
            set_state=AsyncMock(),
            update_data=AsyncMock(),
            clear=AsyncMock(),
        )
        await automation_admin.followup_stop_events_edit(target, state)
        markup = target.message.edit_text.await_args.kwargs["reply_markup"]
        return [button.text for row in markup.inline_keyboard for button in row]

    await send("/admin")
    await press("💬 Догоняющие сообщения")
    await press(lambda text: text.endswith("Stop parity"))
    await press("⚙️ Условия")
    await press("✏️ Изменить события остановки")
    _, buttons = _latest_screen(client)
    assert [button["text"] for button in buttons] == await expected_telegram_labels()
    await press("🧹 Очистить список")

    await press("✏️ Изменить события остановки")
    await send("PAYMENT_SUCCESS")
    await press("✏️ Изменить события остановки")
    _, buttons = _latest_screen(client)
    assert [button["text"] for button in buttons] == await expected_telegram_labels()


@pytest.mark.asyncio
async def test_max_step_input_rejects_plus_prefix_through_real_message_route(followup_db, monkeypatch):
    raw_admin_id = 711
    admin_id = MAX_ID_OFFSET + raw_admin_id
    campaign = FollowupCampaign(name="Numeric parity", include_main_dialogue=True)
    async with followup_db() as session:
        session.add_all([User(id=admin_id, name="Admin", is_admin=True), campaign])
        await session.commit()

    for module in (max_app_module, max_storage, max_admin_followups):
        monkeypatch.setattr(module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)
    update_number = 0

    async def send(text):
        nonlocal update_number
        update_number += 1
        await _send_admin_text(app, raw_admin_id, update_number, text)
        return _latest_screen(client)

    async def press(label):
        nonlocal update_number
        update_number += 1
        return await _press_visible_button(app, client, raw_admin_id, update_number, label)

    await send("/admin")
    await press("💬 Догоняющие сообщения")
    await press(lambda text: text.endswith("Numeric parity"))
    await press(lambda text: text.startswith("🪜 Шаги"))
    await press("➕ Обычный текст")
    await send("+1\nShould not save")
    async with followup_db() as session:
        assert (await session.scalars(select(FollowupStep))).all() == []
    await send("1\nSaved")
    async with followup_db() as session:
        saved = (await session.scalars(select(FollowupStep))).all()
    assert len(saved) == 1
    assert saved[0].delay_minutes == 1
    assert saved[0].message_text == "Saved"


@pytest.mark.asyncio
async def test_max_long_static_step_detail_matches_telegram_presentation(followup_db, monkeypatch):
    long_text = "x" * 3200
    campaign = FollowupCampaign(name="Long parity", include_main_dialogue=True)
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=1, message_type="static", message_text=long_text))
    async with followup_db() as session:
        session.add(campaign)
        await session.commit()
        campaign_id = campaign.id
        step_id = campaign.steps[0].id

    monkeypatch.setattr(automation_admin, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    telegram_target = SimpleNamespace(edit_text=AsyncMock(), message=SimpleNamespace(edit_text=AsyncMock()))
    telegram_target.message.edit_text = telegram_target.edit_text
    await automation_admin._show_followup_step_detail(telegram_target, campaign_id, step_id)
    telegram_text = telegram_target.edit_text.await_args.args[0]

    client = FakeMaxClient()
    await max_admin_followups.show_step(client, 1, campaign_id, step_id)
    assert client.sent[-1]["text"] == telegram_text


@pytest.mark.asyncio
async def test_max_admin_followup_campaign_journey_uses_real_application_routes(followup_db, monkeypatch):
    raw_admin_id = 505
    admin_id = MAX_ID_OFFSET + raw_admin_id
    async with followup_db() as session:
        session.add(User(id=admin_id, first_name="Admin", is_admin=True, current_dialogue_id=1))
        await session.commit()
    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    client = FakeMaxClient()
    app = MaxBotApplication(client)
    update_id = 0

    async def message(text: str):
        nonlocal update_id
        update_id += 1
        update = {
            "update_type": "message_created",
            "update_id": update_id,
            "message": {
                "body": {"text": text},
                "recipient": {"chat_id": raw_admin_id},
                "sender": {"user_id": raw_admin_id, "name": "Admin"},
            },
        }
        await app.handle_update(update)

    async def callback(payload: str):
        nonlocal update_id
        update_id += 1
        await app.handle_update({
            "update_type": "message_callback",
            "update_id": update_id,
            "callback": {
                "callback_id": f"cb-{update_id}",
                "payload": payload,
                "sender": {"user_id": raw_admin_id, "name": "Admin"},
            },
            "message": {"recipient": {"chat_id": raw_admin_id}, "body": {"attachments": []}},
        })

    await message("/admin")
    assert "admin_followups" in _callback_payloads(client.sent[-1]["attachments"])
    await callback("admin_followups")
    await callback("admin_fu_add")
    await message("MAX campaign")
    async with followup_db() as session:
        campaign = await session.scalar(select(FollowupCampaign).where(FollowupCampaign.name == "MAX campaign"))
    assert campaign is not None
    await callback(f"admin_fu_steps_{campaign.id}")
    await callback(f"admin_fu_step_add_{campaign.id}_static")
    await message("5\nНапомним о диалоге")
    async with followup_db() as session:
        step = await session.scalar(select(FollowupStep).where(FollowupStep.campaign_id == campaign.id))
    assert step is not None
    await callback(f"admin_fu_campaign_{campaign.id}")
    await callback(f"admin_fu_steps_{campaign.id}")
    await callback(f"admin_fu_step_{campaign.id}_{step.id}")
    assert "Шаг 1" in client.sent[-1]["text"]
    await callback(f"admin_fu_step_text_{step.id}")
    assert "Сообщения follow-up" in client.sent[-1]["text"]
    await callback(f"admin_fu_step_text_edit_{step.id}_ru")
    await message("Обновлённое напоминание")
    async with followup_db() as session:
        saved_step = await session.get(FollowupStep, step.id)
    assert saved_step.message_text == "Обновлённое напоминание"
    await callback(f"admin_fu_steps_{campaign.id}")
    assert any(payload.startswith("admin_fu_campaign_") for payload in _callback_payloads(client.sent[-1]["attachments"]))

    await callback(f"admin_fu_toggle_{campaign.id}")
    await callback(f"admin_fu_topics_{campaign.id}")
    await callback(f"admin_fu_scope_all_{campaign.id}")
    await callback(f"admin_fu_campaign_{campaign.id}")
    await callback(f"admin_fu_conditions_{campaign.id}")
    await callback(f"admin_fu_stage_{campaign.id}")
    await callback(f"admin_fu_stage_mode_{campaign.id}_selected")
    await message("completed")
    await callback(f"admin_fu_metadata_{campaign.id}")
    await message("profile.outcome")
    await callback(f"admin_fu_metadata_op_{campaign.id}_equals")
    await callback(f"admin_fu_metadata_operator_edit_{campaign.id}")
    await callback(f"admin_fu_metadata_op_{campaign.id}_equals")
    await message("done")
    await callback(f"admin_fu_stops_{campaign.id}")
    await message("PAYMENT_SUCCESS")
    await callback(f"admin_fu_quiet_{campaign.id}")
    await message("22:00-09:00 Europe/Moscow")
    await callback(f"admin_fu_jitter_{campaign.id}")
    await message("30-180")
    await callback(f"admin_fu_rename_{campaign.id}")
    await message("Renamed MAX campaign")
    async with followup_db() as session:
        campaign = await session.get(FollowupCampaign, campaign.id)
    assert campaign.is_active is True
    assert campaign.all_topics is True
    assert campaign.stage_mode == "selected"
    assert campaign.stage_values == "completed"
    assert campaign.metadata_field_path == "profile.outcome"
    assert campaign.metadata_expected_value == "done"
    assert campaign.stop_events == "PAYMENT_SUCCESS"
    assert campaign.quiet_start_minute == 22 * 60
    assert campaign.jitter_min_seconds == 30
    assert campaign.name == "Renamed MAX campaign"
    await callback(f"admin_fu_delete_ask_{campaign.id}")
    await callback(f"admin_fu_delete_yes_{campaign.id}")
    async with followup_db() as session:
        assert await session.get(FollowupCampaign, campaign.id) is None


@pytest.mark.asyncio
async def test_max_followup_admin_uses_visible_buttons_for_complete_real_journey(followup_db, monkeypatch):
    raw_admin_id = 515
    admin_id = MAX_ID_OFFSET + raw_admin_id
    campaign = FollowupCampaign(
        name="Visible journey",
        is_active=True,
        include_main_dialogue=True,
        quiet_start_minute=0,
        quiet_end_minute=0,
        jitter_min_seconds=0,
        jitter_max_seconds=0,
    )
    topic = Topic(name="Visible topic", is_active=True, show_in_list=True)
    campaign.steps.extend(
        [
            FollowupStep(sort_order=0, delay_minutes=5, message_type="static", message_text="Static"),
            FollowupStep(sort_order=1, delay_minutes=10, message_type="ai", ai_instruction="AI instruction"),
        ]
    )
    async with followup_db() as session:
        session.add_all([User(id=admin_id, first_name="Admin", is_admin=True, current_dialogue_id=1), topic, campaign])
        await session.commit()

    monkeypatch.setattr(max_app_module.common, "is_admin", lambda _user_id: _true_async())
    monkeypatch.setattr(max_app_module.common, "ensure_user", lambda *args, **kwargs: _true_async())
    monkeypatch.setattr(max_app_module, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    monkeypatch.setattr(automation_admin, "async_session_maker", followup_db)
    client = ValidatingMaxClient()
    app = MaxBotApplication(client)
    update_number = 0

    async def send(text):
        nonlocal update_number
        update_number += 1
        await _send_admin_text(app, raw_admin_id, update_number, text)
        return _latest_screen(client)

    async def press(label_or_match):
        nonlocal update_number
        update_number += 1
        return await _press_visible_button(app, client, raw_admin_id, update_number, label_or_match)

    await send("/admin")
    await press("💬 Догоняющие сообщения")
    await press("➕ Новая цепочка")
    await press("⬅️ Назад")
    await press("➕ Новая цепочка")
    await send("Temporary visible campaign")
    await press("⬅️ Назад")
    await press(lambda text: text.endswith("Visible journey"))

    await press("💬 Темы")
    await press(lambda text: text.endswith("Основной диалог"))
    await press(lambda text: text.startswith("❌ Основной диалог"))
    await press(lambda text: text.startswith("❌ Visible topic"))
    await press("⬅️ Назад")

    await press("⚙️ Условия")
    await press("✏️ Изменить этапы")
    await press(lambda text: text.lstrip("✅ ").startswith("На всех этапах"))
    await press("⬅️ Назад")

    await press("⚙️ Условия")
    await press("✏️ Изменить метаданные")
    await press("⬅️ Назад")
    await press("✏️ Изменить метаданные")
    await send("profile.outcome")
    await press("=")
    await send("done")
    await press("⬅️ Назад")

    await press("⚙️ Условия")
    await press("✏️ Изменить события остановки")
    await press("⬅️ Назад")
    await press("✏️ Изменить события остановки")
    await send("PAYMENT_SUCCESS")
    await press("⬅️ Назад")

    await press(lambda text: text.startswith("🪜 Шаги"))
    await press(lambda text: text.startswith("2. через 10 мин."))
    await press("✏️ Редактировать")
    await send("11\nUpdated AI instruction")
    await press("⬅️ Назад")

    await press(lambda text: text.startswith("1. через 5 мин."))
    await press("Текст сообщения")
    await press("Изменить: Сообщение")
    await send("Updated static text")
    await press("⬅️ Назад")
    await press("⬅️ Назад")

    await press("➕ Обычный текст")
    await send("3\nAdded static")
    await press("➕ Сгенерировать через AI")
    await send("4\nAdded AI")
    await press("⬅️ Назад")

    await press(lambda text: text.startswith("🌙 Тихие часы"))
    await press("⬅️ Назад")
    await press(lambda text: text.startswith("🌙 Тихие часы"))
    await send("22:00-09:00 Europe/Moscow")

    await press(lambda text: text.startswith("🎲 Случайная задержка"))
    await press("⬅️ Назад")
    await press(lambda text: text.startswith("🎲 Случайная задержка"))
    await send("30-180")

    await press("🧪 Проверить на себе")
    _, self_test_buttons = _latest_screen(client)
    if any(button.get("text", "").startswith("▶️") for button in self_test_buttons):
        await press(lambda text: text.startswith("▶️"))
    await press("⬅️ Назад")

    await press(lambda text: text.startswith("Статус:"))
    await press(lambda text: text.startswith("Статус:"))
    await press("✏️ Переименовать")
    await send("Visible journey renamed")
    await press("🗑 Удалить")
    _, delete_buttons = _latest_screen(client)
    telegram_delete_target = SimpleNamespace(edit_text=AsyncMock(), message=SimpleNamespace(edit_text=AsyncMock()))
    telegram_delete_target.message.edit_text = telegram_delete_target.edit_text
    await automation_admin.followup_delete_ask(
        SimpleNamespace(data=f"followup_delete_ask_{campaign.id}", message=telegram_delete_target)
    )
    telegram_delete_call = telegram_delete_target.edit_text.await_args
    telegram_delete_labels = [
        button.text
        for row in telegram_delete_call.kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    assert client.messages[-1]["body"]["text"] == telegram_delete_call.args[0]
    assert [button["text"] for button in delete_buttons] == telegram_delete_labels
    await press("⬅️ Назад")
    assert client.messages[-1]["body"]["text"].startswith("💬 <b>Visible journey renamed")
    async with followup_db() as session:
        saved_campaign = await session.scalar(select(FollowupCampaign).where(FollowupCampaign.name == "Visible journey renamed"))
        assert saved_campaign is not None
        assert saved_campaign.metadata_field_path == "profile.outcome"
        assert saved_campaign.metadata_expected_value == "done"
        assert saved_campaign.stop_events == "PAYMENT_SUCCESS"
        assert saved_campaign.jitter_min_seconds == 30
        assert saved_campaign.jitter_max_seconds == 180
        assert saved_campaign.quiet_start_minute == 22 * 60

    await press("🗑 Удалить")
    await press("Да, удалить")
    async with followup_db() as session:
        assert await session.get(FollowupCampaign, campaign.id) is None
    assert client.messages[-1]["body"]["text"].startswith("💬 <b>Догоняющие сообщения")
    await press("⬅️ Назад")
    assert all(
        classify_followup_callback(button["payload"]) is not None
        for button in _keyboard_buttons(client.messages[-1])
        if button.get("type") == "callback" and button.get("payload", "").startswith("admin_fu")
    )

    assert client.messages[-1]["body"]["text"] == "Добро пожаловать в админ-панель MAX."
    assert client.visible_followup_callbacks
    assert all(value in {"navigation", "mutation", "destructive", "external"} for value in client.visible_followup_callbacks.values())


@pytest.mark.asyncio
async def test_max_followup_campaign_and_step_buttons_match_telegram_contract(followup_db, monkeypatch):
    campaign = FollowupCampaign(
        name="Parity",
        is_active=True,
        include_main_dialogue=True,
        quiet_start_minute=0,
        quiet_end_minute=0,
        jitter_min_seconds=0,
        jitter_max_seconds=0,
    )
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=5, message_type="static", message_text="Text"))
    async with followup_db() as session:
        session.add(campaign)
        await session.commit()
        await session.refresh(campaign)
        campaign_id = campaign.id

    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    monkeypatch.setattr(automation_admin, "async_session_maker", followup_db)
    target = SimpleNamespace(edit_text=AsyncMock())
    await automation_admin._show_campaign(target, campaign_id, state=None)
    telegram_markup = target.edit_text.await_args.kwargs["reply_markup"]
    telegram_labels = [button.text for row in telegram_markup.inline_keyboard for button in row]

    async with followup_db() as session:
        item = await max_admin_followups._campaign(session, campaign_id)
    max_labels = [
        button["text"]
        for row in max_admin_followups._campaign_keyboard(item, "admin_fu_list")[0]["payload"]["buttons"]
        for button in row
    ]
    assert max_labels == telegram_labels

    telegram_steps_target = SimpleNamespace(
        data=f"followup_steps_{campaign_id}",
        message=SimpleNamespace(edit_text=AsyncMock()),
    )
    await automation_admin.followup_steps(telegram_steps_target, state=None)
    telegram_steps_markup = telegram_steps_target.message.edit_text.await_args.kwargs["reply_markup"]
    client = FakeMaxClient()
    await max_admin_followups.show_steps(client, 1, campaign_id)
    max_steps_labels = [
        button["text"]
        for row in client.sent[-1]["attachments"][0]["payload"]["buttons"]
        for button in row
    ]
    telegram_steps_labels = [button.text for row in telegram_steps_markup.inline_keyboard for button in row]
    assert max_steps_labels == telegram_steps_labels
    assert client.sent[-1]["text"] == telegram_steps_target.message.edit_text.await_args.args[0]


@pytest.mark.asyncio
async def test_max_self_test_screen_matches_telegram_contract(followup_db, monkeypatch):
    user_id = MAX_ID_OFFSET + 818
    campaign = FollowupCampaign(
        name="Self-test parity",
        is_active=True,
        include_main_dialogue=True,
        quiet_start_minute=0,
        quiet_end_minute=0,
        jitter_min_seconds=0,
        jitter_max_seconds=0,
    )
    campaign.steps.append(
        FollowupStep(
            sort_order=0,
            delay_minutes=5,
            message_type="static",
            message_text="Напоминание",
        )
    )
    async with followup_db() as session:
        user = User(id=user_id, first_name="MAX", current_dialogue_id=1, current_topic_id=None)
        session.add_all([user, campaign])
        await session.commit()
        await session.refresh(campaign)
        campaign_id = campaign.id

    monkeypatch.setattr(automation_admin, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    telegram_target = SimpleNamespace(edit_text=AsyncMock())
    await automation_admin._show_followup_self_test(telegram_target, campaign_id, user_id)
    telegram_text = telegram_target.edit_text.await_args.args[0]

    client = FakeMaxClient()
    await max_admin_followups.show_self_test(client, 818, user_id, campaign_id)
    max_text = client.sent[-1]["text"]
    assert max_text.replace(f"ID {818}", "ID <user>") == telegram_text.replace(f"ID {user_id}", "ID <user>")
    telegram_buttons = [
        button.text
        for row in telegram_target.edit_text.await_args.kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    max_buttons = [
        button["text"]
        for row in client.sent[-1]["attachments"][0]["payload"]["buttons"]
        for button in row
    ]
    assert max_buttons == telegram_buttons


@pytest.mark.asyncio
async def test_max_static_step_text_card_matches_telegram_contract(followup_db, monkeypatch):
    campaign = FollowupCampaign(name="Text card", include_main_dialogue=True)
    campaign.steps.append(FollowupStep(sort_order=0, delay_minutes=5, message_type="static", message_text="x" * 3200))
    async with followup_db() as session:
        session.add(campaign)
        await session.commit()
        step_id = await session.scalar(select(FollowupStep.id).where(FollowupStep.campaign_id == campaign.id))
        campaign_id = campaign.id

    monkeypatch.setattr(automation_admin, "async_session_maker", followup_db)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    telegram_target = SimpleNamespace(message=SimpleNamespace(edit_text=AsyncMock()), bot=None)
    await admin_content_authoring.resource_card(telegram_target, "followup_step", str(step_id))
    telegram_call = telegram_target.message.edit_text.await_args
    telegram_text = telegram_call.args[0]
    telegram_buttons = [
        button.text
        for row in telegram_call.kwargs["reply_markup"].inline_keyboard
        for button in row
    ]

    client = FakeMaxClient()
    await max_admin_followups.show_step_text(client, 818, step_id)
    max_payload = client.sent[-1]
    max_text = max_payload["text"]
    max_buttons = [
        button["text"]
        for row in max_payload["attachments"][0]["payload"]["buttons"]
        for button in row
    ]
    assert max_text == telegram_text
    assert max_buttons == telegram_buttons
    assert max_buttons[-1] == "⬅️ Назад"
    assert max_payload["attachments"][0]["payload"]["buttons"][-1][0]["payload"] == f"admin_fu_step_{campaign_id}_{step_id}"


@pytest.mark.asyncio
async def test_all_followup_screens_match_telegram_text_and_button_contract(followup_db, monkeypatch):
    campaign = FollowupCampaign(
        name="All screens",
        is_active=True,
        include_main_dialogue=True,
        metadata_field_path="profile.outcome",
        metadata_operator="equals",
        metadata_expected_value="done",
        stop_events="PAYMENT_SUCCESS",
        quiet_start_minute=22 * 60,
        quiet_end_minute=9 * 60,
        jitter_min_seconds=30,
        jitter_max_seconds=180,
    )
    campaign.steps.extend(
        [
            FollowupStep(sort_order=0, delay_minutes=5, message_type="static", message_text="Static"),
            FollowupStep(sort_order=1, delay_minutes=10, message_type="ai", ai_instruction="AI"),
        ]
    )
    async with followup_db() as session:
        session.add(campaign)
        await session.commit()
        await session.refresh(campaign)
        campaign_id = campaign.id
        step_id = await session.scalar(select(FollowupStep.id).where(FollowupStep.campaign_id == campaign_id).order_by(FollowupStep.sort_order).limit(1))

    monkeypatch.setattr(automation_admin, "async_session_maker", followup_db)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", followup_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", followup_db)
    monkeypatch.setattr(max_storage, "async_session_maker", followup_db)
    max_states = max_storage.StateStore()

    def telegram_target(data=None):
        edit_text = AsyncMock()
        return SimpleNamespace(
            data=data,
            edit_text=edit_text,
            message=SimpleNamespace(edit_text=edit_text),
            answer=AsyncMock(),
        )

    async def render_telegram(renderer):
        target = telegram_target()
        await renderer(target)
        call = target.edit_text.await_args
        labels = [
            button.text
            for row in call.kwargs["reply_markup"].inline_keyboard
            for button in row
        ]
        return call.args[0], labels

    async def render_telegram_callback(renderer, data, state=None):
        target = telegram_target(data)
        await renderer(target, state=state)
        call = target.edit_text.await_args
        labels = [
            button.text
            for row in call.kwargs["reply_markup"].inline_keyboard
            for button in row
        ]
        return call.args[0], labels

    client = ValidatingMaxClient()

    async def render_max(renderer, *args):
        await renderer(client, *args)
        request, buttons = _latest_screen(client)
        return request["body"]["text"], [button["text"] for button in buttons]

    pairs = []
    pairs.append(
        (
            await render_telegram(lambda target: automation_admin._show_followup_campaigns(target)),
            await render_max(max_admin_followups.show_campaigns, 1),
        )
    )
    pairs.append(
        (
            await render_telegram(lambda target: automation_admin._show_campaign(target, campaign_id)),
            await render_max(max_admin_followups.show_campaign, 1, campaign_id),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_topics, f"followup_topics_{campaign_id}"),
            await render_max(max_admin_followups.show_topics, 1, campaign_id),
        )
    )
    pairs.append(
        (
            await render_telegram(lambda target: automation_admin._show_followup_conditions(target, campaign_id)),
            await render_max(max_admin_followups.show_conditions, 1, campaign_id),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_stage_edit, f"followup_stage_edit_{campaign_id}"),
            await render_max(max_admin_followups.show_stage_picker, 1, campaign_id),
        )
    )

    metadata_state = SimpleNamespace(
        get_data=AsyncMock(return_value={"campaign_id": campaign_id, "metadata_field_path": "profile.outcome"}),
        set_state=AsyncMock(),
        update_data=AsyncMock(),
        clear=AsyncMock(),
        set_data=AsyncMock(),
    )
    await max_states.set(1, 1, "max_followup_metadata_operator", {"campaign_id": campaign_id, "field": "profile.outcome"})
    pairs.append(
        (
            await render_telegram(
                lambda target: automation_admin._show_followup_metadata_operator(target, campaign_id, metadata_state)
            ),
            await render_max(max_admin_followups.show_metadata_operator, max_states, 1, 1, campaign_id),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_stop_events_edit, f"followup_stop_events_edit_{campaign_id}", metadata_state),
            await render_max(max_admin_followups.start_stops, max_states, 1, 1, campaign_id),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_steps, f"followup_steps_{campaign_id}"),
            await render_max(max_admin_followups.show_steps, 1, campaign_id),
        )
    )
    pairs.append(
        (
            await render_telegram(lambda target: automation_admin._show_followup_step_detail(target, campaign_id, step_id)),
            await render_max(max_admin_followups.show_step, 1, campaign_id, step_id),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_quiet, f"followup_quiet_{campaign_id}", metadata_state),
            await render_max(max_admin_followups.start_quiet, max_states, 1, 1, campaign_id),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_jitter, f"followup_jitter_{campaign_id}", metadata_state),
            await render_max(max_admin_followups.start_jitter, max_states, 1, 1, campaign_id),
        )
    )

    input_state = SimpleNamespace(
        get_data=AsyncMock(return_value={}),
        set_state=AsyncMock(),
        update_data=AsyncMock(),
        clear=AsyncMock(),
        set_data=AsyncMock(),
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_stage_mode, f"followup_stage_mode_{campaign_id}_selected", input_state),
            await render_max(max_admin_followups.select_stage_mode, max_states, 1, 1, campaign_id, "selected"),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_metadata_edit, f"followup_metadata_edit_{campaign_id}", input_state),
            await render_max(max_admin_followups.start_metadata, max_states, 1, 1, campaign_id),
        )
    )
    await max_states.set(1, 1, "max_followup_metadata_operator", {"campaign_id": campaign_id, "field": "profile.outcome"})
    input_state.get_data = AsyncMock(return_value={"campaign_id": campaign_id, "metadata_field_path": "profile.outcome"})
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_metadata_operator, f"followup_metadata_operator_{campaign_id}_equals", input_state),
            await render_max(max_admin_followups.select_metadata_operator, max_states, 1, 1, campaign_id, "equals"),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_step_add, f"followup_step_add_{campaign_id}_static", input_state),
            await render_max(max_admin_followups.start_step_add, max_states, 1, 1, campaign_id, "static"),
        )
    )
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_step_add, f"followup_step_add_{campaign_id}_ai", input_state),
            await render_max(max_admin_followups.start_step_add, max_states, 1, 1, campaign_id, "ai"),
        )
    )
    input_state.get_data = AsyncMock(return_value={})
    pairs.append(
        (
            await render_telegram_callback(automation_admin.followup_step_edit, f"followup_step_edit_{campaign_id}_{step_id}", input_state),
            await render_max(max_admin_followups.start_step_edit, max_states, 1, 1, campaign_id, step_id),
        )
    )

    for index, (telegram_result, max_result) in enumerate(pairs):
        assert max_result == telegram_result, index


def _true_async():
    async def value():
        return True

    return value()


def _false_async():
    async def value():
        return False

    return value()


def _callback_payloads(attachments):
    return {
        button["payload"]
        for attachment in attachments or []
        if attachment.get("type") == "inline_keyboard"
        for row in attachment.get("payload", {}).get("buttons", [])
        for button in row
        if button.get("type") == "callback"
    }


def test_every_visible_followup_callback_has_a_contract_classification():
    visible_payloads = {
        "admin_followups",
        "admin_fu_list",
        "admin_fu_add",
        "admin_fu_campaign_1",
        "admin_fu_toggle_1",
        "admin_fu_rename_1",
        "admin_fu_topics_1",
        "admin_fu_steps_1",
        "admin_fu_conditions_1",
        "admin_fu_self_test_1",
        "admin_fu_quiet_1",
        "admin_fu_jitter_1",
        "admin_fu_delete_ask_1",
        "admin_fu_scope_all_1",
        "admin_fu_scope_main_1",
        "admin_fu_scope_topic_1_2",
        "admin_fu_stage_1",
        "admin_fu_stage_mode_1_selected",
        "admin_fu_metadata_1",
        "admin_fu_metadata_op_1_equals",
        "admin_fu_metadata_operator_edit_1",
        "admin_fu_metadata_clear_1",
        "admin_fu_stops_1",
        "admin_fu_stops_clear_1",
        "admin_fu_step_add_1_static",
        "admin_fu_step_add_1_ai",
        "admin_fu_step_1_2",
        "admin_fu_step_edit_1_2",
        "admin_fu_step_text_2",
        "admin_fu_step_text_edit_2_ru",
        "admin_fu_step_delete_1_2",
        "admin_fu_step_delete_yes_1_2",
        "admin_fu_self_test_send_1",
        "admin_fu_delete_yes_1",
    }

    assert all(classify_followup_callback(payload) in {"navigation", "mutation", "destructive", "external"} for payload in visible_payloads)
    assert classify_followup_callback("admin_fu_unknown_1") is None


@pytest.mark.asyncio
async def test_max_followup_transport_uses_real_api_boundary_validation():
    client = ValidatingMaxClient()
    await client.send_message(
        chat_id=1,
        text="Проверка",
        attachments=[
            {
                "type": "inline_keyboard",
                "payload": {"buttons": [[{"type": "callback", "text": "Открыть", "payload": "admin_fu_list"}]]},
            }
        ],
    )
    assert client.validate_latest_message()["body"]["text"] == "Проверка"
    with pytest.raises(AssertionError):
        await client.send_message(
            chat_id=1,
            text="Проверка",
            attachments=[
                {
                    "type": "inline_keyboard",
                    "payload": {"buttons": [[{"type": "callback", "text": "Плохо", "payload": None}]]},
                }
            ],
        )
    with pytest.raises(AssertionError):
        await client.send_message(
            chat_id=1,
            text="Проверка",
            attachments=[{"type": "unknown_attachment", "payload": {}}],
        )
