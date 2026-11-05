import os
import io
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram.methods import SendMediaGroup, SendMessage
from aiogram.types import BufferedInputFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from database import Base, EventDelivery, EventDeliveryPart, EventRegistration, EventScenario, EventScenarioMedia, User
from event_admin_service import add_step, resolve_uncertain_delivery, update_scenario_fields
from event_delivery_worker import (
    _build_delivery_plan,
    claim_pending_deliveries,
    process_single_delivery,
    PreTransmissionError,
    set_runtime_max_client,
)
from event_service import (
    DEFAULT_ADAPTIVE_CONFIRMATION_TEMPLATE,
    DEFAULT_LIVE_TEMPLATE,
    handle_event_callback_max,
    handle_event_callback_telegram,
    handle_event_start_max,
    handle_event_start_telegram,
    cancel_event_registration,
    register_user_for_event,
    render_event_message,
    render_event_template,
)


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _setup_media_event(session_factory, *, description: str = "Описание"):
    now = datetime.utcnow()
    async with session_factory() as session:
        session.add(User(id=9101, first_name="Пользователь"))
        scenario = EventScenario(
            code="media_delivery_event",
            event_type="webinar",
            title="Мероприятие",
            starts_at=now + timedelta(days=1),
            duration_minutes=120,
            link="https://example.test/live",
            description=description,
            media_order="media_top",
            is_active=True,
            is_archived=False,
        )
        session.add(scenario)
        await session.flush()
        session.add_all([
            EventScenarioMedia(event_id=scenario.id, file_type="photo", file_id="tg-photo", max_token="max-photo", sort_order=0),
            EventScenarioMedia(event_id=scenario.id, file_type="video", file_id="tg-video", max_token="max-video", sort_order=1),
        ])
        await session.commit()
    result = await register_user_for_event(session_factory, 9101, "media_delivery_event", now=now)
    return now, result.immediate_delivery.id


async def _create_unregistered_media_event(session_factory, *, user_id: int, code: str, max_tokens: bool = True):
    now = datetime.utcnow()
    async with session_factory() as session:
        session.add(User(id=user_id, first_name="Пользователь"))
        scenario = EventScenario(
            code=code,
            event_type="webinar",
            title="Маршрут мероприятия",
            starts_at=now + timedelta(days=1),
            duration_minutes=120,
            link="https://example.test/route",
            description="<b>HTML описание</b>",
            media_order="media_top",
            is_active=True,
            is_archived=False,
        )
        session.add(scenario)
        await session.flush()
        session.add_all([
            EventScenarioMedia(event_id=scenario.id, file_type="photo", file_id="tg-photo-route", max_token="max-photo-route" if max_tokens else None, sort_order=0),
            EventScenarioMedia(event_id=scenario.id, file_type="video", file_id="tg-video-route", max_token="max-video-route" if max_tokens else None, sort_order=1),
        ])
        await session.commit()
    return now, scenario.id


async def _create_meeting_event(session_factory, *, user_id: int, code: str, starts_in_minutes: int = 10):
    now = datetime.utcnow()
    async with session_factory() as session:
        session.add(User(id=user_id, first_name="Участник встречи"))
        scenario = EventScenario(
            code=code,
            event_type="meeting",
            title="Очная встреча",
            starts_at=now + timedelta(minutes=starts_in_minutes),
            duration_minutes=120,
            place="Москва, ул. Тверская, 7, зал 3",
            link=None,
            is_active=True,
            is_archived=False,
        )
        session.add(scenario)
        await session.commit()
    return now, scenario.id


@pytest.mark.asyncio
async def test_meeting_default_templates_render_place_instead_of_empty_link(session_factory):
    now, event_id = await _create_meeting_event(session_factory, user_id=9210, code="meeting_template_place")
    async with session_factory() as session:
        scenario = await session.get(EventScenario, event_id)

    for template in (DEFAULT_ADAPTIVE_CONFIRMATION_TEMPLATE, DEFAULT_LIVE_TEMPLATE):
        rendered = render_event_template(template, scenario, now=now)
        assert "Москва, ул. Тверская, 7, зал 3" in rendered
        assert "Ссылка для входа:" not in rendered
        assert "Ссылка для подключения:" not in rendered


@pytest.mark.asyncio
async def test_telegram_late_meeting_registration_sends_place(session_factory):
    now, event_id = await _create_meeting_event(session_factory, user_id=9211, code="meeting_late_telegram")
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=81))

    assert await handle_event_start_telegram(
        bot,
        chat_id=9211,
        user_id=9211,
        start_arg="meeting_late_telegram",
        session_maker=session_factory,
        now=now,
    ) is True

    texts = [call.kwargs["text"] for call in bot.send_message.call_args_list]
    assert any("Москва, ул. Тверская, 7, зал 3" in text for text in texts)
    assert all("Ссылка для входа:" not in text for text in texts)


@pytest.mark.asyncio
async def test_max_late_meeting_registration_sends_place(session_factory):
    user_id = 1000000009212
    now, event_id = await _create_meeting_event(session_factory, user_id=user_id, code="meeting_late_max")
    client = MagicMock()
    client.send_message = AsyncMock(return_value={"message_id": "max-meeting-place"})

    assert await handle_event_start_max(
        client,
        chat_id=user_id,
        user_id=user_id,
        start_payload="meeting_late_max",
        session_maker=session_factory,
        now=now,
    ) is True

    text = client.send_message.call_args.kwargs["text"]
    assert "Москва, ул. Тверская, 7, зал 3" in text
    assert "Ссылка для входа:" not in text


@pytest.mark.asyncio
async def test_telegram_adaptive_meeting_confirmation_sends_place(session_factory):
    now, event_id = await _create_meeting_event(
        session_factory,
        user_id=9213,
        code="meeting_adaptive_telegram",
        starts_in_minutes=20,
    )
    result = await register_user_for_event(
        session_factory,
        9213,
        "meeting_adaptive_telegram",
        platform="telegram",
        now=now,
    )
    adaptive_now = now + timedelta(minutes=6)
    claimed = await claim_pending_deliveries(session_factory, now=adaptive_now)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=82))

    assert await process_single_delivery(
        session_factory,
        result.immediate_delivery.id,
        claimed[0][1],
        bot=bot,
        now=adaptive_now,
    ) == "sent"
    assert "Москва, ул. Тверская, 7, зал 3" in bot.send_message.call_args.kwargs["text"]
    assert "Ссылка для входа:" not in bot.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_max_adaptive_meeting_confirmation_sends_place(session_factory):
    user_id = 1000000009214
    now, event_id = await _create_meeting_event(
        session_factory,
        user_id=user_id,
        code="meeting_adaptive_max",
        starts_in_minutes=20,
    )
    result = await register_user_for_event(
        session_factory,
        user_id,
        "meeting_adaptive_max",
        platform="max",
        now=now,
    )
    adaptive_now = now + timedelta(minutes=6)
    claimed = await claim_pending_deliveries(session_factory, now=adaptive_now)
    client = MagicMock()
    client.send_message = AsyncMock(return_value={"message_id": "max-adaptive-place"})

    assert await process_single_delivery(
        session_factory,
        result.immediate_delivery.id,
        claimed[0][1],
        max_client=client,
        now=adaptive_now,
    ) == "sent"
    assert "Москва, ул. Тверская, 7, зал 3" in client.send_message.call_args.kwargs["text"]
    assert "Ссылка для входа:" not in client.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_retry_after_text_failure_does_not_repeat_sent_gallery(session_factory):
    now, delivery_id = await _setup_media_event(session_factory, description="Короткое описание")
    claimed = await claim_pending_deliveries(session_factory, batch_size=10, now=now)
    assert claimed == [(delivery_id, claimed[0][1])]

    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=1), MagicMock(message_id=2)])
    bot.send_message = AsyncMock(side_effect=PreTransmissionError("temporary transport failure"))
    first = await process_single_delivery(session_factory, delivery_id, claimed[0][1], bot=bot, now=now)
    assert first == "pre_transmission_retry"
    assert bot.send_media_group.await_count == 1
    assert bot.send_message.await_count == 1

    retry_now = now + timedelta(seconds=61)
    retry_claim = await claim_pending_deliveries(session_factory, batch_size=10, now=retry_now)
    assert retry_claim
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=3))
    second = await process_single_delivery(session_factory, delivery_id, retry_claim[0][1], bot=bot, now=retry_now)
    assert second == "sent"
    assert bot.send_media_group.await_count == 1
    assert bot.send_message.await_count == 1

    async with session_factory() as session:
        delivery = await session.get(EventDelivery, delivery_id)
        parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == delivery_id).order_by(EventDeliveryPart.part_index))).scalars().all()
        assert delivery.status == "sent"
        assert [part.status for part in parts] == ["sent", "sent"]


@pytest.mark.asyncio
async def test_content_change_stops_unfinished_parts_and_preserves_sent_parts(session_factory):
    now, delivery_id = await _setup_media_event(session_factory, description="Короткое описание")
    claimed = await claim_pending_deliveries(session_factory, batch_size=10, now=now)
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=1), MagicMock(message_id=2)])
    bot.send_message = AsyncMock(side_effect=PreTransmissionError("temporary transport failure"))
    assert await process_single_delivery(session_factory, delivery_id, claimed[0][1], bot=bot, now=now) == "pre_transmission_retry"

    await update_scenario_fields(session_factory, 1, description="Новая версия")
    async with session_factory() as session:
        delivery = await session.get(EventDelivery, delivery_id)
        parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == delivery_id).order_by(EventDeliveryPart.part_index))).scalars().all()
        assert delivery.status == "skipped"
        assert delivery.skip_reason == "content_changed"
        assert parts[0].status == "sent"
        assert parts[1].status == "skipped"


@pytest.mark.asyncio
async def test_admin_force_retry_resumes_only_uncertain_part(session_factory):
    now, delivery_id = await _setup_media_event(session_factory, description="Короткое описание")
    claimed = await claim_pending_deliveries(session_factory, batch_size=10, now=now)
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=1), MagicMock(message_id=2)])
    bot.send_message = AsyncMock(side_effect=TimeoutError("request result unknown"))
    assert await process_single_delivery(session_factory, delivery_id, claimed[0][1], bot=bot, now=now) == "uncertain"

    child, created = await resolve_uncertain_delivery(
        session_factory,
        delivery_id=delivery_id,
        admin_id=999,
        resolution_type="force_retry",
        resolution_reason="Проверка результата в журнале провайдера",
    )
    assert created is True
    async with session_factory() as session:
        child_parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == child.id).order_by(EventDeliveryPart.part_index))).scalars().all()
        assert [part.status for part in child_parts] == ["sent", "pending"]

    retry_claim = await claim_pending_deliveries(session_factory, batch_size=10, now=now + timedelta(seconds=1))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=3))
    assert await process_single_delivery(session_factory, child.id, retry_claim[0][1], bot=bot, now=now + timedelta(seconds=1)) == "sent"
    assert bot.send_media_group.await_count == 1


@pytest.mark.asyncio
async def test_album_success_then_control_timeout_retries_only_control(session_factory):
    now, delivery_id = await _setup_media_event(session_factory, description="Короткое описание")
    async with session_factory() as session:
        scenario = await session.get(EventScenario, 1)
        scenario.media_order = "text_top"
        await session.commit()

    claimed = await claim_pending_deliveries(session_factory, batch_size=10, now=now)
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=10), MagicMock(message_id=11)])
    bot.send_message = AsyncMock(side_effect=[MagicMock(message_id=12), PreTransmissionError("control timeout")])
    assert await process_single_delivery(session_factory, delivery_id, claimed[0][1], bot=bot, now=now) == "pre_transmission_retry"
    assert bot.send_media_group.await_count == 1
    assert bot.send_message.await_count == 2

    retry_now = now + timedelta(seconds=61)
    retry_claim = await claim_pending_deliveries(session_factory, batch_size=10, now=retry_now)
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=13))
    assert await process_single_delivery(session_factory, delivery_id, retry_claim[0][1], bot=bot, now=retry_now) == "sent"
    assert bot.send_media_group.await_count == 1
    assert bot.send_message.await_count == 1

    async with session_factory() as session:
        parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == delivery_id).order_by(EventDeliveryPart.part_index))).scalars().all()
        assert [part.status for part in parts] == ["sent", "sent", "sent"]


@pytest.mark.asyncio
async def test_registration_cancelled_between_parts_stops_remaining_delivery(session_factory):
    now, delivery_id = await _setup_media_event(session_factory, description="Короткое описание")
    claimed = await claim_pending_deliveries(session_factory, batch_size=10, now=now)
    bot = MagicMock()

    async def send_gallery(**kwargs):
        await cancel_event_registration(session_factory, 9101, 1, now=now)
        return [MagicMock(message_id=20), MagicMock(message_id=21)]

    bot.send_media_group = AsyncMock(side_effect=send_gallery)
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=22))
    outcome = await process_single_delivery(session_factory, delivery_id, claimed[0][1], bot=bot, now=now)
    assert outcome == "pre_send_rejected:cancelled"
    bot.send_message.assert_not_awaited()

    async with session_factory() as session:
        delivery = await session.get(EventDelivery, delivery_id)
        parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == delivery_id).order_by(EventDeliveryPart.part_index))).scalars().all()
        registration = await session.get(EventRegistration, delivery.registration_id)
        assert delivery.status == "skipped"
        assert delivery.skip_reason == "cancelled"
        assert [part.status for part in parts] == ["sent", "skipped"]
        assert registration.status == "cancelled"


@pytest.mark.parametrize(
    ("resolution_type", "expected_part_status"),
    [("mark_sent", "resolved_sent"), ("mark_skipped", "resolved_skipped")],
)
@pytest.mark.asyncio
async def test_manual_resolution_targets_uncertain_part_and_resumes_remainder(
    session_factory,
    resolution_type,
    expected_part_status,
):
    now, delivery_id = await _setup_media_event(session_factory, description="Короткое описание")
    async with session_factory() as session:
        scenario = await session.get(EventScenario, 1)
        scenario.media_order = "text_top"
        await session.commit()
    claimed = await claim_pending_deliveries(session_factory, batch_size=10, now=now)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=30))
    bot.send_media_group = AsyncMock(side_effect=TimeoutError("gallery result unknown"))
    assert await process_single_delivery(session_factory, delivery_id, claimed[0][1], bot=bot, now=now) == "uncertain"

    async with session_factory() as session:
        parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == delivery_id).order_by(EventDeliveryPart.part_index))).scalars().all()
        uncertain_part_id = parts[1].id

    resolved, created = await resolve_uncertain_delivery(
        session_factory,
        delivery_id=delivery_id,
        admin_id=999,
        resolution_type=resolution_type,
        resolution_reason="Проверка ручного разрешения части",
        part_id=uncertain_part_id,
    )
    assert created is True
    assert resolved.status == "pending"

    async with session_factory() as session:
        parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == delivery_id).order_by(EventDeliveryPart.part_index))).scalars().all()
        assert [part.status for part in parts] == ["sent", expected_part_status, "pending"]

    retry_claim = await claim_pending_deliveries(session_factory, batch_size=10, now=now + timedelta(seconds=1))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=31))
    assert await process_single_delivery(session_factory, delivery_id, retry_claim[0][1], bot=bot, now=now + timedelta(seconds=1)) == "sent"
    bot.send_media_group.assert_awaited_once()
    assert bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_force_retry_after_content_change_creates_fresh_controlled_sequence(session_factory):
    now, delivery_id = await _setup_media_event(session_factory, description="Старая версия")
    claimed = await claim_pending_deliveries(session_factory, batch_size=10, now=now)
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=1), MagicMock(message_id=2)])
    bot.send_message = AsyncMock(side_effect=TimeoutError("request result unknown"))
    assert await process_single_delivery(session_factory, delivery_id, claimed[0][1], bot=bot, now=now) == "uncertain"

    await update_scenario_fields(session_factory, 1, description="Новая версия")
    child, created = await resolve_uncertain_delivery(
        session_factory,
        delivery_id=delivery_id,
        admin_id=999,
        resolution_type="force_retry",
        resolution_reason="Содержимое изменено, нужна отдельная доставка",
    )
    assert created is True
    async with session_factory() as session:
        scenario = await session.get(EventScenario, 1)
        parts = (await session.execute(select(EventDeliveryPart).where(EventDeliveryPart.delivery_id == child.id))).scalars().all()
        assert child.content_version_snapshot == scenario.content_version
        assert parts == []


@pytest.mark.asyncio
async def test_telegram_user_route_delivers_confirmation_gallery_and_cancel_callback(session_factory):
    now, event_id = await _create_unregistered_media_event(session_factory, user_id=9201, code="media_route_tg")
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=11), MagicMock(message_id=12)])
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=13))

    assert await handle_event_start_telegram(
        bot,
        chat_id=9201,
        user_id=9201,
        start_arg="media_route_tg",
        session_maker=session_factory,
        now=now,
    ) is True
    assert bot.send_media_group.await_count == 1
    sent_text = bot.send_message.call_args.kwargs["text"]
    assert "HTML описание" in sent_text
    assert bot.send_message.call_args.kwargs["reply_markup"] is not None

    callback = MagicMock()
    callback.data = f"ev_cancel:{event_id}"
    callback.from_user.id = 9201
    callback.answer = AsyncMock()
    callback.message.edit_text = AsyncMock()
    assert await handle_event_callback_telegram(callback, session_maker=session_factory, now=now) is True
    callback.answer.assert_awaited_once()
    callback.data = f"ev_rejoin:{event_id}"
    assert await handle_event_callback_telegram(callback, session_maker=session_factory, now=now) is True
    async with session_factory() as session:
        registration = await session.scalar(select(EventRegistration).where(EventRegistration.event_id == event_id))
        assert registration.status == "started"


@pytest.mark.asyncio
async def test_max_user_route_delivers_confirmation_gallery_and_cancel_callback(session_factory):
    user_id = 100000000921
    now, event_id = await _create_unregistered_media_event(session_factory, user_id=user_id, code="media_route_max")
    client = MagicMock()
    client.send_message = AsyncMock(return_value={"message_id": "max-13"})

    assert await handle_event_start_max(
        client,
        chat_id=921,
        user_id=user_id,
        start_payload="media_route_max",
        session_maker=session_factory,
        now=now,
    ) is True
    calls = client.send_message.call_args_list
    assert len(calls) == 2
    media_payload = calls[0].kwargs
    text_payload = calls[1].kwargs
    assert {item.get("type") for item in media_payload["attachments"]} >= {"image", "video"}
    assert "HTML описание" in text_payload["text"]
    assert any(item.get("type") == "inline_keyboard" for item in text_payload["attachments"])

    callback = MagicMock()
    callback.payload = f"ev_cancel:{event_id}"
    callback.sender.user_id = user_id
    callback.chat_id = 921
    callback.callback_id = "max-cancel"
    client.answer_callback = AsyncMock()
    assert await handle_event_callback_max(client, callback, session_maker=session_factory, now=now) is True
    client.answer_callback.assert_awaited_once()
    callback.payload = f"ev_rejoin:{event_id}"
    assert await handle_event_callback_max(client, callback, session_maker=session_factory, now=now) is True
    async with session_factory() as session:
        registration = await session.scalar(select(EventRegistration).where(EventRegistration.event_id == event_id))
        assert registration.status == "started"


@pytest.mark.asyncio
async def test_max_origin_media_is_materialized_for_telegram_and_payloads_validate(session_factory):
    now, event_id = await _create_unregistered_media_event(
        session_factory,
        user_id=9202,
        code="max_origin_to_telegram",
    )
    async with session_factory() as session:
        media = (
            await session.execute(
                select(EventScenarioMedia)
                .where(EventScenarioMedia.event_id == event_id)
                .order_by(EventScenarioMedia.sort_order.asc())
            )
        ).scalars().all()
        for index, row in enumerate(media):
            row.source_platform = "max"
            row.file_id = f"max-origin-file-{index}"
            row.max_token = f"max-origin-token-{index}"
        await session.commit()

    registration = await register_user_for_event(
        session_factory,
        9202,
        "max_origin_to_telegram",
        platform="telegram",
        now=now,
    )
    claimed = await claim_pending_deliveries(session_factory, now=now)
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=41), MagicMock(message_id=42)])
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=43))
    max_client = MagicMock()
    max_client.download_attachment = AsyncMock(side_effect=[b"photo-bytes", b"video-bytes"])

    assert await process_single_delivery(
        session_factory,
        registration.immediate_delivery.id,
        claimed[0][1],
        bot=bot,
        max_client=max_client,
        now=now,
    ) == "sent"
    assert max_client.download_attachment.await_count == 2

    media_payload = bot.send_media_group.call_args.kwargs["media"]
    SendMediaGroup(chat_id=9202, media=media_payload)
    assert all(isinstance(item.media, BufferedInputFile) for item in media_payload)
    message_kwargs = bot.send_message.call_args.kwargs
    SendMessage(chat_id=9202, text=message_kwargs["text"], parse_mode=message_kwargs["parse_mode"], reply_markup=message_kwargs["reply_markup"])


@pytest.mark.asyncio
async def test_telegram_route_warns_when_max_origin_media_cannot_materialize(session_factory):
    now, event_id = await _create_unregistered_media_event(
        session_factory,
        user_id=9203,
        code="max_origin_without_client",
    )
    async with session_factory() as session:
        media = (
            await session.execute(select(EventScenarioMedia).where(EventScenarioMedia.event_id == event_id))
        ).scalars().all()
        for index, row in enumerate(media):
            row.source_platform = "max"
            row.file_id = f"max-origin-file-missing-{index}"
            row.max_token = f"max-origin-token-missing-{index}"
        await session.commit()

    bot = MagicMock()
    bot.send_media_group = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=44))
    assert await handle_event_start_telegram(
        bot,
        chat_id=9203,
        user_id=9203,
        start_arg="max_origin_without_client",
        session_maker=session_factory,
        now=now,
    ) is True
    assert bot.send_media_group.await_count == 0
    assert any("Вложения из MAX" in call.kwargs["text"] for call in bot.send_message.call_args_list)
    async with session_factory() as session:
        delivery = await session.scalar(
            select(EventDelivery).where(
                EventDelivery.event_id == event_id,
                EventDelivery.message_kind == "immediate_confirmation",
            )
        )
        assert delivery.status == "pending"
        assert "MAX-origin" in (delivery.last_error or "")


@pytest.mark.asyncio
async def test_telegram_user_route_uses_runtime_max_transport_for_max_origin_media(session_factory):
    now, event_id = await _create_unregistered_media_event(
        session_factory,
        user_id=9204,
        code="max_origin_runtime_client",
    )
    async with session_factory() as session:
        media = (
            await session.execute(select(EventScenarioMedia).where(EventScenarioMedia.event_id == event_id))
        ).scalars().all()
        for index, row in enumerate(media):
            row.source_platform = "max"
            row.file_id = f"runtime-max-file-{index}"
            row.max_token = f"runtime-max-token-{index}"
        await session.commit()

    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=51), MagicMock(message_id=52)])
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=53))
    max_client = MagicMock()
    max_client.download_attachment = AsyncMock(side_effect=[b"runtime-photo", b"runtime-video"])
    set_runtime_max_client(max_client)
    try:
        assert await handle_event_start_telegram(
            bot,
            chat_id=9204,
            user_id=9204,
            start_arg="max_origin_runtime_client",
            session_maker=session_factory,
            now=now,
        ) is True
    finally:
        set_runtime_max_client(None)

    assert max_client.download_attachment.await_count == 2
    assert bot.send_media_group.await_count == 1
    assert bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_delivery_part_claim_uses_fresh_clock_between_parts(session_factory):
    now, delivery_id = await _setup_media_event(session_factory, description="Короткое описание")
    expires_at = now + timedelta(seconds=1)
    async with session_factory() as session:
        delivery = await session.get(EventDelivery, delivery_id)
        delivery.expires_at = expires_at
        await session.commit()

    claimed = await claim_pending_deliveries(session_factory, now=now)
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=61), MagicMock(message_id=62)])
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=63))
    clock_calls = 0

    def advancing_clock():
        nonlocal clock_calls
        clock_calls += 1
        return now if clock_calls < 6 else now + timedelta(seconds=2)

    outcome = await process_single_delivery(
        session_factory,
        delivery_id,
        claimed[0][1],
        bot=bot,
        now=now,
        clock=advancing_clock,
    )
    assert outcome == "pre_send_rejected:expired"
    assert bot.send_media_group.await_count == 1
    assert bot.send_message.await_count == 0
    async with session_factory() as session:
        delivery = await session.get(EventDelivery, delivery_id)
        parts = (
            await session.execute(
                select(EventDeliveryPart)
                .where(EventDeliveryPart.delivery_id == delivery_id)
                .order_by(EventDeliveryPart.part_index)
            )
        ).scalars().all()
        assert delivery.status == "skipped"
        assert delivery.skip_reason == "expired"
        assert [part.status for part in parts] == ["sent", "skipped"]


@pytest.mark.asyncio
async def test_reminder_default_does_not_send_event_details_or_media(session_factory):
    now, immediate_id = await _setup_media_event(session_factory, description="Не отправлять в напоминании")
    claimed = await claim_pending_deliveries(session_factory, now=now)
    bot = MagicMock()
    bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=31), MagicMock(message_id=32)])
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=33))
    assert await process_single_delivery(session_factory, immediate_id, claimed[0][1], bot=bot, now=now) == "sent"

    step = await add_step(session_factory, 1, 0, "Короткое напоминание")
    async with session_factory() as session:
        scheduled = await session.scalar(
            select(EventDelivery).where(
                EventDelivery.step_id == step.id,
                EventDelivery.message_kind == "scheduled_step",
            )
        )
        scheduled.due_at = now
        scheduled.next_attempt_at = now
        await session.commit()

    claimed = await claim_pending_deliveries(session_factory, now=now)
    assert len(claimed) == 1
    bot.send_media_group.reset_mock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=34))
    assert await process_single_delivery(session_factory, scheduled.id, claimed[0][1], bot=bot, now=now) == "sent"
    bot.send_media_group.assert_not_awaited()
    assert "Не отправлять в напоминании" not in bot.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_max_delivery_materializes_telegram_media_before_send(session_factory):
    now, event_id = await _create_unregistered_media_event(
        session_factory,
        user_id=100000000931,
        code="media_materialize_max",
        max_tokens=False,
    )
    result = await register_user_for_event(
        session_factory,
        100000000931,
        "media_materialize_max",
        platform="max",
        now=now,
    )
    claimed = await claim_pending_deliveries(session_factory, now=now)
    telegram_bot = MagicMock()
    telegram_bot.get_file = AsyncMock(side_effect=[MagicMock(file_path="photo.jpg"), MagicMock(file_path="video.mp4")])
    telegram_bot.download_file = AsyncMock(side_effect=lambda _: io.BytesIO(b"media-bytes"))
    max_client = MagicMock()
    max_client.upload_file = AsyncMock(side_effect=[{"token": "uploaded-photo"}, {"token": "uploaded-video"}])
    max_client.send_message = AsyncMock(return_value={"message_id": "max-materialized"})

    assert await process_single_delivery(
        session_factory,
        result.immediate_delivery.id,
        claimed[0][1],
        bot=telegram_bot,
        max_client=max_client,
        now=now,
    ) == "sent"
    assert max_client.upload_file.await_count == 2
    assert max_client.send_message.await_count == 2
    async with session_factory() as session:
        media = (await session.execute(select(EventScenarioMedia).where(EventScenarioMedia.event_id == event_id).order_by(EventScenarioMedia.sort_order))).scalars().all()
        assert [item.max_token for item in media] == ["uploaded-photo", "uploaded-video"]


@pytest.mark.asyncio
async def test_new_step_delivery_snapshots_new_content_version(session_factory):
    now = datetime.utcnow()
    async with session_factory() as session:
        session.add(User(id=9401, first_name="Step user"))
        scenario = EventScenario(
            code="step_version_event",
            event_type="webinar",
            title="Step version",
            starts_at=now + timedelta(days=1),
            duration_minutes=120,
            link="https://example.test/step",
            is_active=True,
        )
        session.add(scenario)
        await session.commit()
    await register_user_for_event(session_factory, 9401, "step_version_event", now=now)
    step = await add_step(session_factory, scenario.id, 0, "Новое напоминание")
    async with session_factory() as session:
        current = await session.get(EventScenario, scenario.id)
        delivery = await session.scalar(select(EventDelivery).where(EventDelivery.step_id == step.id))
        assert delivery.content_version_snapshot == current.content_version


@pytest.mark.asyncio
async def test_long_html_description_is_split_into_safe_parts(session_factory):
    now = datetime.utcnow()
    async with session_factory() as session:
        scenario = EventScenario(
            code="long_description_event",
            event_type="webinar",
            title="Long description",
            starts_at=now + timedelta(days=1),
            duration_minutes=120,
            link="https://example.test/long",
            description="<b>" + ("текст " * 1200) + "</b>",
            is_active=True,
        )
        session.add(scenario)
        await session.flush()
        session.add(EventScenarioMedia(event_id=scenario.id, file_type="photo", file_id="long-photo", max_token="long-photo-token", sort_order=0))
        await session.commit()
        await session.refresh(scenario, ["media"])
        rendered = render_event_message("Подтверждение", scenario, include_event_details=True)
        plan = _build_delivery_plan(scenario, scenario.id, rendered)
    text_parts = [item["text"] for item in plan if item["kind"] == "text"]
    assert len(text_parts) >= 2
    assert all(len(part) <= 4000 for part in text_parts)
    assert all(part.count("<b>") == part.count("</b>") for part in text_parts)


def test_long_html_attribute_never_exceeds_transport_limit():
    from event_delivery_worker import _split_html_text

    chunks = _split_html_text('<a href="https://example.test/' + ('x' * 10000) + '">ссылка</a>')

    assert chunks
    assert all(len(chunk) <= 4000 for chunk in chunks)


@pytest.mark.asyncio
async def test_single_media_caption_uses_valid_delivery_part_kind(session_factory):
    now = datetime.utcnow()
    async with session_factory() as session:
        session.add(User(id=9501, first_name="Single media user"))
        scenario = EventScenario(
            code="single_media_caption_event",
            event_type="webinar",
            title="Single media",
            starts_at=now + timedelta(days=1),
            duration_minutes=120,
            link="https://example.test/single",
            description="Короткое описание",
            is_active=True,
        )
        session.add(scenario)
        await session.flush()
        session.add(EventScenarioMedia(
            event_id=scenario.id,
            file_type="photo",
            file_id="single-photo",
            max_token="single-photo-max",
            sort_order=0,
        ))
        await session.commit()
    registration = await register_user_for_event(session_factory, 9501, "single_media_caption_event", now=now)
    claimed = await claim_pending_deliveries(session_factory, now=now)
    bot = MagicMock()
    bot.send_photo = AsyncMock(return_value=MagicMock(message_id=21))
    assert await process_single_delivery(
        session_factory,
        registration.immediate_delivery.id,
        claimed[0][1],
        bot=bot,
        now=now,
    ) == "sent"
    async with session_factory() as session:
        part = await session.scalar(select(EventDeliveryPart).where(
            EventDeliveryPart.delivery_id == registration.immediate_delivery.id
        ))
        assert part.part_kind == "media"
        assert part.status == "sent"


@pytest.mark.asyncio
async def test_max_consecutive_messages_rate_limiting_sleep_pause(session_factory):
    """Verifies that consecutive messages on MAX respect the 0.5s rate limit pause."""
    from event_delivery_worker import send_test_delivery_plan
    now = datetime.utcnow()
    async with session_factory() as session:
        session.add(User(id=9801, first_name="RateLimit User"))
        scenario = EventScenario(
            code="max_rate_limit_event",
            event_type="webinar",
            title="Rate Limit Webinar",
            starts_at=now + timedelta(days=1),
            duration_minutes=120,
            link="https://example.test/rate-limit",
            description="Текст для проверки rate limit",
            media_mode="separate",
            media_order="media_top",
            is_active=True,
        )
        session.add(scenario)
        await session.flush()
        session.add(EventScenarioMedia(
            event_id=scenario.id,
            file_type="photo",
            file_id="photo-1",
            max_token="token-1",
            sort_order=0,
        ))
        await session.commit()

    registration = await register_user_for_event(
        session_factory,
        9801,
        "max_rate_limit_event",
        platform="max",
        now=now,
    )
    claimed = await claim_pending_deliveries(session_factory, now=now)
    assert claimed

    max_client = AsyncMock()
    max_client.send_message = AsyncMock(return_value={"message_id": "max_msg_1"})

    sleep_calls = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    with patch("event_delivery_worker.asyncio.sleep", side_effect=mock_sleep):
        result = await process_single_delivery(
            session_factory,
            registration.immediate_delivery.id,
            claimed[0][1],
            max_client=max_client,
            now=now,
        )
        assert result == "sent"
        # Since this was a multi-part delivery on MAX (media + text + control),
        # asyncio.sleep(0.5) must have been called between consecutive parts
        assert 0.5 in sleep_calls

    # Also verify send_test_delivery_plan has the pause
    test_plan = [
        {"kind": "media", "items": [{"file_type": "photo", "max_token": "tok1"}]},
        {"kind": "text", "text": "Hello MAX"},
    ]
    test_sleep_calls = []
    with patch("event_delivery_worker.asyncio.sleep", side_effect=lambda d: test_sleep_calls.append(d)):
        await send_test_delivery_plan(
            test_plan,
            platform="max",
            user_id=9801,
            event_id=scenario.id,
            max_client=max_client,
        )
        assert 0.5 in test_sleep_calls
