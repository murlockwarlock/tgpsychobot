import os
import io
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

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
from event_delivery_worker import _build_delivery_plan, claim_pending_deliveries, process_single_delivery, PreTransmissionError
from event_service import (
    handle_event_callback_max,
    handle_event_callback_telegram,
    handle_event_start_max,
    handle_event_start_telegram,
    register_user_for_event,
    render_event_message,
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
    assert any("не удалось подготовить" in call.kwargs["text"] for call in bot.send_message.call_args_list)
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
