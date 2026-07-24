from __future__ import annotations

import asyncio
import html
import math
import time

from sqlalchemy import exists, func, not_, or_, select

from ..api import MaxApiClient
from ..keyboards import (
    admin_mailing_audience_keyboard,
    admin_mailing_details_keyboard,
    admin_mailing_history_keyboard,
    admin_mailing_input_keyboard,
    admin_mailing_menu_keyboard,
    admin_mailing_preview_keyboard,
)
from ..logging_utils import get_bot_logger
from ..legacy import Mailing, Message as DBMessage, User, UserSubscription, async_session_maker
from ..models import MAX_ID_OFFSET, REUSABLE_ATTACHMENT_TYPES, IncomingMessage, parse_message
from ..storage import StateStore
from ..time_utils import utc_now


PAGE_SIZE = 10
log = get_bot_logger("mailing")

AUDIENCE_NAMES = {
    "all": "Всем пользователям",
    "no_dialogue": "Кто не начал диалог",
    "no_subscription": "Кто ни разу не платил",
    "active_subscription": "Активным подписчикам",
    "inactive_subscription": "Без активной подписки",
    "self": "Только себе",
}

STATUS_NAMES = {
    "pending": "⏳ Ожидает",
    "sending": "🚀 Отправляется",
    "completed": "✅ Завершена",
    "failed": "❌ Ошибка",
}


async def show_menu(client: MaxApiClient, chat_id: int) -> None:
    await client.send_message(
        chat_id=chat_id,
        text="✉️ <b>Рассылки</b>\n\nСоздание, запуск и история рассылок в MAX.",
        attachments=admin_mailing_menu_keyboard(),
    )


async def start_create(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_mailing_audience", {})
    await client.send_message(
        chat_id=chat_id,
        text="Выберите аудиторию для рассылки.",
        attachments=admin_mailing_audience_keyboard(),
    )


async def choose_audience(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, audience: str) -> str | None:
    if audience not in AUDIENCE_NAMES:
        await client.send_message(chat_id=chat_id, text="Неизвестная аудитория.")
        return
    request_id = str(time.time_ns())
    await states.set(
        user_id,
        chat_id,
        "admin_mailing_text",
        {
            "audience": audience,
            "input_after_ms": int(time.time() * 1000),
            "input_request_id": request_id,
        },
    )
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"Выбрана аудитория: <b>{html.escape(AUDIENCE_NAMES[audience])}</b>\n\n"
            "Отправьте текст рассылки или медиа с подписью одним сообщением.\n\n"
            "Пересланные сообщения бот подхватит автоматически. Если предпросмотр не появился, "
            "нажмите «Взять последнее сообщение»."
        ),
        attachments=admin_mailing_input_keyboard(),
    )
    return request_id


def _normalize_media(media_type: str | None, media_token: str | None) -> tuple[str | None, str | None]:
    normalized_type = str(media_type or "").lower()
    if normalized_type not in REUSABLE_ATTACHMENT_TYPES or not media_token:
        return None, None
    return normalized_type, str(media_token)


def _preview_attachments(media_type: str | None, media_token: str | None, include_keyboard: bool = True) -> list[dict] | None:
    rows: list[dict] = []
    media_type, media_token = _normalize_media(media_type, media_token)
    if media_type and media_token:
        rows.append({"type": media_type, "payload": {"token": media_token}})
    if include_keyboard:
        rows.extend(admin_mailing_preview_keyboard())
    return rows or None


async def save_input(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, message: IncomingMessage) -> None:
    mailing_text = (message.text or "").strip()
    formatted_text = (message.html_text or "").strip() or None
    media_type, media_token = _normalize_media(message.media_type, message.media_token)
    if not mailing_text and not (media_type and media_token):
        await client.send_message(chat_id=chat_id, text="Отправьте текст, медиа или медиа с подписью.")
        return
    snapshot = await states.get(user_id)
    audience = snapshot.data.get("audience") if snapshot else None
    if not audience:
        await client.send_message(chat_id=chat_id, text="Состояние рассылки потеряно.")
        return
    await states.set(
        user_id,
        chat_id,
        "admin_mailing_preview",
        {
            "audience": audience,
            "text": mailing_text,
            "formatted_text": formatted_text,
            "media_type": media_type,
            "media_token": media_token,
        },
    )
    preview = mailing_text[:3000] + ("..." if len(mailing_text) > 3000 else "")
    try:
        await client.send_message(
            chat_id=chat_id,
            text=(
                "<b>Предпросмотр рассылки</b>\n\n"
                f"<b>Аудитория:</b> {html.escape(AUDIENCE_NAMES[audience])}\n\n"
                f"<b>Медиа:</b> {html.escape(media_type or 'нет')}\n\n"
                f"<pre><code>{html.escape(preview or 'Без текста')}</code></pre>"
            ),
            attachments=_preview_attachments(media_type, media_token, include_keyboard=True),
        )
    except Exception:
        await states.set(user_id, chat_id, snapshot.state, dict(snapshot.data))
        log.exception("Failed to show mailing preview user_id=%s chat_id=%s", user_id, chat_id)
        try:
            await client.send_message(
                chat_id=chat_id,
                text="Не удалось сформировать предпросмотр. Сообщение не потеряно — попробуйте отправить его ещё раз.",
                attachments=admin_mailing_input_keyboard(),
            )
        except Exception:
            log.exception("Failed to notify about mailing preview error user_id=%s chat_id=%s", user_id, chat_id)


def _is_inbound_history_message(raw_message: dict, user_id: int) -> bool:
    recipient = raw_message.get("recipient") or {}
    raw_user_id = user_id - MAX_ID_OFFSET if user_id >= MAX_ID_OFFSET else user_id
    return recipient.get("user_id") != raw_user_id


async def _find_latest_history_input(
    client: MaxApiClient,
    chat_id: int,
    user_id: int,
    input_after_ms: int,
    *,
    shares_only: bool,
) -> IncomingMessage | None:
    result = await client.get_messages(chat_id, count=50)
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        try:
            timestamp = int(raw_message.get("timestamp") or 0)
        except (TypeError, ValueError):
            continue
        if timestamp < input_after_ms or not _is_inbound_history_message(raw_message, user_id):
            continue
        attachments = ((raw_message.get("body") or {}).get("attachments") or [])
        link = raw_message.get("link") or {}
        is_forward = isinstance(link, dict) and link.get("type") == "forward"
        if shares_only and not is_forward and not any(
            item.get("type") == "share" for item in attachments if isinstance(item, dict)
        ):
            continue
        message = parse_message({"message": raw_message})
        if message and ((message.text or "").strip() or (message.media_type and message.media_token)):
            return message
    return None


async def capture_latest_input(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    *,
    notify_if_missing: bool = True,
    shares_only: bool = False,
) -> bool:
    snapshot = await states.get(user_id)
    if not snapshot or snapshot.state != "admin_mailing_text":
        if notify_if_missing:
            await client.send_message(chat_id=chat_id, text="Сначала выберите аудиторию рассылки.")
        return False
    message = await _find_latest_history_input(
        client,
        chat_id,
        user_id,
        int(snapshot.data.get("input_after_ms") or 0),
        shares_only=shares_only,
    )
    if message is None:
        if notify_if_missing:
            await client.send_message(
                chat_id=chat_id,
                text="После выбора аудитории новых сообщений не найдено. Отправьте сообщение и нажмите кнопку ещё раз.",
                attachments=admin_mailing_input_keyboard(),
            )
        return False
    await save_input(client, states, chat_id, user_id, message)
    return True


async def watch_for_shared_input(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    request_id: str,
    *,
    poll_interval: float = 2.0,
    max_checks: int = 90,
) -> None:
    for _ in range(max_checks):
        snapshot = await states.get(user_id)
        if (
            not snapshot
            or snapshot.state != "admin_mailing_text"
            or snapshot.data.get("input_request_id") != request_id
        ):
            return
        try:
            if await capture_latest_input(
                client,
                states,
                chat_id,
                user_id,
                notify_if_missing=False,
                shares_only=True,
            ):
                return
        except Exception:
            log.exception("Failed to capture shared mailing input user_id=%s chat_id=%s", user_id, chat_id)
        await asyncio.sleep(poll_interval)


async def restart_text_edit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> str | None:
    snapshot = await states.get(user_id)
    audience = snapshot.data.get("audience") if snapshot else None
    if not audience:
        await client.send_message(chat_id=chat_id, text="Состояние рассылки потеряно.")
        return
    request_id = str(time.time_ns())
    await states.set(
        user_id,
        chat_id,
        "admin_mailing_text",
        {
            "audience": audience,
            "input_after_ms": int(time.time() * 1000),
            "input_request_id": request_id,
        },
    )
    await client.send_message(
        chat_id=chat_id,
        text="Отправьте новый текст или медиа с подписью.",
        attachments=admin_mailing_input_keyboard(),
    )
    return request_id


async def _get_recipient_ids(session, audience: str, user_id: int) -> list[int]:
    from ..models import MAX_ID_OFFSET
    if audience == "self":
        return [user_id]

    stmt = select(User.id).where(User.id >= MAX_ID_OFFSET)

    if audience == "all":
        return list((await session.execute(stmt)).scalars().all())

    if audience == "no_dialogue":
        subquery = select(DBMessage.id).where(DBMessage.user_id == User.id)
        return list((await session.execute(stmt.where(not_(exists(subquery))))).scalars().all())

    if audience == "no_subscription":
        subquery = select(UserSubscription.id).where(UserSubscription.user_id == User.id)
        return list((await session.execute(stmt.where(not_(exists(subquery))))).scalars().all())

    now = utc_now()

    if audience == "active_subscription":
        subquery = select(UserSubscription.id).where(UserSubscription.user_id == User.id, UserSubscription.end_date > now)
        return list((await session.execute(stmt.where(exists(subquery)))).scalars().all())

    if audience == "inactive_subscription":
        active_subquery = select(UserSubscription.id).where(UserSubscription.user_id == User.id, UserSubscription.end_date > now)
        any_subquery = select(UserSubscription.id).where(UserSubscription.user_id == User.id)
        return list(
            (
                await session.execute(
                    stmt.where(or_(not_(exists(any_subquery)), not_(exists(active_subquery))))
                )
            ).scalars().all()
        )

    return []


async def confirm_send(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    snapshot = await states.get(user_id)
    if not snapshot:
        await client.send_message(chat_id=chat_id, text="Состояние рассылки потеряно.")
        return
    audience = snapshot.data.get("audience")
    mailing_text = snapshot.data.get("text")
    formatted_text = snapshot.data.get("formatted_text")
    media_type, media_token = _normalize_media(
        snapshot.data.get("media_type"),
        snapshot.data.get("media_token"),
    )
    if not audience or (not mailing_text and not (media_type and media_token)):
        await client.send_message(chat_id=chat_id, text="Состояние рассылки потеряно.")
        return

    async with async_session_maker() as session:
        mailing = Mailing(
            text=mailing_text,
            media_file_id=media_token,
            media_file_type=media_type,
            target_audience=audience,
            creator_id=user_id,
            status="pending",
            success_count=0,
            failure_count=0,
        )
        session.add(mailing)
        await session.commit()
        mailing_id = mailing.id

        recipient_ids = await _get_recipient_ids(session, audience, user_id)
        mailing = await session.get(Mailing, mailing_id)
        mailing.status = "sending"
        mailing.start_time = utc_now()
        await session.commit()

    success_count = 0
    failure_count = 0
    for recipient_id in recipient_ids:
        try:
            attachments = _preview_attachments(media_type, media_token, include_keyboard=False)
            from ..models import MAX_ID_OFFSET
            max_api_user_id = recipient_id - MAX_ID_OFFSET if recipient_id >= MAX_ID_OFFSET else recipient_id
            await client.send_message(
                user_id=max_api_user_id,
                text=formatted_text or mailing_text,
                attachments=attachments,
                format_="html" if formatted_text else "",
            )
            success_count += 1
            log.info("Mailing delivered mailing_id=%s target_id=%s", mailing_id, recipient_id)
        except Exception:
            failure_count += 1
            log.exception("Mailing delivery failed mailing_id=%s target_id=%s", mailing_id, recipient_id)

    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)
        if mailing:
            mailing.success_count = success_count
            mailing.failure_count = failure_count
            mailing.end_time = utc_now()
            mailing.status = "completed" if failure_count == 0 else ("failed" if success_count == 0 else "completed")
            await session.commit()
            log.info(
                "Mailing completed mailing_id=%s status=%s success=%s failure=%s",
                mailing_id,
                mailing.status,
                success_count,
                failure_count,
            )

    await states.clear(user_id)
    await client.send_message(
        chat_id=chat_id,
        text=(
            "✅ Рассылка завершена.\n\n"
            f"<b>Аудитория:</b> {html.escape(AUDIENCE_NAMES[audience])}\n"
            f"<b>Получателей:</b> {len(recipient_ids)}\n"
            f"<b>Успешно:</b> {success_count}\n"
            f"<b>Ошибки:</b> {failure_count}"
        ),
        attachments=admin_mailing_menu_keyboard(),
    )


async def show_history(client: MaxApiClient, chat_id: int, page: int) -> None:
    from ..models import MAX_ID_OFFSET
    async with async_session_maker() as session:
        total_mailings = await session.scalar(
            select(func.count()).select_from(Mailing).where(Mailing.creator_id >= MAX_ID_OFFSET)
        ) or 0
        total_pages = max(1, math.ceil(total_mailings / PAGE_SIZE))
        page = max(0, min(page, total_pages - 1))
        mailings = (
            await session.execute(
                select(Mailing)
                .where(Mailing.creator_id >= MAX_ID_OFFSET)
                .order_by(Mailing.created_at.desc(), Mailing.id.desc())
                .offset(page * PAGE_SIZE)
                .limit(PAGE_SIZE)
            )
        ).scalars().all()
    text = "📜 История рассылок пуста." if not mailings and total_mailings == 0 else f"📜 <b>История рассылок</b>\n\nСтраница {page + 1}/{total_pages}"
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_mailing_history_keyboard(mailings, page, total_pages))


async def show_details(client: MaxApiClient, chat_id: int, mailing_id: int) -> None:
    from ..keyboards import callback_button, inline_keyboard
    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)
    if not mailing:
        await client.send_message(chat_id=chat_id, text="Рассылка не найдена.")
        return

    created = mailing.created_at.strftime("%d.%m.%Y %H:%M") if mailing.created_at else "N/A"
    started = mailing.start_time.strftime("%d.%m.%Y %H:%M") if mailing.start_time else "Еще не запускалась"
    ended = mailing.end_time.strftime("%d.%m.%Y %H:%M") if mailing.end_time else "N/A"
    preview = mailing.text[:3000] + ("..." if mailing.text and len(mailing.text) > 3000 else "")
    text = (
        f"<b>Рассылка #{mailing.id}</b>\n\n"
        f"<b>Аудитория:</b> {html.escape(AUDIENCE_NAMES.get(mailing.target_audience or '', mailing.target_audience or ''))}\n"
        f"<b>Статус:</b> {STATUS_NAMES.get(mailing.status, mailing.status)}\n"
        f"<b>Создана:</b> {created}\n"
        f"<b>Старт:</b> {started}\n"
        f"<b>Завершение:</b> {ended}\n"
        f"<b>Успешно:</b> {mailing.success_count}\n"
        f"<b>Ошибки:</b> {mailing.failure_count}\n"
        f"<b>Медиа:</b> {html.escape(mailing.media_file_type or 'нет')}\n\n"
        f"<pre><code>{html.escape(preview or 'Нет текста')}</code></pre>"
    )
    detail_rows = [
        [callback_button("⬅️ К истории", "mailing_history_page_0")],
    ]
    attachments = []
    media_type, media_token = _normalize_media(mailing.media_file_type, mailing.media_file_id)
    if media_type and media_token:
        attachments.append({"type": media_type, "payload": {"token": media_token}})
    attachments.extend(inline_keyboard(detail_rows))
    await client.send_message(chat_id=chat_id, text=text, attachments=attachments)
