from __future__ import annotations

import html
import math

from sqlalchemy import exists, func, not_, or_, select

from ..api import MaxApiClient
from ..keyboards import (
    admin_mailing_audience_keyboard,
    admin_mailing_details_keyboard,
    admin_mailing_history_keyboard,
    admin_mailing_menu_keyboard,
    admin_mailing_preview_keyboard,
)
from ..logging_utils import get_bot_logger
from ..legacy import Mailing, Message as DBMessage, User, UserSubscription, async_session_maker
from ..models import IncomingMessage
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


async def choose_audience(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, audience: str) -> None:
    if audience not in AUDIENCE_NAMES:
        await client.send_message(chat_id=chat_id, text="Неизвестная аудитория.")
        return
    await states.set(user_id, chat_id, "admin_mailing_text", {"audience": audience})
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"Выбрана аудитория: <b>{html.escape(AUDIENCE_NAMES[audience])}</b>\n\n"
            "Отправьте текст рассылки или медиа с подписью одним сообщением."
        ),
    )


def _preview_attachments(media_type: str | None, media_token: str | None, include_keyboard: bool = True) -> list[dict] | None:
    rows: list[dict] = []
    if media_type and media_token:
        rows.append({"type": media_type, "payload": {"token": media_token}})
    if include_keyboard:
        rows.extend(admin_mailing_preview_keyboard())
    return rows or None


async def save_input(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, message: IncomingMessage) -> None:
    mailing_text = (message.text or "").strip()
    media_type = message.media_type
    media_token = message.media_token
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
            "media_type": media_type,
            "media_token": media_token,
        },
    )
    preview = mailing_text[:3000] + ("..." if len(mailing_text) > 3000 else "")
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


async def restart_text_edit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    snapshot = await states.get(user_id)
    audience = snapshot.data.get("audience") if snapshot else None
    if not audience:
        await client.send_message(chat_id=chat_id, text="Состояние рассылки потеряно.")
        return
    await states.set(user_id, chat_id, "admin_mailing_text", {"audience": audience})
    await client.send_message(chat_id=chat_id, text="Отправьте новый текст или медиа с подписью.")


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
    media_type = snapshot.data.get("media_type")
    media_token = snapshot.data.get("media_token")
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
            await client.send_message(user_id=max_api_user_id, text=mailing_text, attachments=attachments)
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
    is_enabled = getattr(mailing, 'is_enabled', True)
    preview = mailing.text[:3000] + ("..." if mailing.text and len(mailing.text) > 3000 else "")
    text = (
        f"<b>Рассылка #{mailing.id}</b>\n\n"
        f"<b>Аудитория:</b> {html.escape(AUDIENCE_NAMES.get(mailing.target_audience or '', mailing.target_audience or ''))}\n"
        f"<b>Статус:</b> {STATUS_NAMES.get(mailing.status, mailing.status)}\n"
        f"<b>Активна:</b> {'✅' if is_enabled else '❌'}\n"
        f"<b>Создана:</b> {created}\n"
        f"<b>Старт:</b> {started}\n"
        f"<b>Завершение:</b> {ended}\n"
        f"<b>Успешно:</b> {mailing.success_count}\n"
        f"<b>Ошибки:</b> {mailing.failure_count}\n"
        f"<b>Медиа:</b> {html.escape(mailing.media_file_type or 'нет')}\n\n"
        f"<pre><code>{html.escape(preview or 'Нет текста')}</code></pre>"
    )
    toggle_label = "❌ Выключить" if is_enabled else "✅ Включить"
    detail_rows = [
        [callback_button("📤 Тест (отправить себе)", f"mailing_send_test_{mailing_id}")],
        [callback_button(toggle_label, f"mailing_toggle_enabled_{mailing_id}")],
        [callback_button("⬅️ К истории", "mailing_history_page_0")],
    ]
    attachments = []
    if mailing.media_file_type and mailing.media_file_id:
        attachments.append({"type": mailing.media_file_type, "payload": {"token": mailing.media_file_id}})
    attachments.extend(inline_keyboard(detail_rows))
    await client.send_message(chat_id=chat_id, text=text, attachments=attachments)


async def send_test(client: MaxApiClient, chat_id: int, user_id: int, mailing_id: int) -> None:
    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)
    if not mailing:
        await client.send_message(chat_id=chat_id, text="Рассылка не найдена.")
        return

    from ..formatting import markdown_to_html
    text = mailing.text or ""
    formatted = markdown_to_html(text)

    attachments = []
    if mailing.media_file_type and mailing.media_file_id:
        attachments.append({"type": mailing.media_file_type, "payload": {"token": mailing.media_file_id}})

    try:
        await client.send_message(user_id=user_id, text=f"🧪 <b>Тест рассылки:</b>\n\n{formatted}", attachments=attachments or None)
        await client.send_message(chat_id=chat_id, text=f"✅ Тест рассылки #{mailing_id} отправлен вам.")
    except Exception:
        log.exception("Failed to send test mailing mailing_id=%s user_id=%s", mailing_id, user_id)
        await client.send_message(chat_id=chat_id, text="❌ Не удалось отправить тест.")


async def toggle_enabled(client: MaxApiClient, chat_id: int, mailing_id: int) -> None:
    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)
        if not mailing:
            await client.send_message(chat_id=chat_id, text="Рассылка не найдена.")
            return
        current = getattr(mailing, 'is_enabled', True)
        mailing.is_enabled = not current
        await session.commit()
    await show_details(client, chat_id, mailing_id)
