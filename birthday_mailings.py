import asyncio
import logging
from datetime import datetime

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from sqlalchemy import select

from config import OWNER_IDS
from database import Mailing, MailingDeliveryLog, User, async_session_maker
from mailing_utils import (
    BIRTHDAY_MAILING_TYPE,
    get_mailing_audience_label,
    render_mailing_text,
    send_mailing_content,
)
from time_helpers import to_msk, utc_now
from error_reporting import notify_admins_about_error


log = logging.getLogger(__name__)


async def process_birthday_mailings(bot: Bot, *, now: datetime | None = None):
    now = now or utc_now()
    today = to_msk(now).date()
    delivery_date = today.isoformat()

    try:
        async with async_session_maker() as session:
            result = await session.execute(
                select(Mailing).where(
                    Mailing.recurring_type == BIRTHDAY_MAILING_TYPE,
                    Mailing.is_enabled == True,
                )
            )
            mailings = result.scalars().all()

            if not mailings:
                return

            users_result = await session.execute(
                select(User).where(
                    User.birth_day == today.day,
                    User.birth_month == today.month,
                )
            )
            users = users_result.scalars().all()

            if not users:
                return

            all_admin_ids = set(OWNER_IDS)

            for mailing in mailings:
                success_count = 0
                failure_count = 0
                attempted = 0
                mailing.start_time = now

                for user in users:
                    already_sent = await session.scalar(
                        select(MailingDeliveryLog.id).where(
                            MailingDeliveryLog.mailing_id == mailing.id,
                            MailingDeliveryLog.user_id == user.id,
                            MailingDeliveryLog.delivery_date == delivery_date,
                        )
                    )
                    if already_sent:
                        continue

                    attempted += 1
                    rendered_text = render_mailing_text(mailing.text, user)
                    error_text = None
                    status = "sent"

                    try:
                        await send_mailing_content(bot, user.id, mailing, rendered_text=rendered_text)
                        success_count += 1
                    except TelegramRetryAfter as exc:
                        await asyncio.sleep(exc.retry_after)
                        try:
                            await send_mailing_content(bot, user.id, mailing, rendered_text=rendered_text)
                            success_count += 1
                        except Exception as retry_exc:
                            status = "failed"
                            error_text = str(retry_exc)
                            failure_count += 1
                    except (TelegramForbiddenError, TelegramBadRequest) as exc:
                        status = "failed"
                        error_text = str(exc)
                        failure_count += 1
                    except Exception as exc:
                        status = "failed"
                        error_text = str(exc)
                        failure_count += 1

                    session.add(
                        MailingDeliveryLog(
                            mailing_id=mailing.id,
                            user_id=user.id,
                            delivery_date=delivery_date,
                            status=status,
                            error=error_text,
                            sent_at=now,
                        )
                    )
                    await session.commit()
                    await asyncio.sleep(0.05)

                if attempted == 0:
                    continue

                mailing.end_time = utc_now()
                mailing.success_count = (mailing.success_count or 0) + success_count
                mailing.failure_count = (mailing.failure_count or 0) + failure_count
                await session.commit()

                report = (
                    f"🎂 ДР-рассылка #{mailing.id} завершена\n"
                    f"Аудитория: {get_mailing_audience_label(mailing.target_audience)}\n"
                    f"Дата: {delivery_date}\n"
                    f"Успешно: {success_count}\n"
                    f"Ошибки: {failure_count}"
                )
                for admin_id in all_admin_ids:
                    try:
                        await bot.send_message(admin_id, report)
                    except Exception:
                        pass

                log.info(
                    "Birthday mailing processed mailing_id=%s delivery_date=%s success=%s failure=%s",
                    mailing.id,
                    delivery_date,
                    success_count,
                    failure_count,
                )
    except Exception as exc:
        log.error("Birthday mailing fatal error: %s", exc, exc_info=exc)
        await notify_admins_about_error(
            bot,
            title="Сбой ДР-рассылки",
            stage="process_birthday_mailings",
            details=str(exc),
            extra={"delivery_date": delivery_date},
            exception=exc,
            logger=log,
        )
        raise
