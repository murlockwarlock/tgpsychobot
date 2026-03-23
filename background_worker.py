import asyncio
import logging

from sqlalchemy import select, update, func
from sqlalchemy.orm import selectinload
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter

from database import (async_session_maker, IndexingQueue, KnowledgeBase, Mailing, User,
                      UserSubscription, Message as DBMessage)
from file_parser import parse_file
from mailing_utils import get_mailing_audience_label, send_mailing_content
from time_helpers import utc_now
from vector_store import update_vector_index
import keyboards as kb


class DailyLimitError(Exception):
    pass


class InsufficientBalanceError(Exception):
    pass


async def process_queue(bot: Bot):
    logging.info("Background worker started.")
    while True:
        try:
            async with async_session_maker() as session:
                stmt = select(IndexingQueue).where(IndexingQueue.status == 'pending').order_by(
                    IndexingQueue.created_at).limit(1)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

                if not job:
                    await asyncio.sleep(10)
                    continue

                logging.info(f"Processing job {job.id} for file {job.filename}")
                await session.execute(
                    update(IndexingQueue).where(IndexingQueue.id == job.id).values(status='processing'))
                await session.commit()

                progress_msg = None
                try:
                    progress_msg = await bot.send_message(job.uploader_id,
                                                          f"⏳ Начинаю обработку файла: `{job.filename}`...")

                    await asyncio.sleep(1)
                    await bot.edit_message_text(
                        text=f"⏳ Загружаю файл: `{job.filename}`...",
                        chat_id=job.uploader_id,
                        message_id=progress_msg.message_id
                    )
                    file_info = await bot.get_file(job.file_id)
                    file_bytes = await bot.download_file(file_info.file_path)

                    await asyncio.sleep(1)
                    await bot.edit_message_text(
                        text=f"⏳ Анализирую и извлекаю текст из `{job.filename}`...",
                        chat_id=job.uploader_id,
                        message_id=progress_msg.message_id
                    )
                    indexed_content = await parse_file(file_bytes, job.filename)
                    if not indexed_content:
                        raise ValueError("Не удалось извлечь текст из файла.")

                    new_kb_entry = KnowledgeBase(filename=job.filename, indexed_content=indexed_content)
                    session.add(new_kb_entry)
                    await session.flush()

                    await bot.edit_message_text(
                        text=f"⏳ Создаю векторы и индексирую `{job.filename}`...",
                        chat_id=job.uploader_id,
                        message_id=progress_msg.message_id
                    )
                    await update_vector_index(new_kb_entry.id, indexed_content)

                    await session.execute(
                        update(IndexingQueue).where(IndexingQueue.id == job.id).values(status='completed'))
                    await session.commit()
                    logging.info(f"Job {job.id} completed successfully.")

                    await bot.delete_message(chat_id=job.uploader_id, message_id=progress_msg.message_id)
                    await bot.send_message(job.uploader_id,
                                           f"✅ Файл `{job.filename}` полностью обработан и добавлен в Базу Знаний.")

                except (DailyLimitError, InsufficientBalanceError) as api_error:
                    error_type = 'paused_limit' if isinstance(api_error, DailyLimitError) else 'paused_balance'
                    await session.execute(
                        update(IndexingQueue).where(IndexingQueue.id == job.id).values(status=error_type))
                    await session.commit()

                    if isinstance(api_error, DailyLimitError):
                        logging.warning(f"Job {job.id} paused due to daily limit. Worker is sleeping.")
                        msg_text = f"⌛️ Достигнут дневной лимит. Обработка файла `{job.filename}` будет возобновлена позже. Очередь приостановлена."
                        await bot.edit_message_text(
                            text=msg_text,
                            chat_id=job.uploader_id,
                            message_id=progress_msg.message_id
                        )
                        await asyncio.sleep(24 * 60 * 60)
                        async with async_session_maker() as resume_session:
                            await resume_session.execute(
                                update(IndexingQueue).where(IndexingQueue.status == 'paused_limit').values(
                                    status='pending'))
                            await resume_session.commit()
                        await bot.send_message(job.uploader_id, "🌞 Лимиты обновлены, возобновляю обработку файлов.")

                    elif isinstance(api_error, InsufficientBalanceError):
                        logging.error(f"Job {job.id} paused due to zero balance.")
                        msg_text = f"❗️ Закончился баланс API! Обработка файла `{job.filename}` и вся очередь остановлены."
                        await bot.edit_message_text(
                            text=msg_text,
                            chat_id=job.uploader_id,
                            message_id=progress_msg.message_id
                        )
                        await bot.send_message(job.uploader_id, "Пожалуйста, пополните баланс и нажмите кнопку ниже.",
                                               reply_markup=kb.balance_refilled_keyboard())

                except (TelegramBadRequest, Exception) as e:
                    await session.rollback()
                    await session.execute(
                        update(IndexingQueue).where(IndexingQueue.id == job.id).values(status='failed'))
                    await session.commit()
                    logging.error(f"Job {job.id} failed: {e}")
                    if progress_msg:
                        await bot.edit_message_text(
                            text=f"❌ Ошибка при обработке `{job.filename}`: {e}",
                            chat_id=job.uploader_id,
                            message_id=progress_msg.message_id
                        )
        except Exception as e:
            logging.critical(f"Critical error in background worker: {e}. Restarting loop in 60s.")
            await asyncio.sleep(60)


async def process_mailings(bot: Bot):
    logging.info("Mailing worker started.")
    while True:
        await asyncio.sleep(15)
        async with async_session_maker() as session:
            stmt = (
                select(Mailing)
                .where(Mailing.status == 'pending', Mailing.recurring_type.is_(None))
                .order_by(Mailing.created_at)
                .limit(1)
            )
            result = await session.execute(stmt)
            mailing = result.scalar_one_or_none()
            if not mailing:
                continue

            mailing.status = 'sending'
            mailing.start_time = utc_now()
            await session.commit()

            audience = mailing.target_audience
            target_users_stmt = None
            if audience == "all":
                target_users_stmt = select(User.id)
            elif audience == "self":
                target_users_stmt = select(User.id).where(User.id == mailing.creator_id)
            elif audience == "no_dialogue":
                subquery = select(DBMessage.user_id, func.count(DBMessage.id).label("msg_count")).group_by(
                    DBMessage.user_id).subquery()
                target_users_stmt = select(User.id).outerjoin(subquery, User.id == subquery.c.user_id).where(
                    (subquery.c.msg_count == None) | (subquery.c.msg_count <= 1))
            elif audience == "no_subscription":
                target_users_stmt = select(User.id).outerjoin(UserSubscription).where(UserSubscription.id == None)
            elif audience == "active_subscription":
                target_users_stmt = select(User.id).join(UserSubscription).where(
                    UserSubscription.end_date > datetime.utcnow())
            elif audience == "inactive_subscription":
                target_users_stmt = select(User.id).join(UserSubscription).where(
                    UserSubscription.end_date <= datetime.utcnow())

            if target_users_stmt is None:
                mailing.status = 'failed'
                await session.commit()
                continue

            user_ids = (await session.execute(target_users_stmt)).scalars().all()
            success_count, failure_count = 0, 0

            for user_id in user_ids:
                try:
                    await send_mailing_content(bot, user_id, mailing)
                    success_count += 1
                except TelegramRetryAfter as e:
                    await asyncio.sleep(e.retry_after)
                    try:
                        await send_mailing_content(bot, user_id, mailing)
                        success_count += 1
                    except:
                        failure_count += 1
                except (TelegramForbiddenError, TelegramBadRequest):
                    failure_count += 1
                except Exception:
                    failure_count += 1

                await asyncio.sleep(0.05)

            mailing.status = 'completed'
            mailing.end_time = utc_now()
            mailing.success_count = success_count
            mailing.failure_count = failure_count
            await session.commit()

            from database import get_all_admin_ids
            report = (
                f"✅ Рассылка #{mailing.id} завершена!\n"
                f"Аудитория: {get_mailing_audience_label(audience)}\n"
                f"Успешно: {success_count}\nОшибки: {failure_count}"
            )
            for admin_id in await get_all_admin_ids():
                try:
                    await bot.send_message(admin_id, report)
                except:
                    pass
