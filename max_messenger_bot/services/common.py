from __future__ import annotations

import html
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import selectinload

from ..ai import AIServiceError, generate_image, get_ai_response, edit_image
from ..api import MaxApiClient
from ..formatting import markdown_to_html, split_text
from ..keyboards import build_main_menu, disclaimer_keyboard, inline_keyboard, main_menu_row, response_buttons_keyboard
from ..logging_utils import get_bot_logger
from ..legacy import (
    AIConfig,
    Content,
    Message as DBMessage,
    RandomMessage,
    SubscriptionConfig,
    Topic,
    User,
    UserSubscription,
    UserTopicState,
    async_session_maker,
)
from ..storage import MaxContentMedia, StateStore
from ..time_utils import utc_now
from memory_mode import normalize_memory_mode, start_new_dialogue
from response_buttons import ResponseButton, extract_response_buttons


log = get_bot_logger("common")


MSK = timezone(timedelta(hours=3))
TEST_START_DIRECTIVE_RE = re.compile(r"(?<!\w)\[?\s*(?:START|RUN)\\?_TEST\s*\]?(?!\w)", re.IGNORECASE)


def _extract_test_start_directive(text: str | None) -> tuple[bool, str]:
    raw = text or ""
    has_directive = bool(TEST_START_DIRECTIVE_RE.search(raw))
    if not has_directive:
        return False, raw.strip()
    clean_text = TEST_START_DIRECTIVE_RE.sub("", raw)
    clean_text = re.sub(r"\s+([.,!?;:])", r"\1", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return True, clean_text.strip(" \t\r\n-—–:;")


async def _notify_referrer_about_registration(
    client: MaxApiClient,
    referrer_id: int,
    bonus_days: int,
) -> None:
    from ..models import MAX_ID_OFFSET

    await client.send_message(
        user_id=referrer_id - MAX_ID_OFFSET,
        text=(
            "🎉 По вашей реферальной ссылке зарегистрировался новый пользователь!\n"
            f"Вам начислено <b>{bonus_days} бонусных дн.</b> к доступу. "
            "Спасибо, что рекомендуете нас!"
        ),
    )


def _response_buttons_keyboard(response_buttons: list[list[ResponseButton]] | None) -> list[dict]:
    return response_buttons_keyboard(response_buttons or [], include_main_menu=True)


async def _send_ai_text(
    client: MaxApiClient,
    chat_id: int,
    thinking_message_id: str | None,
    chunks: list[str],
    response_buttons: list[list[ResponseButton]] | None = None,
) -> None:
    main_menu_kb = _response_buttons_keyboard(response_buttons)
    if not chunks:
        return

    if len(chunks) == 1:
        if thinking_message_id:
            await client.edit_message(
                thinking_message_id,
                text=chunks[0],
                attachments=main_menu_kb,
            )
        else:
            await client.send_message(chat_id=chat_id, text=chunks[0], attachments=main_menu_kb)
        return

    if thinking_message_id:
        await client.edit_message(
            thinking_message_id,
            text=chunks[0],
            attachments=None,
        )
        remaining_chunks = chunks[1:]
    else:
        remaining_chunks = chunks

    for i, chunk in enumerate(remaining_chunks):
        is_last = (i == len(remaining_chunks) - 1)
        kb = main_menu_kb if is_last else None
        await client.send_message(chat_id=chat_id, text=chunk, attachments=kb)


async def is_admin(user_id: int) -> bool:
    from ..settings import apply_legacy_env_defaults
    from config import OWNER_IDS  # type: ignore

    apply_legacy_env_defaults()
    if user_id in OWNER_IDS:
        return True
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        return bool(user and user.is_admin)


async def notify_telegram_admins(text: str) -> None:
    import os
    bot_token = os.getenv("BOT_TOKEN")
    if bot_token:
        from aiogram import Bot
        from database import get_all_admin_ids, SubscriptionConfig
        
        async with async_session_maker() as session:
            config = await session.get(SubscriptionConfig, 1)
            if not config or not config.notifications_enabled:
                return

        async with Bot(token=bot_token) as bot:
            for admin_id in await get_all_admin_ids():
                try:
                    await bot.send_message(admin_id, text, parse_mode="HTML")
                except Exception:
                    pass


async def ensure_user(user_id: int, username: str | None, full_name: str) -> User:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if user:
            if username and user.username != username:
                user.username = username
                await session.commit()
            return user
        user = User(
            id=user_id,
            username=username,
            first_name=full_name,
            is_admin=await is_admin(user_id),
        )
        session.add(user)
        await session.commit()
        return user


async def get_random_message_by_topic(topic_id: int | None) -> str | None:
    if not topic_id:
        return None
    async with async_session_maker() as session:
        return await session.scalar(
            select(RandomMessage.content)
            .where(RandomMessage.topic_id == topic_id)
            .order_by(func.random())
            .limit(1)
        )


async def get_content(content_key: str) -> Content | None:
    async with async_session_maker() as session:
        return await session.get(Content, content_key)


async def get_content_attachments(content_key: str) -> list[dict]:
    async with async_session_maker() as session:
        media_rows = (
            await session.execute(
                select(MaxContentMedia).where(MaxContentMedia.content_key == content_key).order_by(MaxContentMedia.id.asc())
            )
        ).scalars().all()
    attachments = []
    for row in media_rows:
        mtype = "image" if row.media_type == "photo" else row.media_type
        attachments.append({"type": mtype, "payload": {"token": row.token}})
    return attachments


async def send_main_menu(client: MaxApiClient, chat_id: int, text: str = "Главное меню:", user_id: int | None = None) -> None:
    target_user_id = user_id or chat_id
    await client.send_message(chat_id=chat_id, text=text, attachments=await build_main_menu(target_user_id))


async def show_help(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    if not await is_admin(user_id):
        text = (
            "👋 Я персональный ИИ-помощник.\n\n"
            "Пишите вопрос в чат или пользуйтесь кнопками меню."
        )
    else:
        text = (
            "<b>Режим администратора</b>\n\n"
            "Доступны основные экраны: статистика, подписки, темы, рефералка и тест. "
            "Для MAX я вынес логику в отдельный модуль, поэтому административные сценарии будут расширяться поэтапно без возврата к монолиту."
        )
    await client.send_message(chat_id=chat_id, text=text)


async def show_start_screen(client: MaxApiClient, chat_id: int, user_id: int, start_payload: str | None = None, states: StateStore | None = None) -> None:
    referrer_notification: tuple[int, int] | None = None
    async with async_session_maker() as session:
        user = await session.get(User, user_id, options=[selectinload(User.current_topic), selectinload(User.subscription)])
        if not user:
            return

        welcome_bonus_text = ""
        is_new_trial_needed = user.subscription is None
        sub_config = await session.get(SubscriptionConfig, 1)
        is_probably_new_user = (
            user.subscription is None
            and not user.referred_by
            and not user.name
            and (user.current_dialogue_id or 1) == 1
            and user.current_topic_id is None
        )

        if start_payload and start_payload.startswith("ref_") and is_probably_new_user and sub_config and sub_config.referral_enabled:
            try:
                referrer_id = int(start_payload.split("_", 1)[1])
            except ValueError:
                referrer_id = 0
            # If the referral link used the original MAX ID (without offset), add offset for DB lookup
            from ..models import MAX_ID_OFFSET
            if 0 < referrer_id < MAX_ID_OFFSET:
                referrer_id += MAX_ID_OFFSET
            if referrer_id and referrer_id != user_id:
                referrer = await session.get(User, referrer_id, options=[selectinload(User.subscription)])
                if referrer:
                    now = utc_now()
                    user.referred_by = referrer_id
                    if sub_config.referral_bonus_days_referral > 0:
                        session.add(
                            UserSubscription(
                                user_id=user.id,
                                plan_id=None,
                                start_date=now,
                                end_date=now + timedelta(days=sub_config.referral_bonus_days_referral),
                                auto_renewal=False,
                                payment_provider="Trial Referral",
                                payment_attempt_count=0,
                                discount_percent=0,
                            )
                        )
                        welcome_bonus_text = (
                            f"🎁 <b>Вам начислено {sub_config.referral_bonus_days_referral} бонусных дн.</b>\n"
                            "Регистрация выполнена по пригласительной ссылке."
                        )
                        is_new_trial_needed = False
                    if sub_config.referral_bonus_days_referrer > 0:
                        if referrer.subscription and referrer.subscription.end_date > now:
                            referrer.subscription.end_date += timedelta(days=sub_config.referral_bonus_days_referrer)
                        elif referrer.subscription:
                            referrer.subscription.plan_id = None
                            referrer.subscription.start_date = now
                            referrer.subscription.end_date = now + timedelta(days=sub_config.referral_bonus_days_referrer)
                            referrer.subscription.payment_provider = "Trial Referral Bonus"
                            referrer.subscription.auto_renewal = False
                            referrer.subscription.payment_attempt_count = 0
                        else:
                            session.add(
                                UserSubscription(
                                    user_id=referrer_id,
                                    plan_id=None,
                                    start_date=now,
                                    end_date=now + timedelta(days=sub_config.referral_bonus_days_referrer),
                                    auto_renewal=False,
                                    payment_provider="Trial Referral Bonus",
                                    payment_attempt_count=0,
                                    discount_percent=0,
                                )
                            )
                        referrer_notification = (
                            referrer_id,
                            sub_config.referral_bonus_days_referrer,
                        )
                    await session.commit()

        if is_new_trial_needed and sub_config and sub_config.subscriptions_enabled and sub_config.welcome_bonus_days > 0:
            now = utc_now()
            session.add(
                UserSubscription(
                    user_id=user.id,
                    plan_id=None,
                    start_date=now,
                    end_date=now + timedelta(days=sub_config.welcome_bonus_days),
                    auto_renewal=False,
                    payment_provider="Trial Welcome",
                    payment_attempt_count=0,
                    discount_percent=0,
                )
            )
            welcome_bonus_text = (
                f"🎁 <b>Вам начислен приветственный бонус!</b>\n"
                f"Бесплатный доступ на {sub_config.welcome_bonus_days} дн."
            )
            await session.commit()

    if referrer_notification:
        referrer_id, bonus_days = referrer_notification
        try:
            await _notify_referrer_about_registration(client, referrer_id, bonus_days)
        except Exception:
            log.exception(
                "Failed to notify MAX referrer user_id=%s about registration",
                referrer_id,
            )

    if start_payload:
        if start_payload == "sub":
            from .subscriptions import show_subscription_info

            await show_subscription_info(client, chat_id, user_id)
            return
        if start_payload == "test":
            from .tests import start_test

            await start_test(client, chat_id, user_id, states)
            return
        if start_payload.startswith("topic_"):
            from .topics import select_topic

            await select_topic(client, chat_id, user_id, int(start_payload.split("_", 1)[1]))
            return
        content = await get_content(start_payload)
        if content and content.is_visible:
            from ..formatting import translate_telegram_links_to_max
            content_text = translate_telegram_links_to_max(content.text_content) or "Раздел пока пуст."
            await client.send_message(
                chat_id=chat_id,
                text=content_text,
                attachments=await get_content_attachments(start_payload) or None,
            )
            await send_main_menu(client, chat_id, user_id=user_id)
            return

    start_content = await get_content("start_message")
    from ..formatting import translate_telegram_links_to_max
    if start_content and start_content.text_content:
        await client.send_message(
            chat_id=chat_id,
            text=translate_telegram_links_to_max(start_content.text_content),
            attachments=await get_content_attachments("start_message") or None,
        )
    else:
        await client.send_message(chat_id=chat_id, text="Здравствуйте. Бот в MAX готов к работе.")
    if welcome_bonus_text:
        await client.send_message(chat_id=chat_id, text=translate_telegram_links_to_max(welcome_bonus_text))
    await send_main_menu(client, chat_id, user_id=user_id)


async def reset_dialogue(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if user:
            config = await session.get(AIConfig, 1)
            await start_new_dialogue(session, user, user.current_topic_id or 0, normalize_memory_mode(config))
            await session.execute(
                delete(DBMessage).where(
                    DBMessage.user_id == user_id,
                    DBMessage.dialogue_id == user.current_dialogue_id - 1,
                )
            )
            await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Память очищена.", attachments=inline_keyboard([main_menu_row()]))


async def ensure_access_before_chat(client: MaxApiClient, chat_id: int, user: User) -> bool:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
    if await is_admin(user.id):
        return True
    if config and config.subscriptions_enabled:
        if not user.subscription or user.subscription.end_date < utc_now():
            await client.send_message(
                chat_id=chat_id,
                text="Чтобы продолжить диалог, активируйте подписку или бонусные дни.",
                attachments=[{
                    "type": "inline_keyboard",
                    "payload": {
                        "buttons": [
                            [{"type": "callback", "text": "Открыть подписки", "payload": "show_subscription_info_from_chat"}],
                            main_menu_row(),
                        ]
                    },
                }],
            )
            return False
    return True


async def begin_onboarding(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, initial_prompt: str) -> None:
    await states.set(user_id, chat_id, "awaiting_name", {"initial_prompt": initial_prompt})
    await client.send_message(chat_id=chat_id, text="Прежде чем начнём, как мне к вам обращаться?")


async def save_user_message(user_id: int, prompt_text: str) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            return
        session.add(
            DBMessage(
                user_id=user_id,
                role="user",
                content=prompt_text,
                dialogue_id=user.current_dialogue_id,
                topic_id=user.current_topic_id,
            )
        )
        await session.commit()


async def save_ai_message(user_id: int, response_text: str) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            return
        session.add(
            DBMessage(
                user_id=user_id,
                role="assistant",
                content=response_text,
                dialogue_id=user.current_dialogue_id,
                topic_id=user.current_topic_id,
            )
        )
        await session.commit()


async def run_ai_dialogue(client: MaxApiClient, chat_id: int, user_id: int, prompt_text: str, states: StateStore | None = None) -> None:
    log.info("AI dialogue requested user_id=%s chat_id=%s", user_id, chat_id)
    await save_user_message(user_id, prompt_text)
    thinking = await client.send_message(chat_id=chat_id, text="🤖 Думаю...")
    thinking_message_id = ((thinking.get("message") or {}).get("mid") if isinstance(thinking, dict) else None)

    topic_phrase = None
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if user and user.current_topic_id:
            topic_phrase = await get_random_message_by_topic(user.current_topic_id)
    final_prompt = prompt_text
    if topic_phrase:
        final_prompt = f"{prompt_text}\n\nКонтекст для ответа: {topic_phrase}"

    try:
        response_text = await get_ai_response(user_id, final_prompt)
        if not response_text or not response_text.strip():
            raise AIServiceError("ИИ вернул пустой ответ")

        should_start_test, response_without_test_directive = _extract_test_start_directive(response_text)
        await save_ai_message(user_id, response_without_test_directive)

        if should_start_test:
            clean_text = response_without_test_directive
            if clean_text:
                await client.edit_message(thinking_message_id, text=markdown_to_html(clean_text))
            elif thinking_message_id:
                await client.edit_message(thinking_message_id, text="Запускаю тест.")
            from .tests import start_test

            await start_test(client, chat_id, user_id, states)
            return

        clean_response_text, response_buttons = extract_response_buttons(response_without_test_directive)

        # Check for image generation directive GEN_IMG: [...] or [IMG: ...]
        img_match = re.search(r"GEN_IMG:\s*\[(.*?)\]|\[IMG:\s*(.*?)\]", clean_response_text, re.DOTALL)
        if img_match:
            img_prompt = (img_match.group(1) or img_match.group(2) or "").strip()
            clean_text = re.sub(r"GEN_IMG:\s*\[.*?\]|\[IMG:\s*.*?\]", "", clean_response_text, flags=re.DOTALL).strip()
            buttons_sent = False
            if thinking_message_id and clean_text:
                await client.edit_message(
                    thinking_message_id,
                    text=markdown_to_html(clean_text),
                    attachments=_response_buttons_keyboard(response_buttons),
                )
                thinking_message_id = None
                buttons_sent = bool(response_buttons)
            elif clean_text:
                await client.send_message(
                    chat_id=chat_id,
                    text=markdown_to_html(clean_text),
                    attachments=_response_buttons_keyboard(response_buttons),
                )
                buttons_sent = bool(response_buttons)
            try:
                img_bytes = await generate_image(img_prompt)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                try:
                    result = await client.upload_file("image", tmp_path)
                    token = result.get("token") or result.get("fileId")
                    if token:
                        await client.send_media_attachment(chat_id=chat_id, media_type="image", token=token)
                    else:
                        await client.send_message(chat_id=chat_id, text="⚠️ Изображение сгенерировано, но не удалось отправить.")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception as img_exc:
                log.exception("Image generation failed user_id=%s: %s", user_id, img_exc)
                await client.send_message(chat_id=chat_id, text=f"⚠️ Не удалось создать изображение: {img_exc}")
            if response_buttons and not buttons_sent:
                await client.send_message(
                    chat_id=chat_id,
                    text="Выберите действие:",
                    attachments=_response_buttons_keyboard(response_buttons),
                )
            return

        if not clean_response_text and response_buttons:
            clean_response_text = "Выберите действие:"
        html_text = markdown_to_html(clean_response_text)
        chunks = split_text(html_text)
        await _send_ai_text(client, chat_id, thinking_message_id, chunks, response_buttons)
        log.info("AI dialogue completed user_id=%s chat_id=%s chunks=%s", user_id, chat_id, len(chunks))
    except AIServiceError as exc:
        log.exception("AIServiceError: %s", exc)
        if thinking_message_id:
            await client.edit_message(thinking_message_id, text="Сервис ИИ временно недоступен. Попробуйте позже.")
        else:
            await client.send_message(chat_id=chat_id, text="Сервис ИИ временно недоступен. Попробуйте позже.")
    except Exception:
        log.exception("Unexpected bot dialogue failure user_id=%s chat_id=%s", user_id, chat_id)
        if thinking_message_id:
            await client.edit_message(thinking_message_id, text="Произошла внутренняя ошибка. Попробуйте позже.")
        else:
            await client.send_message(chat_id=chat_id, text="Произошла внутренняя ошибка. Попробуйте позже.")


def _extract_ai_directive_payload(text: str, directive: str) -> tuple[str | None, str]:
    pattern = rf"{directive}:\s*(.+?)(?=\n\s*\n|\n(?:[#>*-]|\d+\.)\s|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None, text.strip()

    payload = match.group(1).strip()
    clean_text = (text[:match.start()] + text[match.end():]).strip()
    return payload, clean_text


async def run_ai_dialogue_with_image(client: MaxApiClient, chat_id: int, user_id: int, image_bytes: bytes, caption: str) -> None:
    """Run AI vision analysis on the provided image bytes."""
    from ..ai import analyze_image

    log.info("AI vision requested user_id=%s chat_id=%s", user_id, chat_id)
    prompt = caption or "Опиши это изображение подробно."
    await save_user_message(user_id, f"[Изображение] {prompt}")
    thinking = await client.send_message(chat_id=chat_id, text="🤖 Анализирую изображение...")
    thinking_message_id = ((thinking.get("message") or {}).get("mid") if isinstance(thinking, dict) else None)

    try:
        response_text = await analyze_image(user_id, image_bytes, prompt)
        if not response_text or not response_text.strip():
            raise AIServiceError("ИИ вернул пустой ответ при анализе изображения")

        edit_prompt, clean_text = _extract_ai_directive_payload(response_text, "EDIT_IMG")
        gen_prompt, clean_text = _extract_ai_directive_payload(clean_text, "GEN_IMG")

        await save_ai_message(user_id, clean_text)
        html_text = markdown_to_html(clean_text)
        chunks = split_text(html_text)

        if chunks:
            await _send_ai_text(client, chat_id, thinking_message_id, chunks)
            thinking_message_id = None

        if edit_prompt:
            edit_status = await client.send_message(chat_id=chat_id, text="🎨 Редактирую ваше фото...")
            edit_status_id = ((edit_status.get("message") or {}).get("mid") if isinstance(edit_status, dict) else None)
            try:
                edited_data = await edit_image(edit_prompt, image_bytes)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(edited_data)
                    tmp_path = tmp.name
                try:
                    result = await client.upload_file("image", tmp_path)
                    token = result.get("token") or result.get("fileId")
                    if token:
                        await client.send_media_attachment(chat_id=chat_id, media_type="image", token=token)
                    else:
                        await client.send_message(chat_id=chat_id, text="⚠️ Фото отредактировано, но не удалось отправить.")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception as img_exc:
                log.exception("Image edit failed user_id=%s: %s", user_id, img_exc)
                await client.send_message(chat_id=chat_id, text="⚠️ Не удалось изменить изображение.")
            finally:
                if edit_status_id:
                    try:
                        await client.delete_message(edit_status_id)
                    except Exception:
                        pass

        elif gen_prompt:
            gen_status = await client.send_message(chat_id=chat_id, text="🖼 Генерирую новое изображение...")
            gen_status_id = ((gen_status.get("message") or {}).get("mid") if isinstance(gen_status, dict) else None)
            try:
                new_img = await generate_image(gen_prompt)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(new_img)
                    tmp_path = tmp.name
                try:
                    result = await client.upload_file("image", tmp_path)
                    token = result.get("token") or result.get("fileId")
                    if token:
                        await client.send_media_attachment(chat_id=chat_id, media_type="image", token=token)
                    else:
                        await client.send_message(chat_id=chat_id, text="⚠️ Изображение сгенерировано, но не удалось отправить.")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception as img_exc:
                log.exception("Image generation failed user_id=%s: %s", user_id, img_exc)
                await client.send_message(chat_id=chat_id, text="⚠️ Не удалось сгенерировать изображение.")
            finally:
                if gen_status_id:
                    try:
                        await client.delete_message(gen_status_id)
                    except Exception:
                        pass

    except AIServiceError as exc:
        log.exception("Vision AIServiceError: %s", exc)
        if thinking_message_id:
            await client.edit_message(thinking_message_id, text="Сервис анализа изображений временно недоступен.")
        else:
            await client.send_message(chat_id=chat_id, text="Сервис анализа изображений временно недоступен.")
    except Exception:
        log.exception("Vision unexpected failure user_id=%s", user_id)
        if thinking_message_id:
            await client.edit_message(thinking_message_id, text="Произошла ошибка при анализе изображения.")
        else:
            await client.send_message(chat_id=chat_id, text="Произошла ошибка при анализе изображения.")


async def run_ai_dialogue_with_voice(client: MaxApiClient, chat_id: int, user_id: int, audio_bytes: bytes, filename: str = "audio.ogg") -> None:
    """Transcribe audio and run AI dialogue with the transcription."""
    from ..ai import transcribe_audio

    log.info("Voice message received user_id=%s chat_id=%s", user_id, chat_id)
    thinking = await client.send_message(chat_id=chat_id, text="🎙 Распознаю голосовое сообщение...")
    thinking_message_id = ((thinking.get("message") or {}).get("mid") if isinstance(thinking, dict) else None)

    try:
        transcription = await transcribe_audio(audio_bytes, filename)
        log.info("Voice transcribed user_id=%s len=%s", user_id, len(transcription))
        if thinking_message_id:
            await client.edit_message(thinking_message_id, text=f"🎙 <i>{html.escape(transcription)}</i>")
            thinking_message_id = None
        else:
            await client.send_message(chat_id=chat_id, text=f"🎙 <i>{html.escape(transcription)}</i>")
    except AIServiceError as exc:
        log.exception("Transcription failed: %s", exc)
        msg = f"Не удалось распознать аудио: {exc}"
        if thinking_message_id:
            await client.edit_message(thinking_message_id, text=msg)
        else:
            await client.send_message(chat_id=chat_id, text=msg)
        return
    except Exception:
        log.exception("Unexpected transcription failure user_id=%s", user_id)
        msg = "Произошла ошибка при распознавании аудио."
        if thinking_message_id:
            await client.edit_message(thinking_message_id, text=msg)
        else:
            await client.send_message(chat_id=chat_id, text=msg)
        return

    async with async_session_maker() as session:
        user = await session.get(User, user_id, options=[selectinload(User.subscription)])
    if user and await ensure_access_before_chat(client, chat_id, user):
        await run_ai_dialogue(client, chat_id, user_id, transcription)


async def maybe_require_disclaimer(client: MaxApiClient, states: StateStore, chat_id: int, user: User, pending_prompt: str) -> bool:
    if user.accepted_disclaimer:
        return False
    disclaimer = await get_content("disclaimer")
    if disclaimer and disclaimer.is_visible:
        await states.set(user.id, chat_id, "awaiting_disclaimer_acceptance", {"initial_prompt": pending_prompt})
        await client.send_message(
            chat_id=chat_id,
            text=disclaimer.text_content or "Примите дисклеймер для продолжения.",
            attachments=(await get_content_attachments("disclaimer") or []) + disclaimer_keyboard(),
        )
        return True
    async with async_session_maker() as session:
        await session.execute(update(User).where(User.id == user.id).values(accepted_disclaimer=True))
        await session.commit()
    return False
