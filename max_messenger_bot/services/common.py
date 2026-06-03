from __future__ import annotations

import html
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import selectinload

from ..ai import AIServiceError, generate_image, get_ai_response
from ..api import MaxApiClient
from ..formatting import markdown_to_html, split_text
from ..keyboards import build_main_menu, disclaimer_keyboard, inline_keyboard, main_menu_row
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


log = get_bot_logger("common")


MSK = timezone(timedelta(hours=3))


async def is_admin(user_id: int) -> bool:
    from ..settings import apply_legacy_env_defaults
    from config import OWNER_IDS  # type: ignore

    apply_legacy_env_defaults()
    if user_id in OWNER_IDS:
        return True
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        return bool(user and user.is_admin)


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
    return [{"type": row.media_type, "payload": {"token": row.token}} for row in media_rows]


async def send_main_menu(client: MaxApiClient, chat_id: int, text: str = "Главное меню:") -> None:
    await client.send_message(chat_id=chat_id, text=text, attachments=await build_main_menu())


async def show_help(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    if not await is_admin(user_id):
        text = (
            "👋 Я персональный ИИ-помощник.\n\n"
            "Пишите вопрос в чат или пользуйтесь кнопками меню. "
            "В MAX навигация перенесена в inline-кнопки под сообщениями."
        )
    else:
        text = (
            "<b>Режим администратора</b>\n\n"
            "Доступны основные экраны: статистика, подписки, темы, рефералка и тест. "
            "Для MAX я вынес логику в отдельный модуль, поэтому административные сценарии будут расширяться поэтапно без возврата к монолиту."
        )
    await client.send_message(chat_id=chat_id, text=text)


async def show_start_screen(client: MaxApiClient, chat_id: int, user_id: int, start_payload: str | None = None) -> None:
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

    if start_payload:
        if start_payload == "sub":
            from .subscriptions import show_subscription_info

            await show_subscription_info(client, chat_id, user_id)
            return
        if start_payload == "test":
            from .tests import start_test

            await start_test(client, chat_id, user_id)
            return
        if start_payload.startswith("topic_"):
            from .topics import select_topic

            await select_topic(client, chat_id, user_id, int(start_payload.split("_", 1)[1]))
            return
        content = await get_content(start_payload)
        if content and content.is_visible:
            await client.send_message(
                chat_id=chat_id,
                text=content.text_content or "Раздел пока пуст.",
                attachments=await get_content_attachments(start_payload) or None,
            )
            await send_main_menu(client, chat_id)
            return

    start_content = await get_content("start_message")
    if start_content and start_content.text_content:
        await client.send_message(
            chat_id=chat_id,
            text=start_content.text_content,
            attachments=await get_content_attachments("start_message") or None,
        )
    else:
        await client.send_message(chat_id=chat_id, text="Здравствуйте. Бот в MAX готов к работе.")
    if welcome_bonus_text:
        await client.send_message(chat_id=chat_id, text=welcome_bonus_text)
    await send_main_menu(client, chat_id)


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


async def run_ai_dialogue(client: MaxApiClient, chat_id: int, user_id: int, prompt_text: str) -> None:
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
        await save_ai_message(user_id, response_text)

        # Check for image generation directive GEN_IMG: [...] or [IMG: ...]
        img_match = re.search(r"GEN_IMG:\s*\[(.*?)\]|\[IMG:\s*(.*?)\]", response_text, re.DOTALL)
        if img_match:
            img_prompt = (img_match.group(1) or img_match.group(2) or "").strip()
            clean_text = re.sub(r"GEN_IMG:\s*\[.*?\]|\[IMG:\s*.*?\]", "", response_text, flags=re.DOTALL).strip()
            if thinking_message_id and clean_text:
                await client.edit_message(thinking_message_id, text=markdown_to_html(clean_text))
                thinking_message_id = None
            elif clean_text:
                await client.send_message(chat_id=chat_id, text=markdown_to_html(clean_text))
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
            return

        html_text = markdown_to_html(response_text)
        chunks = split_text(html_text)
        main_menu_kb = inline_keyboard([main_menu_row()])
        if thinking_message_id and chunks:
            if len(chunks) == 1:
                await client.edit_message(thinking_message_id, text=chunks[0], attachments=main_menu_kb)
            else:
                await client.edit_message(thinking_message_id, text=chunks[0])
                for chunk in chunks[1:-1]:
                    await client.send_message(chat_id=chat_id, text=chunk)
                await client.send_message(chat_id=chat_id, text=chunks[-1], attachments=main_menu_kb)
        else:
            for chunk in chunks[:-1]:
                await client.send_message(chat_id=chat_id, text=chunk)
            if chunks:
                await client.send_message(chat_id=chat_id, text=chunks[-1], attachments=main_menu_kb)
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


async def run_ai_dialogue_with_image(client: MaxApiClient, chat_id: int, user_id: int, image_bytes: bytes, caption: str) -> None:
    """Run AI vision analysis on the provided image bytes."""
    from ..ai import analyze_image

    log.info("AI vision requested user_id=%s chat_id=%s", user_id, chat_id)
    prompt = caption or "Опиши это изображение подробно."
    await save_user_message(user_id, f"[Изображение] {prompt}")
    thinking = await client.send_message(chat_id=chat_id, text="🤖 Анализирую изображение...")
    thinking_message_id = ((thinking.get("message") or {}).get("mid") if isinstance(thinking, dict) else None)

    try:
        response_text = await analyze_image(image_bytes, prompt)
        await save_ai_message(user_id, response_text)
        html_text = markdown_to_html(response_text)
        chunks = split_text(html_text)
        if thinking_message_id and chunks:
            await client.edit_message(thinking_message_id, text=chunks[0])
            for chunk in chunks[1:]:
                await client.send_message(chat_id=chat_id, text=chunk)
        else:
            for chunk in chunks:
                await client.send_message(chat_id=chat_id, text=chunk)
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
