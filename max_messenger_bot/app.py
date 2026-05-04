from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from .api import MaxApiClient
from .legacy import Content, SubscriptionConfig, Topic, User, async_session_maker, init_db
from .logging_utils import configure_logging, get_bot_logger, get_max_logger
from .models import IncomingCallback, IncomingMessage, parse_callback, parse_message
from .services import admin as admin_service
from .services import admin_ai as admin_ai_service
from .services import admin_admins as admin_admins_service
from .services import admin_billing as admin_billing_service
from .services import admin_buttons as admin_buttons_service
from .services import admin_clients as admin_clients_service
from .services import admin_content as admin_content_service
from .services import admin_export as admin_export_service
from .services import admin_kb as admin_kb_service
from .services import admin_mailing as admin_mailing_service
from .services import admin_payments as admin_payments_service
from .services import admin_test_content as admin_test_content_service
from .services import admin_tests as admin_tests_service
from .services import admin_topics as admin_topics_service
from .services import admin_referral as admin_referral_service
from .services import admin_collections as admin_collections_service
from .services import admin_topic_media as admin_topic_media_service
from .services import common, settings as settings_service, subscriptions as subscriptions_service, tests as tests_service, topics as topics_service
from .settings import get_settings, validate_webhook_runtime_settings
from .storage import StateStore, init_storage


configure_logging()
log = get_bot_logger("app")
max_log = get_max_logger("transport")

TEXT_FILE_INPUT_STATES = {
    "admin_edit_content_text",
    "admin_edit_topic_prompt",
    "admin_edit_topic_intro",
    "admin_test_set_prompt",
    "admin_kb_create_content",
    "admin_kb_edit_content",
    "admin_ai_set_system_prompt",
    "admin_ai_set_global_prompt_appendix",
    "admin_mailing_text",
    "admin_ref_tpl_add",
    "admin_ref_tpl_edit",
}


def _log_background_task_result(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        max_log.info("MAX background task cancelled")
    except Exception:
        max_log.exception("MAX background task failed")


def _spawn_update_task(app: web.Application, update: dict[str, Any]) -> None:
    task = asyncio.create_task(app["bot_app"].handle_update(update))
    tasks: set[asyncio.Task[None]] = app["background_tasks"]
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    task.add_done_callback(_log_background_task_result)


class MaxBotApplication:
    def __init__(self, client: MaxApiClient) -> None:
        self.client = client
        self.states = StateStore()

    async def handle_update(self, update: dict[str, Any]) -> None:
        update_type = update.get("update_type") or update.get("type") or ""
        update_id = update.get("update_id") or update.get("id") or update.get("event_id")
        try:
            max_log.info("MAX update received update_type=%s update_id=%s", update_type, update_id)
            if update_type in {"message_created", "bot_started"}:
                message = parse_message(update)
                if message:
                    await self.handle_message(message, force_start=(update_type == "bot_started"))
                return
            if update_type == "message_callback":
                callback = parse_callback(update)
                if callback:
                    await self.handle_callback(callback)
                else:
                    log.warning("Received message_callback but parse_callback returned None update=%s", update)
                return
            max_log.info("Skipping unsupported MAX update type=%s update_id=%s", update_type, update_id)
        except Exception:
            log.exception("Update processing failed update_type=%s update_id=%s payload=%s", update_type, update_id, update)
            await self._notify_update_failure(update_type, update)

    async def _notify_update_failure(self, update_type: str, update: dict[str, Any]) -> None:
        try:
            if update_type in {"message_created", "bot_started"}:
                message = parse_message(update)
                if message:
                    await self.client.send_message(
                        chat_id=message.chat_id,
                        text="Произошла внутренняя ошибка. Попробуйте позже.",
                    )
                    return
            if update_type == "message_callback":
                callback = parse_callback(update)
                if callback:
                    await self.client.answer_callback(
                        callback.callback_id,
                        notification="Произошла ошибка. Попробуйте позже.",
                    )
                    return
        except Exception:
            log.exception("Failed to notify user about update failure update_type=%s", update_type)

    async def handle_message(self, message: IncomingMessage, force_start: bool = False) -> None:
        log.info(
            "Incoming message user_id=%s chat_id=%s force_start=%s text=%s",
            message.sender.user_id,
            message.chat_id,
            force_start,
            (message.text or "")[:300],
        )
        await common.ensure_user(message.sender.user_id, message.sender.username, message.sender.full_name)
        if force_start:
            await common.show_start_screen(self.client, message.chat_id, message.sender.user_id, message.start_payload)
            return

        text = (message.text or "").strip()
        state = await self.states.get(message.sender.user_id)
        if state and not text and message.media_type == "file" and message.media_token and state.state in TEXT_FILE_INPUT_STATES:
            text = await self._read_text_attachment(message)
            if not text:
                return
        # Commands always escape any pending input state so users can't get stuck
        if state and text.startswith("/"):
            await self.states.clear(message.sender.user_id)
            state = None
        if state:
            if state.state == "awaiting_tg_id":
                from .services.link_tg import process_tg_link

                await process_tg_link(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "awaiting_name":
                await settings_service.process_new_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                state_data = state.data
                await settings_service.start_change_gender(
                    self.client,
                    self.states,
                    message.chat_id,
                    message.sender.user_id,
                    is_settings=False,
                    initial_prompt=state_data.get("initial_prompt"),
                )
                return
            if state.state == "awaiting_new_name":
                await settings_service.process_new_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "awaiting_age":
                snapshot = await settings_service.save_age(self.client, self.states, message.chat_id, message.sender.user_id, text)
                if snapshot and not snapshot.get("is_settings"):
                    await tests_service.start_test(self.client, message.chat_id, message.sender.user_id)
                return
            if state.state == "awaiting_promo_code":
                await subscriptions_service.apply_promo_code(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_edit_content_text":
                await admin_content_service.save_text_edit(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_create_topic_name":
                await admin_topics_service.create_topic(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_edit_topic_name":
                await admin_topics_service.save_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_edit_topic_prompt":
                await admin_topics_service.save_prompt(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_edit_topic_intro":
                await admin_topics_service.save_intro(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_test_set_admin_username":
                await admin_tests_service.save_admin_username(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_test_set_marathon_url":
                await admin_tests_service.save_marathon_url(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_test_add_secret_question":
                await admin_tests_service.save_secret_question(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_test_set_prompt":
                await admin_tests_service.save_prompt(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_test_question_create_text":
                await admin_test_content_service.save_new_question_text(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_test_question_edit_text":
                await admin_test_content_service.save_question_text(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_test_question_edit_sort":
                await admin_test_content_service.save_question_sort(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_case_study_create_text":
                await admin_test_content_service.save_new_case_study(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_case_study_edit_text":
                await admin_test_content_service.save_case_study_text(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_mailing_text":
                await admin_mailing_service.save_input(self.client, self.states, message.chat_id, message.sender.user_id, message)
                return
            if state.state == "admin_button_edit_title":
                await admin_buttons_service.save_title(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_button_create_title":
                await admin_buttons_service.create_button(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_kb_create_filename":
                await admin_kb_service.save_new_filename(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_kb_create_content":
                await admin_kb_service.save_new_content(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_kb_edit_filename":
                await admin_kb_service.save_filename(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_kb_edit_content":
                await admin_kb_service.save_content(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_add_admin_id":
                await admin_admins_service.add_admin(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_ai_set_key":
                await admin_ai_service.save_key(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_ai_set_context_first":
                await admin_ai_service.save_int(self.client, self.states, message.chat_id, message.sender.user_id, text, minimum=0)
                return
            if state.state == "admin_ai_set_context_recent":
                await admin_ai_service.save_int(self.client, self.states, message.chat_id, message.sender.user_id, text, minimum=0)
                return
            if state.state == "admin_ai_set_audio_limit":
                await admin_ai_service.save_int(self.client, self.states, message.chat_id, message.sender.user_id, text, minimum=0)
                return
            if state.state == "admin_ai_set_temperature":
                await admin_ai_service.save_temperature(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_ai_set_system_prompt":
                await admin_ai_service.save_system_prompt(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_ai_set_global_prompt_appendix":
                await admin_ai_service.save_global_prompt_appendix(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_payment_set_bonus":
                await admin_payments_service.save_bonus(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_payment_set_key":
                await admin_payments_service.save_key(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_create_name":
                await admin_billing_service.save_new_plan_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_create_description":
                await admin_billing_service.save_new_plan_description(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_create_price":
                await admin_billing_service.save_new_plan_price(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_create_duration_value":
                await admin_billing_service.save_new_plan_duration_value(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_edit_name":
                await admin_billing_service.save_plan_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_edit_description":
                await admin_billing_service.save_plan_description(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_edit_price":
                await admin_billing_service.save_plan_price(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_edit_duration_value":
                await admin_billing_service.save_plan_duration_value(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_plan_edit_cooldown":
                await admin_billing_service.save_plan_trial_cooldown(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_create_code":
                await admin_billing_service.save_new_promo_code(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_create_discount":
                await admin_billing_service.save_new_promo_discount(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_create_days":
                await admin_billing_service.save_new_promo_days(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_create_uses":
                await admin_billing_service.save_new_promo_uses(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_edit_code":
                await admin_billing_service.save_promo_code(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_edit_discount":
                await admin_billing_service.save_promo_discount(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_edit_days":
                await admin_billing_service.save_promo_days(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_promo_edit_uses":
                await admin_billing_service.save_promo_uses(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "secret_test_answering":
                await tests_service.save_secret_answers(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "awaiting_disclaimer_acceptance":
                await self.client.send_message(message.chat_id, text="Подтвердите дисклеймер кнопкой ниже.")
                return
            if state.state == "admin_client_search":
                if text:
                    await admin_clients_service.search_clients(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_ai_set_kie_threshold":
                await admin_ai_service.save_kie_threshold(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_referral_set_bonus_referrer":
                await admin_referral_service.save_bonus_referrer(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_referral_set_bonus_referral":
                await admin_referral_service.save_bonus_referral(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_referral_set_pay_days":
                await admin_referral_service.save_pay_days(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_referral_set_btn_name":
                await admin_referral_service.save_btn_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_referral_set_sub_btn_name":
                await admin_referral_service.save_sub_btn_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_ref_tpl_add":
                await admin_referral_service.save_new_template(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_ref_tpl_edit":
                await admin_referral_service.save_template_edit(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            # ── Collections states ──────────────────────────────────────────
            if state.state == "admin_coll_creating":
                await admin_collections_service.save_create(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_coll_renaming":
                await admin_collections_service.save_rename(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            # ── Topic media states ─────────────────────────────────────────
            if state.state == "admin_media_edit_name":
                await admin_topic_media_service.save_edit_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_media_edit_category":
                await admin_topic_media_service.save_edit_category(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_media_edit_desc":
                await admin_topic_media_service.save_edit_description(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_media_edit_file":
                token = message.media_token if not text else None
                mtype = message.media_type if not text else None
                await admin_topic_media_service.save_edit_file(self.client, self.states, message.chat_id, message.sender.user_id, token=token, media_type=mtype, text=text)
                return
            if state.state == "admin_media_add_file":
                token = message.media_token if not text else None
                mtype = message.media_type if not text else None
                await admin_topic_media_service.receive_add_file(self.client, self.states, message.chat_id, message.sender.user_id, text=text, media_token=token, media_type=mtype)
                return
            if state.state == "admin_media_add_type":
                await admin_topic_media_service.resolve_add_type(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_media_add_name":
                await admin_topic_media_service.save_add_name(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_media_add_category":
                await admin_topic_media_service.save_add_category(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_media_add_desc":
                await admin_topic_media_service.save_add_description(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            # ── Export date states ─────────────────────────────────────────
            if state.state == "admin_export_date_from":
                await admin_export_service.save_date_from(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return
            if state.state == "admin_export_date_to":
                await admin_export_service.save_date_to(self.client, self.states, message.chat_id, message.sender.user_id, text)
                return

        if not text:
            # Handle media attachments for admin upload states
            if message.media_token and state:
                if state.state in {"admin_media_add_file", "admin_media_edit_file"}:
                    token = message.media_token
                    mtype = "photo" if message.media_type == "image" else "audio"
                    if state.state == "admin_media_add_file":
                        await admin_topic_media_service.receive_add_file(self.client, self.states, message.chat_id, message.sender.user_id, media_token=token, media_type=mtype)
                    else:
                        await admin_topic_media_service.save_edit_file(self.client, self.states, message.chat_id, message.sender.user_id, token=token, media_type=mtype)
                    return
            # Handle media attachments
            if message.media_type in {"audio", "video"} and message.media_token:
                await self._handle_voice(message)
                return
            if message.media_type == "image" and message.media_token:
                await self._handle_image(message, caption="")
                return
            if message.media_type == "file" and message.media_token:
                await self.client.send_message(chat_id=message.chat_id, text="📎 Файлы пока не поддерживаются в этом боте.")
                return
            await self.client.send_message(chat_id=message.chat_id, text="Пожалуйста, отправьте текстовое сообщение.")
            return

        if text.startswith("/start"):
            payload = text.split(" ", 1)[1].strip() if " " in text else None
            await common.show_start_screen(self.client, message.chat_id, message.sender.user_id, payload)
            return
        if text == "/help":
            await common.show_help(self.client, message.chat_id, message.sender.user_id)
            return
        if text == "/admin":
            if await common.is_admin(message.sender.user_id):
                await admin_service.show_admin_panel(self.client, message.chat_id)
            return
        if text == "/promo":
            await subscriptions_service.start_promo_entry(self.client, self.states, message.chat_id, message.sender.user_id)
            return
        if text == "/ref":
            await subscriptions_service.show_referral_info(self.client, message.chat_id, message.sender.user_id)
            return
        if text == "/test" or text == "📝 Пройти тест":
            await tests_service.start_test(self.client, message.chat_id, message.sender.user_id)
            return
        if text == "⭐️ Подписка":
            await subscriptions_service.show_subscription_info(self.client, message.chat_id, message.sender.user_id)
            return
        if text == "⚙️ Настройки":
            await settings_service.show_settings(self.client, message.chat_id, message.sender.user_id)
            return
        if text == "🗑️ Новый диалог":
            await common.reset_dialogue(self.client, message.chat_id, message.sender.user_id)
            return

        async with async_session_maker() as session:
            user = await session.get(User, message.sender.user_id)
            sub_config = await session.get(SubscriptionConfig, 1)
            topics = (await session.execute(select(Topic).where(Topic.is_active == True))).scalars().all()

        if sub_config:
            if text == sub_config.topics_btn_name:
                await topics_service.show_topics(self.client, message.chat_id, message.sender.user_id)
                return
            if sub_config.referral_enabled and text == sub_config.referral_btn_name:
                await subscriptions_service.show_referral_info(self.client, message.chat_id, message.sender.user_id)
                return

        topic_by_name = next((topic for topic in topics if topic.name == text and topic.show_in_main_menu), None)
        if topic_by_name:
            await topics_service.select_topic(self.client, message.chat_id, message.sender.user_id, topic_by_name.id)
            return

        async with async_session_maker() as session:
            content = await session.scalar(select(Content).where(Content.button_title == text, Content.is_visible == True).limit(1))
            user = await session.get(User, message.sender.user_id, options=[selectinload(User.subscription)])
        if content:
            from .keyboards import callback_button, inline_keyboard
            content_attachments = await common.get_content_attachments(content.key)
            nav = inline_keyboard([[callback_button("◀️ Главное меню", "main_menu")]])
            await self.client.send_message(
                chat_id=message.chat_id,
                text=content.text_content or "Раздел пока пуст.",
                attachments=(content_attachments or []) + nav,
            )
            return

        if not user:
            return
        if not user.name:
            await common.begin_onboarding(self.client, self.states, message.chat_id, message.sender.user_id, text)
            return
        if await common.maybe_require_disclaimer(self.client, self.states, message.chat_id, user, text):
            return
        if not await common.ensure_access_before_chat(self.client, message.chat_id, user):
            return
        # If user sent image with caption, handle as vision
        if message.media_type == "image" and message.media_token:
            await self._handle_image(message, caption=text)
            return
        await common.run_ai_dialogue(self.client, message.chat_id, message.sender.user_id, text)

    async def _read_text_attachment(self, message: IncomingMessage) -> str | None:
        try:
            payload = await self.client.download_attachment(message.media_token, message.media_url)
        except Exception:
            log.exception("Failed to download text attachment token=%s", message.media_token)
            await self.client.send_message(chat_id=message.chat_id, text="Не удалось скачать файл. Отправьте текст сообщением или повторите файл.")
            return None
        if len(payload) > 512 * 1024:
            await self.client.send_message(chat_id=message.chat_id, text="Файл слишком большой. Максимум для текстового поля — 512 КБ.")
            return None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                text = payload.decode(encoding).strip()
                break
            except UnicodeDecodeError:
                text = ""
        if not text:
            await self.client.send_message(chat_id=message.chat_id, text="Не удалось прочитать файл как текст. Поддерживаются обычные .txt/.md в UTF-8 или Windows-1251.")
            return None
        return text

    async def _handle_voice(self, message: IncomingMessage) -> None:
        """Download and transcribe a voice/audio message."""
        try:
            audio_bytes = await self.client.download_attachment(message.media_token, message.media_url)
        except Exception as exc:
            log.exception("Failed to download audio token=%s: %s", message.media_token, exc)
            await self.client.send_message(chat_id=message.chat_id, text="Не удалось загрузить аудиофайл. Попробуйте ещё раз.")
            return

        async with async_session_maker() as session:
            user = await session.get(User, message.sender.user_id, options=[selectinload(User.subscription)])
        if not user or not user.name:
            await self.client.send_message(chat_id=message.chat_id, text="Пожалуйста, пройдите первоначальную настройку.")
            return
        if await common.maybe_require_disclaimer(self.client, self.states, message.chat_id, user, ""):
            return
        if not await common.ensure_access_before_chat(self.client, message.chat_id, user):
            return
        await common.run_ai_dialogue_with_voice(self.client, message.chat_id, message.sender.user_id, audio_bytes)

    async def _handle_image(self, message: IncomingMessage, caption: str) -> None:
        """Download and analyze an image message."""
        try:
            image_bytes = await self.client.download_attachment(message.media_token, message.media_url)
        except Exception as exc:
            log.exception("Failed to download image token=%s: %s", message.media_token, exc)
            await self.client.send_message(chat_id=message.chat_id, text="Не удалось загрузить изображение. Попробуйте ещё раз.")
            return

        async with async_session_maker() as session:
            user = await session.get(User, message.sender.user_id, options=[selectinload(User.subscription)])
        if not user or not user.name:
            await self.client.send_message(chat_id=message.chat_id, text="Пожалуйста, пройдите первоначальную настройку.")
            return
        if await common.maybe_require_disclaimer(self.client, self.states, message.chat_id, user, caption or ""):
            return
        if not await common.ensure_access_before_chat(self.client, message.chat_id, user):
            return
        await common.run_ai_dialogue_with_image(self.client, message.chat_id, message.sender.user_id, image_bytes, caption)

    async def handle_callback(self, callback: IncomingCallback) -> None:
        log.info("Incoming callback user_id=%s chat_id=%s payload=%s", callback.sender.user_id, callback.chat_id, callback.payload)
        data = callback.payload
        user_id = callback.sender.user_id
        chat_id = callback.chat_id

        if data == "main_menu":
            await common.send_main_menu(self.client, chat_id)
            return
        if data == "topic_start_dialogue":
            await self.client.send_message(chat_id=chat_id, text="✍️ Напишите ваш первый вопрос, и я отвечу.")
            return
        if data == "noop":
            await self.client.answer_callback(callback.callback_id)
            return
        if data == "admin_panel":
            await admin_service.show_admin_panel(self.client, chat_id)
            return
        if data == "disclaimer_accepted":
            snapshot = await self.states.get(user_id)
            async with async_session_maker() as session:
                await session.execute(update(User).where(User.id == user_id).values(accepted_disclaimer=True))
                await session.commit()
                user = await session.get(User, user_id, options=[selectinload(User.subscription)])
            pending_prompt = snapshot.data.get("initial_prompt") if snapshot else None
            await self.states.clear(user_id)
            await self.client.answer_callback(callback.callback_id, notification="Дисклеймер принят")
            if pending_prompt and user and await common.ensure_access_before_chat(self.client, chat_id, user):
                await common.run_ai_dialogue(self.client, chat_id, user_id, pending_prompt)
            return
        if data.startswith("gender_"):
            state_data = await settings_service.save_gender(self.client, self.states, chat_id, user_id, data.split("_", 1)[1])
            await self.client.answer_callback(callback.callback_id, notification="Пол сохранён")
            if state_data.get("is_onboarding"):
                await settings_service.start_change_age(self.client, self.states, chat_id, user_id, is_settings=False)
            return
        if data == "settings_change_name":
            await settings_service.start_change_name(self.client, self.states, chat_id, user_id)
            return
        if data == "settings_change_gender":
            await settings_service.start_change_gender(self.client, self.states, chat_id, user_id)
            return
        if data == "settings_change_age":
            await settings_service.start_change_age(self.client, self.states, chat_id, user_id)
            return
        if data == "settings_toggle_length":
            await settings_service.toggle_response_length(self.client, chat_id, user_id)
            return
        if data.startswith("select_topic_"):
            await topics_service.select_topic(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
            return
        if data == "reset_topic":
            await topics_service.reset_topic(self.client, chat_id, user_id)
            return
        if data in {"show_subscription_info_from_chat", "back_to_sub_info", "sub_info"}:
            await subscriptions_service.show_subscription_info(self.client, chat_id, user_id)
            return
        if data == "sub_select_plan":
            await subscriptions_service.show_plans(self.client, chat_id, user_id)
            return
        if data.startswith("sub_pay_"):
            await subscriptions_service.choose_payment_provider(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
            return
        if data.startswith("pay_yookassa_"):
            await subscriptions_service.create_yookassa_link(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
            return
        if data.startswith("pay_robokassa_"):
            await subscriptions_service.create_robokassa_link(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
            return
        if data == "sub_enter_promo":
            await subscriptions_service.start_promo_entry(self.client, self.states, chat_id, user_id)
            return
        if data == "link_tg_start":
            from .services.link_tg import show_link_tg_prompt

            await self.states.set(user_id, chat_id, "awaiting_tg_id", {})
            await show_link_tg_prompt(self.client, chat_id, user_id)
            return
        if data == "sub_enable_renewal":
            await subscriptions_service.set_renewal(self.client, chat_id, user_id, True)
            return
        if data == "sub_disable_renewal":
            await subscriptions_service.set_renewal(self.client, chat_id, user_id, False)
            return
        if data == "sub_retry_now" or data == "sub_cancel_retry":
            await subscriptions_service.show_subscription_info(self.client, chat_id, user_id)
            return
        if data == "referral_sub_info":
            await subscriptions_service.show_referral_info(self.client, chat_id, user_id)
            return
        if data.startswith("test_ans_"):
            await tests_service.process_answer(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
            return
        if data == "test_confirm_case":
            await tests_service.show_results(self.client, chat_id, user_id)
            return
        if data == "start_secret_test":
            await tests_service.start_secret_test(self.client, self.states, chat_id, user_id)
            return
        if data == "continue_dialogue_after_test":
            await self.states.clear(user_id)
            await self.client.send_message(chat_id=chat_id, text="Я здесь. Можем обсудить результаты или любой другой вопрос.")
            return
        if await common.is_admin(user_id):
            if data == "admin_stats":
                await admin_service.show_stats(self.client, chat_id)
                return
            if data == "admin_ai_settings":
                await admin_ai_service.show_settings(self.client, chat_id)
                return
            if data.startswith("admin_ai_provider_"):
                await admin_ai_service.set_provider(self.client, chat_id, data.replace("admin_ai_provider_", "", 1))
                return
            if data == "admin_ai_keys":
                await admin_ai_service.show_keys(self.client, chat_id)
                return
            if data.startswith("admin_ai_key_"):
                await admin_ai_service.start_set_key(self.client, self.states, chat_id, user_id, data.replace("admin_ai_key_", "", 1))
                return
            if data.startswith("admin_ai_models_"):
                await admin_ai_service.show_models(self.client, chat_id, data.replace("admin_ai_models_", "", 1))
                return
            if data.startswith("admin_ai_set_model_"):
                payload = data.replace("admin_ai_set_model_", "", 1)
                provider, model_name = payload.split("_", 1)
                await admin_ai_service.set_model(self.client, chat_id, provider, model_name)
                return
            if data == "admin_ai_toggle_transcription":
                await admin_ai_service.toggle_transcription(self.client, chat_id)
                return
            if data == "admin_ai_toggle_vision":
                await admin_ai_service.toggle_vision(self.client, chat_id)
                return
            if data == "admin_ai_vision_models":
                await admin_ai_service.show_vision_models(self.client, chat_id)
                return
            if data.startswith("admin_ai_set_vision_model_"):
                await admin_ai_service.set_vision_model(self.client, chat_id, data.replace("admin_ai_set_vision_model_", "", 1))
                return
            if data == "admin_ai_set_context_first":
                await admin_ai_service.start_set_int(self.client, self.states, chat_id, user_id, "admin_ai_set_context_first", "context_limit_first", "Введите количество первых сообщений контекста.")
                return
            if data == "admin_ai_set_context_recent":
                await admin_ai_service.start_set_int(self.client, self.states, chat_id, user_id, "admin_ai_set_context_recent", "context_limit_recent", "Введите количество последних сообщений контекста.")
                return
            if data == "admin_ai_set_audio_limit":
                await admin_ai_service.start_set_int(self.client, self.states, chat_id, user_id, "admin_ai_set_audio_limit", "max_voice_duration_sec", "Введите лимит аудио в секундах.")
                return
            if data == "admin_ai_set_temperature":
                await admin_ai_service.start_set_temperature(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_ai_cycle_memory_scope":
                await admin_ai_service.cycle_memory_scope(self.client, chat_id)
                return
            if data == "admin_ai_system_prompt":
                await admin_ai_service.start_edit_system_prompt(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_ai_global_prompt_appendix":
                await admin_ai_service.start_edit_global_prompt_appendix(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_clients":
                await admin_clients_service.list_clients(self.client, chat_id, 0)
                return
            if data.startswith("admin_clients_page_"):
                await admin_clients_service.list_clients(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("view_client_"):
                await admin_clients_service.show_client_profile(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("client_history_"):
                _, _, user_part, page_part = data.split("_", 3)
                await admin_clients_service.show_client_history(self.client, chat_id, int(user_part), int(page_part))
                return
            if data == "admin_client_search":
                await admin_clients_service.start_search(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_export":
                await admin_export_service.show_export_menu(self.client, chat_id)
                return
            if data == "admin_export_users_csv":
                await admin_export_service.export_users_csv(self.client, chat_id)
                return
            if data == "admin_export_messages_csv":
                await admin_export_service.show_date_filter_menu(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_subscriptions":
                await admin_service.show_subscriptions_summary(self.client, chat_id)
                return
            if data == "admin_payment_settings":
                await admin_payments_service.show_settings(self.client, chat_id)
                return
            if data == "admin_plans":
                await admin_billing_service.list_plans(self.client, chat_id)
                return
            if data == "admin_create_plan":
                await admin_billing_service.start_create_plan(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_edit_plan_"):
                await admin_billing_service.show_plan_editor(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_edit_name_"):
                await admin_billing_service.start_edit_plan_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_plan_edit_name",
                    "Введите новое название тарифа.",
                )
                return
            if data.startswith("admin_plan_edit_description_"):
                await admin_billing_service.start_edit_plan_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_plan_edit_description",
                    "Введите новое описание тарифа.",
                )
                return
            if data.startswith("admin_plan_edit_price_"):
                await admin_billing_service.start_edit_plan_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_plan_edit_price",
                    "Введите новую цену тарифа.",
                )
                return
            if data.startswith("admin_plan_edit_duration_value_"):
                await admin_billing_service.start_edit_plan_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_plan_edit_duration_value",
                    "Введите новую длительность тарифа.",
                )
                return
            if data.startswith("admin_plan_duration_unit_menu_"):
                await admin_billing_service.show_duration_unit_menu(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_set_duration_unit_"):
                _, _, _, _, _, plan_id, duration_unit = data.split("_", 6)
                await admin_billing_service.save_plan_duration_unit(self.client, chat_id, int(plan_id), duration_unit)
                return
            if data.startswith("admin_plan_toggle_active_"):
                await admin_billing_service.toggle_plan_active(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_toggle_admin_"):
                await admin_billing_service.toggle_plan_admin_only(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_toggle_renewal_"):
                await admin_billing_service.toggle_plan_renewal(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_toggle_trial_"):
                await admin_billing_service.toggle_plan_trial(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_edit_cooldown_"):
                await admin_billing_service.start_edit_plan_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_plan_edit_cooldown",
                    "Введите кулдаун пробного тарифа в днях.",
                )
                return
            if data.startswith("admin_plan_upgrade_menu_"):
                await admin_billing_service.show_plan_upgrade_menu(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_set_upgrade_none_"):
                await admin_billing_service.set_plan_upgrade_target(self.client, chat_id, int(data.rsplit("_", 1)[1]), None)
                return
            if data.startswith("admin_plan_set_upgrade_"):
                _, _, _, _, plan_id, target_plan_id = data.split("_", 5)
                await admin_billing_service.set_plan_upgrade_target(self.client, chat_id, int(plan_id), int(target_plan_id))
                return
            if data.startswith("admin_plan_create_duration_unit_"):
                await admin_billing_service.save_new_plan_duration_unit(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    data.replace("admin_plan_create_duration_unit_", "", 1),
                )
                return
            if data.startswith("admin_plan_delete_"):
                await admin_billing_service.delete_plan(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_promocodes":
                await admin_billing_service.list_promocodes(self.client, chat_id)
                return
            if data == "admin_create_promo":
                await admin_billing_service.start_create_promo(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_edit_promo_"):
                await admin_billing_service.show_promo_editor(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_promo_edit_code_"):
                await admin_billing_service.start_edit_promo_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_promo_edit_code",
                    "Введите новый код промокода.",
                )
                return
            if data.startswith("admin_promo_edit_discount_"):
                await admin_billing_service.start_edit_promo_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_promo_edit_discount",
                    "Введите новую скидку в процентах.",
                )
                return
            if data.startswith("admin_promo_edit_days_"):
                await admin_billing_service.start_edit_promo_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_promo_edit_days",
                    "Введите новое количество бесплатных дней.",
                )
                return
            if data.startswith("admin_promo_edit_uses_"):
                await admin_billing_service.start_edit_promo_field(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                    "admin_promo_edit_uses",
                    "Введите новое максимальное количество использований.",
                )
                return
            if data.startswith("admin_promo_toggle_active_"):
                await admin_billing_service.toggle_promo_active(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_promo_toggle_all_"):
                await admin_billing_service.toggle_promo_all_plans(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_promo_assign_toggle_all_"):
                await admin_billing_service.toggle_promo_all_plans_from_assign_menu(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_promo_assign_menu_"):
                await admin_billing_service.show_promo_assign_menu(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_promo_assign_toggle_"):
                _, _, _, _, promo_id, plan_id = data.split("_", 5)
                await admin_billing_service.toggle_promo_plan_assignment(self.client, chat_id, int(promo_id), int(plan_id))
                return
            if data.startswith("admin_promo_delete_"):
                await admin_billing_service.delete_promo(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_toggle_sub_notifs":
                await admin_payments_service.toggle_notifications(self.client, chat_id)
                return
            if data == "admin_toggle_subscriptions":
                await admin_payments_service.toggle_subscriptions(self.client, chat_id)
                return
            if data == "admin_set_welcome_bonus":
                await admin_payments_service.start_set_bonus(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_payment_keys_menu":
                await admin_payments_service.show_keys(self.client, chat_id)
                return
            if data.startswith("set_payment_key_"):
                await admin_payments_service.start_set_key(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    data.replace("set_payment_key_", "", 1),
                )
                return
            if data == "admin_topics":
                await admin_topics_service.list_topics(self.client, chat_id)
                return
            if data == "admin_create_topic":
                await admin_topics_service.start_create_topic(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_edit_topic_"):
                await admin_topics_service.show_topic_editor(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_edit_name_"):
                await admin_topics_service.start_edit_name(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_edit_prompt_"):
                await admin_topics_service.start_edit_prompt(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_edit_intro_"):
                await admin_topics_service.start_edit_intro(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_toggle_active_"):
                await admin_topics_service.toggle_active(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_toggle_admin_"):
                await admin_topics_service.toggle_admin_only(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_toggle_menu_"):
                await admin_topics_service.toggle_menu(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_toggle_list_"):
                await admin_topics_service.toggle_list(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_topic_kb_") and "_page_" in data:
                payload = data.replace("admin_topic_kb_", "", 1)
                topic_id_str, page_str = payload.split("_page_", 1)
                await admin_kb_service.show_topic_assignments(self.client, chat_id, int(topic_id_str), int(page_str))
                return
            if data.startswith("admin_topic_kb_toggle_"):
                payload = data.replace("admin_topic_kb_toggle_", "", 1)
                topic_id_str, kb_id_str, page_str = payload.split("_", 2)
                await admin_kb_service.toggle_topic_assignment(self.client, chat_id, int(topic_id_str), int(kb_id_str), int(page_str))
                return
            if data.startswith("admin_topic_delete_"):
                await admin_topics_service.delete_topic(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_kb":
                await admin_kb_service.list_entries(self.client, chat_id, 0)
                return
            if data.startswith("admin_kb_page_"):
                await admin_kb_service.list_entries(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_kb_create":
                await admin_kb_service.start_create_entry(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_kb_open_"):
                await admin_kb_service.show_entry_editor(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_kb_edit_filename_"):
                await admin_kb_service.start_edit_filename(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_kb_edit_content_"):
                await admin_kb_service.start_edit_content(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_kb_toggle_general_"):
                await admin_kb_service.toggle_general_mode(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_kb_delete_"):
                await admin_kb_service.delete_entry(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_content":
                await admin_content_service.show_content_list(self.client, chat_id)
                return
            if data.startswith("admin_edit_content_"):
                await admin_content_service.show_content_editor(self.client, chat_id, data.replace("admin_edit_content_", "", 1))
                return
            if data.startswith("admin_content_edit_text_"):
                await admin_content_service.start_text_edit(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    data.replace("admin_content_edit_text_", "", 1),
                )
                return
            if data.startswith("admin_toggle_content_visibility_"):
                await admin_content_service.toggle_visibility(
                    self.client,
                    chat_id,
                    data.replace("admin_toggle_content_visibility_", "", 1),
                )
                return
            if data == "admin_referral_menu":
                await admin_referral_service.show_menu(self.client, chat_id)
                return
            if data == "admin_referral_settings":
                await admin_referral_service.show_settings(self.client, chat_id)
                return
            if data == "admin_referral_toggle_enabled":
                await admin_referral_service.toggle_enabled(self.client, chat_id)
                return
            if data == "admin_referral_toggle_pay_bonus":
                await admin_referral_service.toggle_pay_bonus(self.client, chat_id)
                return
            if data == "admin_referral_toggle_pay_first_only":
                await admin_referral_service.toggle_pay_first_only(self.client, chat_id)
                return
            if data == "admin_referral_set_bonus_referrer":
                await admin_referral_service.start_set_bonus_referrer(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_referral_set_bonus_referral":
                await admin_referral_service.start_set_bonus_referral(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_referral_set_pay_days":
                await admin_referral_service.start_set_pay_days(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_referral_set_btn_name":
                await admin_referral_service.start_set_btn_name(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_referral_set_sub_btn_name":
                await admin_referral_service.start_set_sub_btn_name(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_referral_cancel_input":
                await admin_referral_service.cancel_input(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_referral_templates":
                await admin_referral_service.show_templates(self.client, chat_id)
                return
            if data == "admin_ref_tpl_add":
                await admin_referral_service.start_add_template(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_ref_tpl_edit_"):
                await admin_referral_service.start_edit_template(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_ref_tpl_toggle_"):
                await admin_referral_service.toggle_template(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_ref_tpl_up_"):
                await admin_referral_service.move_template(self.client, chat_id, int(data.rsplit("_", 1)[1]), "up")
                return
            if data.startswith("admin_ref_tpl_down_"):
                await admin_referral_service.move_template(self.client, chat_id, int(data.rsplit("_", 1)[1]), "down")
                return
            if data.startswith("admin_ref_tpl_del_confirm_"):
                await admin_referral_service.delete_template(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_ref_tpl_del_"):
                await admin_referral_service.confirm_delete_template(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_ref_tpl_"):
                await admin_referral_service.show_template_detail(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_referral_referrers_page_"):
                await admin_referral_service.show_referrers_page(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_referral_referrers_"):
                await admin_referral_service.show_referrers_page(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_referral_referrer_detail_"):
                payload = data.replace("admin_referral_referrer_detail_", "", 1)
                parts = payload.rsplit("_", 1)
                await admin_referral_service.show_referrer_detail(self.client, chat_id, int(parts[0]), int(parts[1]))
                return
            if data.startswith("admin_referral_referrer_"):
                payload = data.replace("admin_referral_referrer_", "", 1)
                parts = payload.rsplit("_", 1)
                await admin_referral_service.show_referrer_detail(self.client, chat_id, int(parts[0]), int(parts[1]))
                return
            if data == "admin_manage_buttons":
                await admin_buttons_service.show_buttons(self.client, chat_id)
                return
            if data == "admin_add_button":
                await admin_buttons_service.start_create_button(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_button_open_"):
                await admin_buttons_service.show_button_editor(self.client, chat_id, data.replace("admin_button_open_", "", 1))
                return
            if data.startswith("admin_button_edit_title_"):
                await admin_buttons_service.start_edit_title(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    data.replace("admin_button_edit_title_", "", 1),
                )
                return
            if data.startswith("admin_button_toggle_visibility_"):
                await admin_buttons_service.toggle_visibility(self.client, chat_id, data.replace("admin_button_toggle_visibility_", "", 1))
                return
            if data.startswith("admin_button_move_up_"):
                await admin_buttons_service.move_button(self.client, chat_id, data.replace("admin_button_move_up_", "", 1), "up")
                return
            if data.startswith("admin_button_move_down_"):
                await admin_buttons_service.move_button(self.client, chat_id, data.replace("admin_button_move_down_", "", 1), "down")
                return
            if data.startswith("admin_button_delete_"):
                await admin_buttons_service.delete_button(self.client, chat_id, data.replace("admin_button_delete_", "", 1))
                return
            if data == "admin_manage_admins":
                await admin_admins_service.show_admins(self.client, chat_id)
                return
            if data == "admin_add_admin":
                await admin_admins_service.start_add_admin(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_profile_"):
                await admin_admins_service.show_admin_profile(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_toggle_history_"):
                await admin_admins_service.toggle_history_access(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_revoke_"):
                await admin_admins_service.revoke_admin(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_mailing_menu":
                await admin_mailing_service.show_menu(self.client, chat_id)
                return
            if data == "mailing_create":
                await admin_mailing_service.start_create(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("mailing_audience_"):
                await admin_mailing_service.choose_audience(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    data.replace("mailing_audience_", "", 1),
                )
                return
            if data == "mailing_edit_content":
                await admin_mailing_service.restart_text_edit(self.client, self.states, chat_id, user_id)
                return
            if data == "mailing_confirm_send":
                await admin_mailing_service.confirm_send(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("mailing_history_page_"):
                await admin_mailing_service.show_history(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("mailing_details_"):
                await admin_mailing_service.show_details(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_test_menu":
                await admin_tests_service.show_menu(self.client, chat_id)
                return
            if data == "admin_test_toggle_status":
                await admin_tests_service.toggle_status(self.client, chat_id)
                return
            if data == "admin_test_links":
                await admin_tests_service.show_links(self.client, chat_id)
                return
            if data == "admin_test_set_link_admin":
                await admin_tests_service.start_set_admin_username(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_test_set_link_marathon":
                await admin_tests_service.start_set_marathon_url(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_secret_questions":
                await admin_tests_service.show_secret_questions(self.client, chat_id)
                return
            if data == "admin_test_questions":
                await admin_test_content_service.list_questions(self.client, chat_id)
                return
            if data == "admin_test_question_create":
                await admin_test_content_service.start_create_question(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_edit_test_question_"):
                await admin_test_content_service.show_question_editor(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_test_question_edit_text_"):
                await admin_test_content_service.start_edit_question_text(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                )
                return
            if data.startswith("admin_test_question_category_menu_"):
                await admin_test_content_service.show_question_category_menu(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_test_question_set_category_"):
                payload = data.replace("admin_test_question_set_category_", "", 1)
                question_id_str, category = payload.split("_", 1)
                await admin_test_content_service.set_question_category(self.client, chat_id, int(question_id_str), category)
                return
            if data.startswith("admin_test_question_toggle_reverse_"):
                await admin_test_content_service.toggle_question_reverse(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_test_question_edit_sort_"):
                await admin_test_content_service.start_edit_question_sort(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(data.rsplit("_", 1)[1]),
                )
                return
            if data.startswith("admin_test_question_delete_"):
                await admin_test_content_service.delete_question(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_test_question_create_category_"):
                category = data.replace("admin_test_question_create_category_", "", 1)
                await admin_test_content_service.choose_new_question_category(self.client, self.states, chat_id, user_id, category)
                return
            if data.startswith("admin_test_question_create_reverse_"):
                reverse_mode = data.replace("admin_test_question_create_reverse_", "", 1)
                await admin_test_content_service.create_question(self.client, self.states, chat_id, user_id, reverse_mode)
                return
            if data == "admin_secret_question_add":
                await admin_tests_service.start_add_secret_question(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_secret_question_delete_"):
                await admin_tests_service.delete_secret_question(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_edit_test_prompt":
                await admin_tests_service.start_edit_prompt(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_case_studies_page_"):
                await admin_test_content_service.list_case_studies(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_case_study_create":
                await admin_test_content_service.start_create_case_study(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_edit_case_study_"):
                payload = data.replace("admin_edit_case_study_", "", 1)
                case_id_str, page_str = payload.rsplit("_", 1)
                await admin_test_content_service.show_case_study_editor(self.client, chat_id, int(case_id_str), int(page_str))
                return
            if data.startswith("admin_case_study_edit_text_"):
                payload = data.replace("admin_case_study_edit_text_", "", 1)
                case_id_str, page_str = payload.rsplit("_", 1)
                await admin_test_content_service.start_edit_case_study(
                    self.client,
                    self.states,
                    chat_id,
                    user_id,
                    int(case_id_str),
                    int(page_str),
                )
                return
            if data.startswith("admin_case_study_delete_"):
                payload = data.replace("admin_case_study_delete_", "", 1)
                case_id_str, page_str = payload.rsplit("_", 1)
                await admin_test_content_service.delete_case_study(self.client, chat_id, int(case_id_str), int(page_str))
                return
            # ── Client extras ──────────────────────────────────────────────
            if data.startswith("download_history_"):
                await admin_clients_service.download_history_txt(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_delete_history_confirmed_"):
                await admin_clients_service.delete_history_confirmed(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_delete_history_"):
                await admin_clients_service.confirm_delete_history(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("client_payment_info_"):
                await admin_clients_service.show_client_payment_info(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_reset_sub_confirmed_"):
                await admin_clients_service.reset_subscription_confirmed(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_reset_sub_"):
                await admin_clients_service.reset_subscription(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("reset_user_promos_"):
                await admin_billing_service.reset_user_promos(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            # ── Payment stats & log ────────────────────────────────────────
            if data == "admin_payment_stats":
                await admin_payments_service.show_payment_stats(self.client, chat_id)
                return
            if data.startswith("admin_plog_"):
                parts = data.replace("admin_plog_", "", 1).split("_", 1)
                await admin_payments_service.show_payment_log(self.client, chat_id, int(parts[0]), parts[1] if len(parts) > 1 else "all")
                return
            # ── AI extras ─────────────────────────────────────────────────
            if data == "admin_ai_toggle_image_generation":
                await admin_ai_service.toggle_image_generation(self.client, chat_id)
                return
            if data == "admin_ai_toggle_image_edit":
                await admin_ai_service.toggle_image_edit(self.client, chat_id)
                return
            if data == "admin_ai_image_generation_models":
                await admin_ai_service.show_image_generation_models(self.client, chat_id)
                return
            if data.startswith("admin_ai_set_image_gen_model_"):
                await admin_ai_service.set_image_generation_model(self.client, chat_id, data.replace("admin_ai_set_image_gen_model_", "", 1))
                return
            if data == "admin_ai_image_edit_models":
                await admin_ai_service.show_image_edit_models(self.client, chat_id)
                return
            if data.startswith("admin_ai_set_image_edit_model_"):
                await admin_ai_service.set_image_edit_model(self.client, chat_id, data.replace("admin_ai_set_image_edit_model_", "", 1))
                return
            if data == "admin_ai_toggle_fallback":
                await admin_ai_service.toggle_fallback(self.client, chat_id)
                return
            if data == "admin_ai_fallback_models":
                await admin_ai_service.show_fallback_models(self.client, chat_id)
                return
            if data.startswith("admin_ai_set_fallback_provider_"):
                await admin_ai_service.set_fallback_provider(self.client, chat_id, data.replace("admin_ai_set_fallback_provider_", "", 1))
                return
            if data.startswith("admin_ai_save_fallback_"):
                payload = data.replace("admin_ai_save_fallback_", "", 1)
                provider, model_name = payload.split("_", 1)
                await admin_ai_service.save_fallback_model(self.client, chat_id, provider, model_name)
                return
            if data == "admin_ai_set_kie_threshold":
                await admin_ai_service.start_set_kie_threshold(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_ai_set_kie_field_"):
                await admin_ai_service.start_set_kie_field(self.client, self.states, chat_id, user_id, data.replace("admin_ai_set_kie_field_", "", 1))
                return
            # ── Billing extras ────────────────────────────────────────────
            if data.startswith("admin_plan_toggle_is_trial_"):
                await admin_billing_service.toggle_plan_is_trial(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_upgrade_target_"):
                await admin_billing_service.show_upgrade_target_picker(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_plan_clear_upgrade_"):
                await admin_billing_service.clear_upgrade_target(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            # ── Mailing extras ────────────────────────────────────────────
            if data.startswith("mailing_send_test_"):
                await admin_mailing_service.send_test(self.client, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("mailing_toggle_enabled_"):
                await admin_mailing_service.toggle_enabled(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            # ── Collections ───────────────────────────────────────────────
            if data.startswith("admin_collections_page_"):
                await admin_collections_service.show_list(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_coll_view_"):
                await admin_collections_service.show_detail(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_coll_create":
                await admin_collections_service.start_create(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("admin_coll_rename_"):
                await admin_collections_service.start_rename(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_coll_delete_"):
                await admin_collections_service.delete(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_coll_files_"):
                parts = data.split("_")
                # admin_coll_files_{coll_id}_{page}
                await admin_collections_service.show_files(self.client, chat_id, int(parts[3]), int(parts[4]))
                return
            if data.startswith("coll_file_"):
                # coll_file_{action}_{coll_id}_{media_id}_{page}
                parts = data.split("_")
                await admin_collections_service.toggle_file(self.client, chat_id, parts[2], int(parts[3]), int(parts[4]), int(parts[5]))
                return
            # ── Topic media ───────────────────────────────────────────────
            if data.startswith("admin_topic_media_"):
                # admin_topic_media_{topic_id}_{page}
                parts = data.split("_")
                await admin_topic_media_service.show_list(self.client, chat_id, int(parts[3]), int(parts[4]))
                return
            if data.startswith("admin_media_view_"):
                await admin_topic_media_service.show_media_detail(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_media_editname_"):
                await admin_topic_media_service.start_edit_name(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_media_editcat_"):
                await admin_topic_media_service.start_edit_category(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_media_editdesc_"):
                await admin_topic_media_service.start_edit_description(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_media_editfile_"):
                await admin_topic_media_service.start_edit_file(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_media_add_"):
                await admin_topic_media_service.start_add_media(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data.startswith("admin_media_delete_"):
                await admin_topic_media_service.delete_media(self.client, chat_id, int(data.rsplit("_", 1)[1]))
                return
            # ── Export with date filters ───────────────────────────────────
            if data == "admin_export_date_filter":
                await admin_export_service.show_date_filter_menu(self.client, self.states, chat_id, user_id)
                return
            if data == "admin_export_date_all":
                await admin_export_service.set_date_preset(self.client, self.states, chat_id, user_id, 0)
                return
            if data.startswith("admin_export_date_preset_"):
                await admin_export_service.set_date_preset(self.client, self.states, chat_id, user_id, int(data.rsplit("_", 1)[1]))
                return
            if data == "admin_export_date_manual":
                await admin_export_service.start_date_manual_from(self.client, self.states, chat_id, user_id)
                return
            if data.startswith("run_export_"):
                parts = data.split("_")
                fmt = parts[2]
                anonymize = parts[3] == "yes" if len(parts) > 3 else False
                state_snap = await self.states.get(user_id)
                sdata = state_snap.data if state_snap else {}
                await admin_export_service.run_export(self.client, chat_id, sdata.get("date_from"), sdata.get("date_to"), anonymize)
                return
        await self.client.answer_callback(callback.callback_id, notification="Команда пока не реализована в MAX-версии")


async def webhook_handler(request: web.Request) -> web.Response:
    settings = request.app["settings"]
    if settings.webhook_secret:
        if request.headers.get("X-Max-Bot-Api-Secret") != settings.webhook_secret:
            max_log.error("Rejected MAX webhook due to invalid secret remote=%s path=%s", request.remote, request.path)
            return web.Response(status=403, text="forbidden")
    try:
        update = await request.json()
    except Exception:
        max_log.exception("Failed to decode MAX webhook JSON remote=%s path=%s", request.remote, request.path)
        return web.Response(status=400, text="invalid json")
    _spawn_update_task(request.app, update)
    return web.Response(text="ok")


async def polling_loop(bot_app: MaxBotApplication, client: MaxApiClient) -> None:
    settings = get_settings()
    marker: int | None = None
    while True:
        try:
            data = await client.get_updates(marker, settings.polling_timeout, settings.polling_limit, settings.update_types)
            for update in data.get("updates", []):
                await bot_app.handle_update(update)
            marker = data.get("marker", marker)
        except Exception:
            max_log.exception("MAX polling error marker=%s", marker)
            await asyncio.sleep(3)


async def create_web_app() -> web.Application:
    settings = get_settings()
    validate_webhook_runtime_settings(settings)
    await init_db()
    await init_storage()

    if not settings.max_token:
        raise RuntimeError("MAX_BOT_TOKEN не задан. Добавьте токен MAX в env или config.ini [max].")

    client = MaxApiClient(settings.max_token, settings.max_api_base)
    await client.__aenter__()
    await client.set_commands([
        {"name": "start", "description": "Запустить бота"},
        {"name": "help", "description": "Помощь"},
        {"name": "promo", "description": "Ввести промокод"},
        {"name": "ref", "description": "Реферальная программа"},
        {"name": "test", "description": "Пройти тест"},
        {"name": "admin", "description": "Админ-панель"},
    ])
    bot_app = MaxBotApplication(client)

    if settings.webhook_base_url and not settings.use_polling:
        webhook_url = f"{settings.webhook_base_url}{settings.webhook_path}"
        await client.set_webhook(webhook_url, settings.webhook_secret or None, settings.update_types)
        max_log.info("MAX webhook configured url=%s", webhook_url)

    app = web.Application()
    app["settings"] = settings
    app["max_client"] = client
    app["bot_app"] = bot_app
    app["background_tasks"] = set()
    app.router.add_post(settings.webhook_path, webhook_handler)

    async def on_startup(app: web.Application) -> None:
        if settings.use_polling:
            app["polling_task"] = asyncio.create_task(polling_loop(bot_app, client))
            max_log.info("MAX polling started")
        else:
            max_log.info("MAX webhook app started path=%s", settings.webhook_path)

    async def on_shutdown(app: web.Application) -> None:
        task = app.get("polling_task")
        if task:
            task.cancel()
        background_tasks: set[asyncio.Task[None]] = app.get("background_tasks", set())
        if background_tasks:
            for bg_task in list(background_tasks):
                bg_task.cancel()
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await client.__aexit__(None, None, None)
        max_log.info("MAX application shutdown completed")

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    settings = get_settings()
    web.run_app(create_web_app(), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
