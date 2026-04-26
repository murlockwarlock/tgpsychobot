import asyncio
import math
import html
import re
import io
import os
import logging
import traceback
import textwrap
import json
import zipfile
from types import SimpleNamespace
import ai_integration
import keyboards
from database import TestSession, TestQuestion, TestConfig, SecretTestQuestion, CaseStudy
from aiogram import Router, F, Bot
from pydantic import ValidationError
from aiogram.types import (Message, CallbackQuery, FSInputFile, Document,
                           InlineKeyboardMarkup, InlineKeyboardButton)
from aiogram.filters import CommandStart, Command, StateFilter, Filter, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from sqlalchemy import select, func, update, delete, or_, and_, cast, String, desc
from sqlalchemy.orm import selectinload
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BufferedInputFile
from sqlalchemy.orm import selectinload, relationship
from aiogram.types import PreCheckoutQuery, SuccessfulPayment
from aiogram.types import InputMediaPhoto, InputMediaVideo
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import OWNER_IDS
from database import (async_session_maker, User, Message as DBMessage, AIConfig, KnowledgeBase, Content, IndexingQueue,
                     ContentMedia, Topic, SubscriptionPlan, UserSubscription, PromoCode, SubscriptionConfig, Mailing,
                     RobokassaPayment, YookassaPayment, TrialUsageHistory, RandomMessage, MediaLibrary, UserTopicState, get_all_admin_ids,
                     ReferralPaymentLog, MailingDeliveryLog, TopicMediaDeck,
                     MediaCollection, media_collection_items, topic_collection_association)
from aiogram.types import LabeledPrice
import keyboards as kb
from file_parser import parse_file, parse_questions_file
from ai_integration import (
    get_ai_response,
    generate_openai_image,
    analyze_image_content,
    InsufficientBalanceError,
    AIServiceError,
    transcribe_voice_message,
    edit_image,
    _call_gemini_api,
    _call_openai_api,
    _call_deepseek_api,
    _call_claude_api,
    _describe_subscription_status,
)
from error_reporting import notify_admins_about_error
from memory_mode import (
    MEMORY_MODE_GLOBAL,
    MEMORY_MODE_RESET,
    MEMORY_MODE_TOPIC,
    get_memory_mode,
    is_global_memory_mode,
    is_topic_memory_mode,
    memory_mode_description,
    next_memory_mode,
)
from mailing_utils import (
    BIRTHDAY_MAILING_TYPE,
    BIRTHDAY_PLACEHOLDER_HINT,
    DEFAULT_BIRTHDAY_TEMPLATE,
    get_mailing_audience_label,
    get_mailing_status_label,
    is_birthday_mailing,
    render_mailing_text,
    send_mailing_content,
)
from prompt_blocks import DEFAULT_SERVICE_PROMPT_TEMPLATE
from sqlalchemy import delete
from datetime import datetime, timedelta, timezone
from telegram_birthdate import extract_birthdate_parts, format_birthdate, has_birthdate
from time_helpers import format_msk, to_msk
from vector_store import delete_document_vectors, update_case_study_index, search_relevant_case, delete_case_study_vectors
from yookassa import Configuration, Payment
from uuid import uuid4
import decimal
import hashlib
from urllib import parse
from urllib.parse import urlparse
from dateutil.relativedelta import relativedelta
from scheduler import (
    process_recurring_payment,
    process_recurring_robokassa_payment,
    check_robokassa_op_state,
    disable_auto_renewal_after_failed_attempts,
    has_robokassa_pending_timed_out,
    _plog_yookassa_tech,
    _encode_log_json,
    _serialize_yookassa_payment,
)
from subscription_dates import extend_subscription_end_date
from subscription_retry_policy import can_retry_manually

router = Router()
log = logging.getLogger(__name__)
plog = logging.getLogger("payment_events")
user_locks = {}
user_message_buffers = {}
user_processing_tasks = {}
user_spread_state = {}  # {user_id: {category, topic_id, rounds_left, cards_per_round, hidden, chosen_card_ids, selected_file_ids}}
PAGE_SIZE = 5
USER_HISTORY_PAGE_SIZE = 10
KB_PAGE_SIZE = 6
ROBOKASSA_INVOICE_LIFETIME = timedelta(hours=2)


async def _sync_user_birthdate_from_telegram(bot: Bot, user: User) -> bool:
    if has_birthdate(user.birth_day, user.birth_month):
        return False
    try:
        chat = await bot.get_chat(user.id)
    except Exception as exc:
        log.debug("Could not fetch chat profile for birthdate user_id=%s error=%s", user.id, exc)
        return False

    birth_day, birth_month, birth_year = extract_birthdate_parts(getattr(chat, "birthdate", None))
    if not has_birthdate(birth_day, birth_month):
        return False

    user.birth_day = birth_day
    user.birth_month = birth_month
    user.birth_year = birth_year
    return True

MODELS_INFO = {
    "Gemini": {
        'gemini-2.5-pro': {
            'name': 'Gemini 2.5 Pro',
            'desc': 'Самая мощная и продвинутая модель с адаптивным мышлением для решения наиболее сложных задач.'
        },
        'gemini-2.5-flash': {
            'name': 'Gemini 2.5 Flash',
            'desc': 'Оптимальная модель по соотношению цены и производительности для большинства повседневных задач.'
        },
        'pricing': '<b>Вход (Pro):</b> $1.25 / 1M токенов\n<b>Выход (Pro):</b> $10.00 / 1M токенов\n(Flash значительно дешевле)'
    },
    "KIE": {
        'gemini-3-flash': {
            'name': 'KIE Gemini 3 Flash',
            'desc': 'Основная KIE-модель для чата, фото и транскрибации через мультимодальный chat API.'
        },
        'gemini-2.5-flash': {
            'name': 'KIE Gemini 2.5 Flash',
            'desc': 'Быстрая KIE-модель для более дешёвых сценариев и fallback.'
        },
        'pricing': '<b>Зависит от выбранной модели в KIE.</b>\nДля чата используйте Gemini 3 Flash.'
    },
    "Claude": {
        'claude-sonnet-4-5-20250929': {
            'name': 'Claude 4.5 Sonnet',
            'desc': 'Наша самая умная модель для сложных агентов и кодирования (Рекомендуется).'
        },
        'claude-opus-4-1-20250805': {
            'name': 'Claude 4.1 Opus',
            'desc': 'Исключительная модель для специализированных задач.'
        },
        'claude-haiku-4-5-20251001': {
            'name': 'Claude 4.5 Haiku',
            'desc': 'Наша самая быстрая модель с почти передовым интеллектом.'
        },
        'claude-3-haiku-20240307': {
            'name': 'Claude 3 Haiku (Legacy)',
            'desc': 'Самая быстрая и компактная модель 3-го поколения для простых задач.'
        },
        'pricing': '<b>Sonnet 4.5:</b> $3 / $15\n<b>Opus 4.1:</b> $15 / $75\n<b>Haiku 4.5:</b> $1 / $5\n<b>Haiku 3:</b> $0.25 / $1.25\n(Вход / Выход за 1M токенов)'
    },
    "Deepseek": {
        'deepseek-chat': {
            'name': 'Deepseek Chat',
            'desc': 'Основная модель для общения, оптимизированная для диалогов и ответов на вопросы.'
        },
        'deepseek-coder': {
            'name': 'Deepseek Coder',
            'desc': 'Специализированная модель для написания и отладки программного кода.'
        },
        'pricing': '<b>Вход:</b> $0.14 / 1M токенов\n<b>Выход:</b> $0.28 / 1M токенов'
    },
    "OpenAI": {
        'gpt-4o': {
            'name': 'GPT-4o',
            'desc': 'Новейшая, самая быстрая и мощная модель от OpenAI.'
        },
        'gpt-4-turbo': {
            'name': 'GPT-4 Turbo',
            'desc': 'Модель с большим окном контекста (128K) и актуальными знаниями.'
        },
        'gpt-3.5-turbo': {
            'name': 'GPT-3.5 Turbo',
            'desc': 'Быстрая и недорогая модель для простых задач и быстрого ответа.'
        },
        'pricing': '<b>GPT-4o:</b> $5 / $15\n<b>GPT-4T:</b> $10 / $30\n<b>GPT-3.5T:</b> $0.50 / $1.50\n(Вход / Выход за 1M токенов)'
    }
}


CATEGORY_NAMES = {
    "body": "Отношение к телу",
    "face": "Отношение к лицу",
    "age": "Отношение к возрасту",
    "health": "Отношение к здоровью",
    "abilities": "Отношение к способностям",
    "relations": "Отношения с окружающими",
    "success": "Успешность/Реализация целей"
}


PROMPT_BLOCKS = {
    "shared": {
        "db_field": "shared_prompt_block",
        "title": "Общий блок для тем",
        "empty_text": "Блок пуст.",
        "download_callback": "download_shared_prompt_block",
        "filename": "shared_prompt_block.txt",
        "description": (
            "Этот текст будет автоматически добавляться после общего или тематического промпта "
            "во всех обычных диалогах. Подходит для мини-FAQ, общих правил и подсказок, которых нет в БЗ."
        ),
        "placeholders": "",
    },
    "service": {
        "db_field": "service_prompt_block",
        "title": "Служебный блок промпта",
        "empty_text": "Блок пуст. Это отключит служебные правила форматирования, медиа и коротких ответов.",
        "download_callback": "download_service_prompt_block",
        "filename": "service_prompt_block.txt",
        "description": (
            "Этот блок добавляется после основного промпта и содержит служебные инструкции, "
            "которые раньше были захардкожены в коде."
        ),
        "placeholders": (
            "Доступные плейсхолдеры:\n"
            "<code>{available_media_text}</code>\n"
            "<code>{test_context_injection}</code>\n"
            "<code>{short_response_instruction}</code>"
        ),
    },
}


class AdminStates(StatesGroup):
    set_api_key = State()
    set_model = State()
    set_system_prompt = State()
    set_prompt_block = State()
    upload_kb_file = State()
    edit_content = State()
    set_topic_name = State()
    set_topic_prompt = State()
    set_topic_intro_msg = State()
    set_topic_btn_text = State()
    set_topic_btn_payload = State()
    set_content_btn_text = State()
    set_content_btn_payload = State()
    set_plan_name = State()
    set_plan_description = State()
    set_plan_price = State()
    set_plan_duration_unit = State()
    set_plan_duration_value = State()
    set_promo_code = State()
    set_promo_discount = State()
    set_promo_days = State()
    set_promo_uses = State()
    set_single_payment_key = State()
    edit_plan_field = State()
    edit_promo_field = State()
    add_admin_id = State()
    mailing_audience = State()
    mailing_content = State()
    mailing_media_position = State()
    mailing_confirmation = State()

    set_button_title = State()
    add_button_key = State()
    add_button_title = State()
    delete_button_key = State()
    set_voice_limit = State()
    set_plan_upgrade_target = State()
    set_context_first_limit = State()
    set_context_recent_limit = State()
    set_temperature = State()
    set_kie_credit_threshold = State()
    upload_test_questions_file = State()
    set_test_system_prompt = State()
    set_welcome_bonus_days = State()
    search_client = State()
    selecting_for_export = State()
    export_date_from = State()
    export_date_to = State()
    set_referral_bonus_referrer = State()
    set_referral_bonus_referral = State()
    set_referral_pay_days = State()
    set_referral_btn_name = State()
    set_referral_sub_btn_name = State()


class UserStates(StatesGroup):
    awaiting_disclaimer_acceptance = State()
    awaiting_name = State()
    awaiting_promo_code = State()
    awaiting_new_name = State()
    awaiting_gender = State()
    awaiting_age = State()
    in_test = State()
    secret_test_answering = State()


class TestButtonFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if not message.text:
            return False
        async with async_session_maker() as session:
            stmt = select(Content.button_title).where(Content.key == "test_button").limit(1)
            title = await session.scalar(stmt)
            return title is not None and message.text == title


class TopicsButtonFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if not message.text: return False
        async with async_session_maker() as session:
            config = await session.get(SubscriptionConfig, 1)
            btn_name = config.topics_btn_name if config else "📚 Темы диалога"
            return message.text == btn_name

class ReferralButtonFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if not message.text:
            return False
        async with async_session_maker() as session:
            config = await session.get(SubscriptionConfig, 1)
            if not config or not config.referral_enabled:
                return False
            return message.text == config.referral_btn_name


class DynamicButtonFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if not message.text:
            return False
        async with async_session_maker() as session:
            stmt = select(Content.key).where(Content.button_title == message.text, Content.is_visible == True).limit(1)
            result = await session.scalar(stmt)
            if result == "test_button":
                return False
            return result is not None


class TopicPhrasesState(StatesGroup):
    waiting_for_file = State()


class AdminMediaState(StatesGroup):
    waiting_for_file = State()
    waiting_for_name = State()
    waiting_for_category = State()
    waiting_for_description = State()
    editing_name = State()
    editing_category = State()
    editing_description = State()
    editing_file = State()


class AdminCollectionState(StatesGroup):
    waiting_for_name = State()
    waiting_for_rename = State()
    waiting_for_upload_file = State()


def _user_ref(user_id: int, username: str = None, full_name: str = None) -> str:
    """Форматирует кликабельную ссылку на пользователя для уведомлений админу."""
    link = f"<a href='tg://user?id={user_id}'>перейти в профиль</a>"
    if username:
        return f"@{html.escape(username)} ({link})"
    name = html.escape(full_name) if full_name else str(user_id)
    return f"<a href='tg://user?id={user_id}'>{name}</a>"


def _resolve_ai_provider_model(config: AIConfig | None, channel: str) -> tuple[str | None, str | None]:
    if not config:
        return None, None
    if channel == "chat":
        provider = config.provider
        model = getattr(config, f"{provider.lower()}_model", None) if provider else None
        return provider, model
    if channel == "transcription":
        provider = config.transcription_provider
        if provider == "Gemini":
            model = config.gemini_model
        elif provider == "KIE":
            model = getattr(config, "kie_transcription_model", None)
        else:
            model = "whisper-1" if provider == "OpenAI" else None
        return provider, model
    if channel == "vision":
        return config.vision_provider, config.vision_model
    if channel == "image_generation":
        return getattr(config, "image_generation_provider", None), getattr(config, "image_generation_model", None)
    if channel == "image_edit":
        return getattr(config, "image_edit_provider", None), getattr(config, "image_edit_model", None)
    return None, None


async def _report_ai_failure(
    bot: Bot,
    *,
    title: str,
    user,
    provider: str | None = None,
    model: str | None = None,
    stage: str | None = None,
    details: str | None = None,
    extra: dict | None = None,
    exception: Exception | None = None,
) -> None:
    await notify_admins_about_error(
        bot,
        title=title,
        user_id=getattr(user, "id", None),
        username=getattr(user, "username", None),
        full_name=getattr(user, "full_name", None),
        provider=provider,
        model=model,
        stage=stage,
        details=details or (str(exception) if exception else None),
        extra=extra,
        exception=exception,
        logger=log,
    )


def _start_chat_action_loop(bot: Bot, chat_id: int, action: str, interval: float = 4.5) -> asyncio.Task:
    async def _loop():
        try:
            while True:
                await bot.send_chat_action(chat_id=chat_id, action=action)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    return asyncio.create_task(_loop())


async def handle_ai_media_content(bot: Bot, user_id: int, response_text: str):
    audio_pattern = r'\[SEND_AUDIO:\s*(.*?)\]'
    random_img_pattern = r'\[RANDOM_IMG:\s*(.+?)(?:\s*\|\s*(\d+))?\s*\]'
    choice_img_pattern = r'\[CHOICE_IMG:\s*(.*?)\s*\|\s*(\d+)(?:\s*\|\s*(\d+))?\]'
    choice_hidden_pattern = r'\[CHOICE_IMG_HIDDEN:\s*(.*?)\s*\|\s*(\d+)(?:\s*\|\s*(\d+))?\]'
    show_img_pattern = r'\[SHOW_IMG:\s*(.*?)\]'

    audios = re.findall(audio_pattern, response_text)
    random_imgs = re.findall(random_img_pattern, response_text)
    choices = re.findall(choice_img_pattern, response_text)
    choices_hidden = re.findall(choice_hidden_pattern, response_text)
    show_imgs = re.findall(show_img_pattern, response_text)

    clean_text = re.sub(audio_pattern, '', response_text)
    clean_text = re.sub(random_img_pattern, '', clean_text)
    clean_text = re.sub(choice_hidden_pattern, '', clean_text)
    clean_text = re.sub(choice_img_pattern, '', clean_text)
    clean_text = re.sub(show_img_pattern, '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

    return clean_text, audios, random_imgs, choices, choices_hidden, show_imgs


async def send_photo_or_document(
    bot: Bot,
    chat_id: int,
    file_id: str,
    *,
    caption: str | None = None,
    parse_mode: str | None = None,
    reply_markup=None,
):
    try:
        await bot.send_photo(
            chat_id=chat_id,
            photo=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
    except Exception:
        await bot.send_document(
            chat_id=chat_id,
            document=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


async def send_card_album(
    bot: Bot,
    chat_id: int,
    file_ids: list[str],
    *,
    context: str,
):
    if not file_ids:
        return

    if len(file_ids) == 1:
        await send_photo_or_document(bot, chat_id, file_ids[0])
        return

    for i in range(0, len(file_ids), 10):
        batch = file_ids[i:i + 10]
        if len(set(batch)) == 1:
            duplicated_file_id = batch[0]
            try:
                file_info = await bot.get_file(duplicated_file_id)
                file_bytes_io = await bot.download_file(file_info.file_path)
                file_bytes = file_bytes_io.read()
                media_group = [
                    InputMediaPhoto(
                        media=BufferedInputFile(
                            file_bytes,
                            filename=f"card_{i + offset + 1}.jpg",
                        )
                    )
                    for offset in range(len(batch))
                ]
                await bot.send_media_group(chat_id=chat_id, media=media_group)
            except Exception as dup_err:
                log.warning(
                    "send_media_group duplicate fallback failed in %s (%d cards): %s",
                    context,
                    len(batch),
                    dup_err,
                )
                for file_id in batch:
                    await send_photo_or_document(bot, chat_id, file_id)
            continue

        media_group = [InputMediaPhoto(media=file_id) for file_id in batch]
        try:
            await bot.send_media_group(chat_id=chat_id, media=media_group)
        except Exception as mg_err:
            log.warning(
                "send_media_group failed in %s (%d cards): %s",
                context,
                len(batch),
                mg_err,
            )
            for file_id in batch:
                await send_photo_or_document(bot, chat_id, file_id)


async def _get_topic_media_ids(session, topic_id: int) -> list[int] | None:
    """Возвращает список media_id из привязанных коллекций (или None если коллекций нет)."""
    if not topic_id:
        return None
    # Сначала проверяем новые коллекции
    coll_stmt = select(topic_collection_association.c.collection_id).where(
        topic_collection_association.c.topic_id == topic_id
    )
    coll_res = await session.execute(coll_stmt)
    coll_ids = [r[0] for r in coll_res.all()]
    if coll_ids:
        media_stmt = select(media_collection_items.c.media_id).where(
            media_collection_items.c.collection_id.in_(coll_ids)
        )
        media_res = await session.execute(media_stmt)
        return [r[0] for r in media_res.all()]
    return None


async def _get_assigned_decks(session, topic_id: int) -> list[str]:
    """Возвращает список имён колод (фоллбэк для старой системы)."""
    if not topic_id:
        return []
    deck_stmt = select(TopicMediaDeck.deck_name).where(TopicMediaDeck.topic_id == topic_id)
    deck_res = await session.execute(deck_stmt)
    return [r[0] for r in deck_res.all()]


def _media_filter(topic_id: int, collection_media_ids: list[int] | None = None, assigned_decks: list[str] | None = None, category: str = None):
    """Универсальный фильтр: коллекции → старые колоды → topic_id."""
    if collection_media_ids is not None:
        if category:
            return and_(MediaLibrary.id.in_(collection_media_ids), MediaLibrary.category == category)
        return or_(MediaLibrary.id.in_(collection_media_ids), MediaLibrary.topic_id == topic_id)
    if assigned_decks:
        if category:
            return MediaLibrary.category == category
        return or_(MediaLibrary.category.in_(assigned_decks), MediaLibrary.topic_id == topic_id)
    if category:
        return and_(MediaLibrary.topic_id == topic_id, MediaLibrary.category == category)
    return MediaLibrary.topic_id == topic_id


async def execute_media_commands(message: Message, response_text: str, user_id: int, bot: Bot):
    async with async_session_maker() as session:
        user_stmt = select(User).where(User.id == user_id)
        user_res = await session.execute(user_stmt)
        user = user_res.scalar_one_or_none()

        if not user or not user.current_topic_id:
            return response_text

        topic_id = user.current_topic_id

        coll_media_ids = await _get_topic_media_ids(session, topic_id)
        assigned_decks = await _get_assigned_decks(session, topic_id) if coll_media_ids is None else None

        audio_matches = re.findall(r"\[SEND_AUDIO:\s*(.+?)\]", response_text)
        for file_name in audio_matches:
            stmt = select(MediaLibrary).where(
                _media_filter(topic_id, coll_media_ids, assigned_decks),
                MediaLibrary.file_name == file_name.strip(),
                MediaLibrary.media_type == 'audio'
            )
            res = await session.execute(stmt)
            media = res.scalar_one_or_none()
            if media:
                await bot.send_audio(chat_id=message.chat.id, audio=media.file_id)

        random_matches = re.findall(r"\[RANDOM_IMG:\s*(.+?)(?:\s*\|\s*(\d+))?\s*\]", response_text)
        all_random_cards = []
        for match in random_matches:
            category = match[0].strip() if isinstance(match, tuple) else match.strip()
            count = int(match[1]) if isinstance(match, tuple) and match[1] else 1
            stmt = select(MediaLibrary).where(
                _media_filter(topic_id, coll_media_ids, assigned_decks, category),
                MediaLibrary.media_type == 'photo',
                MediaLibrary.file_name != '_back',
            ).order_by(func.random()).limit(count)
            res = await session.execute(stmt)
            r_cards = res.scalars().all()
            all_random_cards.extend(r_cards)
        if all_random_cards:
            if len(all_random_cards) == 1:
                await send_photo_or_document(
                    bot,
                    message.chat.id,
                    all_random_cards[0].file_id,
                    caption=all_random_cards[0].description,
                    parse_mode='HTML',
                )
            else:
                await send_card_album(
                    bot,
                    message.chat.id,
                    [c.file_id for c in all_random_cards],
                    context="execute_media_commands.random_cards",
                )

        choice_matches = re.findall(r"\[CHOICE_IMG:\s*(.+?)\s*\|\s*(\d+)(?:\s*\|\s*(\d+))?\]", response_text)
        for match in choice_matches:
            category = match[0].strip()
            cards_per_round = int(match[1])
            rounds = int(match[2]) if match[2] else 1
            stmt = select(MediaLibrary).where(
                _media_filter(topic_id, coll_media_ids, assigned_decks, category),
                MediaLibrary.media_type == 'photo',
                MediaLibrary.file_name != '_back',
            ).order_by(func.random()).limit(cards_per_round)
            res = await session.execute(stmt)
            cards = res.scalars().all()
            if cards:
                if rounds > 1:
                    user_spread_state[user_id] = {
                        'category': category, 'topic_id': topic_id,
                        'rounds_left': rounds - 1, 'cards_per_round': cards_per_round,
                        'hidden': False, 'chosen_card_ids': [], 'selected_file_ids': []
                    }
                await send_card_album(
                    bot,
                    message.chat.id,
                    [c.file_id for c in cards],
                    context="execute_media_commands.choice_spread",
                )
                card_ids = [c.id for c in cards]
                kb_markup = keyboards.card_selection_keyboard(category, card_ids)
                await message.answer("Выбери карту, которая тебе откликается:", reply_markup=kb_markup)

        choice_hidden_matches = re.findall(r"\[CHOICE_IMG_HIDDEN:\s*(.+?)\s*\|\s*(\d+)(?:\s*\|\s*(\d+))?\]", response_text)
        for match in choice_hidden_matches:
            cat_stripped = match[0].strip()
            cards_per_round = int(match[1])
            rounds = int(match[2]) if match[2] else 1
            stmt = select(MediaLibrary).where(
                _media_filter(topic_id, coll_media_ids, assigned_decks, cat_stripped),
                MediaLibrary.media_type == 'photo',
                MediaLibrary.file_name != '_back',
            ).order_by(func.random()).limit(cards_per_round)
            res = await session.execute(stmt)
            cards = res.scalars().all()
            if cards:
                if rounds > 1:
                    user_spread_state[user_id] = {
                        'category': cat_stripped, 'topic_id': topic_id,
                        'rounds_left': rounds - 1, 'cards_per_round': cards_per_round,
                        'hidden': True, 'chosen_card_ids': [], 'selected_file_ids': []
                    }
                back_stmt = select(MediaLibrary).where(
                    MediaLibrary.category == cat_stripped,
                    MediaLibrary.file_name == '_back',
                ).limit(1)
                back_media = await session.scalar(back_stmt)
                if back_media:
                    await send_card_album(
                        bot,
                        message.chat.id,
                        [back_media.file_id for _ in cards],
                        context="execute_media_commands.hidden_choice_spread",
                    )
                card_ids = [c.id for c in cards]
                kb_markup = keyboards.card_selection_keyboard(cat_stripped, card_ids)
                await message.answer("Выбери карту, которая тебе откликается:", reply_markup=kb_markup)

        show_matches = re.findall(r"\[SHOW_IMG:\s*(.+?)\]", response_text)
        for file_name in show_matches:
            stmt = select(MediaLibrary).where(
                MediaLibrary.file_name == file_name.strip(),
                MediaLibrary.media_type == 'photo',
            ).limit(1)
            res = await session.execute(stmt)
            media = res.scalar_one_or_none()
            if media:
                await send_photo_or_document(
                    bot,
                    message.chat.id,
                    media.file_id,
                    caption=media.description,
                    parse_mode='HTML',
                )

    clean_text = re.sub(r"\[(SEND_AUDIO|RANDOM_IMG|CHOICE_IMG_HIDDEN|CHOICE_IMG|SHOW_IMG):.*?\]", "", response_text).strip()
    return clean_text


def _extract_ai_directive_payload(text: str, directive: str) -> tuple[str | None, str]:
    pattern = rf"{directive}:\s*(.+?)(?=\n\s*\n|\n(?:[#>*-]|\d+\.)\s|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None, text.strip()

    payload = match.group(1).strip()
    clean_text = (text[:match.start()] + text[match.end():]).strip()
    return payload, clean_text


async def process_buffered_messages(user_id: int, bot: Bot):
    if user_id not in user_message_buffers:
        return
    messages = user_message_buffers.pop(user_id)
    full_text = "\n".join(messages)

    async def keep_typing_loop():
        try:
            while True:
                await bot.send_chat_action(chat_id=user_id, action="typing")
                await asyncio.sleep(4.5)
        except asyncio.CancelledError:
            pass

    typing_task = asyncio.create_task(keep_typing_loop())
    async with async_session_maker() as session:
        ai_config = await session.get(AIConfig, 1)
    try:
        response_text = await ai_integration.generate_response(user_id, full_text)
        typing_task.cancel()

        clean_text, audios, random_imgs, choices, choices_hidden, show_imgs = await handle_ai_media_content(bot, user_id, response_text)

        async with async_session_maker() as session:
            user = await session.get(User, user_id)
            topic_id = user.current_topic_id if user else None
            coll_media_ids = await _get_topic_media_ids(session, topic_id) if topic_id else None
            assigned_decks = await _get_assigned_decks(session, topic_id) if topic_id and coll_media_ids is None else None

            # --- 1. Текст AI (сначала предисловие) ---

            image_prompt, text_part = _extract_ai_directive_payload(clean_text, "GEN_IMG")
            if image_prompt:
                if text_part:
                    html_text = markdown_to_html(text_part)
                    for chunk in split_html_text(html_text):
                        await _safe_send_html(
                            lambda text, pm: bot.send_message(chat_id=user_id, text=text, parse_mode=pm),
                            chunk,
                        )
                if image_prompt:
                    upload_task = _start_chat_action_loop(bot, user_id, "upload_photo")
                    try:
                        image_data = await ai_integration.generate_image(image_prompt)
                        await bot.send_photo(chat_id=user_id, photo=BufferedInputFile(image_data, filename="gen.png"), caption="✨ Готово!")
                    except Exception as e:
                        gen_provider, gen_model = _resolve_ai_provider_model(ai_config, "image_generation")
                        await _report_ai_failure(
                            bot,
                            title="Сбой генерации изображения",
                            user=SimpleNamespace(id=user_id),
                            provider=gen_provider,
                            model=gen_model,
                            stage="generate_image",
                            details=str(e),
                            extra={"prompt_len": len(image_prompt)},
                            exception=e,
                        )
                        await bot.send_message(chat_id=user_id, text="😔 Не удалось сгенерировать изображение.")
                    finally:
                        upload_task.cancel()
            else:
                if clean_text:
                    html_response = markdown_to_html(clean_text)
                    for chunk in split_html_text(html_response):
                        await _safe_send_html(
                            lambda text, pm: bot.send_message(chat_id=user_id, text=text, parse_mode=pm),
                            chunk,
                        )

            # --- 2. Медиа (карты, аудио) после текста ---

            for audio_name in audios:
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, assigned_decks),
                    MediaLibrary.file_name == audio_name.strip(),
                    MediaLibrary.media_type == 'audio',
                ).limit(1)
                media = await session.scalar(stmt)
                if media:
                    await bot.send_audio(chat_id=user_id, audio=media.file_id, caption=media.description, parse_mode='HTML')

            for file_name in show_imgs:
                stmt = select(MediaLibrary).where(
                    MediaLibrary.file_name == file_name.strip(),
                    MediaLibrary.media_type == 'photo',
                ).limit(1)
                media = await session.scalar(stmt)
                if media:
                    await send_photo_or_document(
                        bot,
                        user_id,
                        media.file_id,
                        caption=media.description,
                        parse_mode='HTML',
                    )

            drawn_cards_info = []
            all_random_cards = []
            for match in random_imgs:
                cat = match[0].strip() if isinstance(match, tuple) else match.strip()
                r_count = int(match[1]) if isinstance(match, tuple) and match[1] else 1
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, assigned_decks, cat),
                    MediaLibrary.media_type == 'photo',
                    MediaLibrary.file_name != '_back',
                ).order_by(func.random()).limit(r_count)
                res = await session.execute(stmt)
                r_cards = res.scalars().all()
                all_random_cards.extend(r_cards)
            if all_random_cards:
                if len(all_random_cards) == 1:
                    await send_photo_or_document(
                        bot,
                        user_id,
                        all_random_cards[0].file_id,
                        caption=all_random_cards[0].description,
                        parse_mode='HTML',
                    )
                else:
                    await send_card_album(
                        bot,
                        user_id,
                        [c.file_id for c in all_random_cards],
                        context="process_buffered_messages.random_cards",
                    )
                for media in all_random_cards:
                    drawn_cards_info.append(f"{media.file_name}: {media.description or 'без описания'}")

            for match in choices:
                cat = match[0].strip()
                cards_per_round = int(match[1])
                rounds = int(match[2]) if match[2] else 1
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, assigned_decks, cat),
                    MediaLibrary.media_type == 'photo',
                    MediaLibrary.file_name != '_back',
                ).order_by(func.random()).limit(cards_per_round)
                res = await session.execute(stmt)
                cards = res.scalars().all()
                if cards:
                    if rounds > 1:
                        user_spread_state[user_id] = {
                            'category': cat, 'topic_id': topic_id,
                            'rounds_left': rounds - 1, 'cards_per_round': cards_per_round,
                            'hidden': False, 'chosen_card_ids': [], 'selected_file_ids': []
                        }
                    await send_card_album(
                        bot,
                        user_id,
                        [c.file_id for c in cards],
                        context="process_buffered_messages.choice_spread",
                    )
                    await bot.send_message(chat_id=user_id, text="Выбери карту, которая тебе откликается:", reply_markup=keyboards.card_selection_keyboard(cat, [c.id for c in cards]))

            for match in choices_hidden:
                cat_stripped = match[0].strip()
                cards_per_round = int(match[1])
                rounds = int(match[2]) if match[2] else 1
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, assigned_decks, cat_stripped),
                    MediaLibrary.media_type == 'photo',
                    MediaLibrary.file_name != '_back',
                ).order_by(func.random()).limit(cards_per_round)
                res = await session.execute(stmt)
                cards = res.scalars().all()
                if cards:
                    if rounds > 1:
                        user_spread_state[user_id] = {
                            'category': cat_stripped, 'topic_id': topic_id,
                            'rounds_left': rounds - 1, 'cards_per_round': cards_per_round,
                            'hidden': True, 'chosen_card_ids': [], 'selected_file_ids': []
                        }
                    back_stmt = select(MediaLibrary).where(
                        MediaLibrary.category == cat_stripped,
                        MediaLibrary.file_name == '_back',
                    ).limit(1)
                    back_media = await session.scalar(back_stmt)
                    if back_media:
                        await send_card_album(
                            bot,
                            user_id,
                            [back_media.file_id for _ in cards],
                            context="process_buffered_messages.hidden_choice_spread",
                        )
                    await bot.send_message(chat_id=user_id, text="Выбери карту, которая тебе откликается:", reply_markup=keyboards.card_selection_keyboard(cat_stripped, [c.id for c in cards]))

            if user:
                session.add(DBMessage(user_id=user.id, role='user', content=full_text, dialogue_id=user.current_dialogue_id, topic_id=user.current_topic_id))
                session.add(DBMessage(user_id=user.id, role='assistant', content=response_text, dialogue_id=user.current_dialogue_id, topic_id=user.current_topic_id))
                if drawn_cards_info:
                    cards_text = "; ".join(drawn_cards_info)
                    card_system_msg = f"[СИСТЕМА: Случайно выпала карта: {cards_text}. Дай интерпретацию этой карты.]"
                    session.add(DBMessage(user_id=user.id, role='user', content=card_system_msg, dialogue_id=user.current_dialogue_id, topic_id=user.current_topic_id))
                await session.commit()

        if drawn_cards_info:
            try:
                cards_text = "; ".join(drawn_cards_info)
                typing_task_interp = asyncio.create_task(keep_typing_loop())
                interpretation = await ai_integration.generate_response(
                    user_id, f"[СИСТЕМА: Случайно выпала карта: {cards_text}. Дай интерпретацию этой карты.]"
                )
                typing_task_interp.cancel()
                clean_interpretation = re.sub(r"\[(SEND_AUDIO|RANDOM_IMG|CHOICE_IMG|CHOICE_IMG_HIDDEN|SHOW_IMG|GEN_IMG):.*?\]", "", interpretation).strip()
                if clean_interpretation:
                    html_interp = markdown_to_html(clean_interpretation)
                    for chunk in split_html_text(html_interp):
                        await _safe_send_html(
                            lambda text, pm: bot.send_message(chat_id=user_id, text=text, parse_mode=pm),
                            chunk,
                        )
                    async with async_session_maker() as s2:
                        u2 = await s2.get(User, user_id)
                        s2.add(DBMessage(user_id=user_id, role='assistant', content=interpretation, dialogue_id=u2.current_dialogue_id, topic_id=u2.current_topic_id))
                        await s2.commit()
            except Exception as e:
                provider, model = _resolve_ai_provider_model(ai_config, "chat")
                await _report_ai_failure(
                    bot,
                    title="Сбой интерпретации карт",
                    user=SimpleNamespace(id=user_id),
                    provider=provider,
                    model=model,
                    stage="buffered_card_interpretation",
                    details=str(e),
                    exception=e,
                )

    except AIServiceError as e:
        if typing_task: typing_task.cancel()
        provider, model = _resolve_ai_provider_model(ai_config, "chat")
        await _report_ai_failure(
            bot,
            title="Сбой AI-сервиса",
            user=SimpleNamespace(id=user_id),
            provider=provider,
            model=model,
            stage="process_buffered_messages",
            details=str(e),
            extra={"prompt_len": len(full_text)},
            exception=e,
        )
        await bot.send_message(
            chat_id=user_id,
            text="Упс... У нас что-то сломалось. Мы уже сообщили нашим создателям. Попробуйте вернуться и повторить через несколько минут."
        )
    except Exception as e:
        if typing_task: typing_task.cancel()
        provider, model = _resolve_ai_provider_model(ai_config, "chat")
        await _report_ai_failure(
            bot,
            title="Непредвиденная ошибка AI-потока",
            user=SimpleNamespace(id=user_id),
            provider=provider,
            model=model,
            stage="process_buffered_messages_unexpected",
            details=str(e),
            extra={"prompt_len": len(full_text)},
            exception=e,
        )
        await bot.send_message(chat_id=user_id, text="Произошла ошибка при обработке сообщения.")


@router.callback_query(F.data.startswith("card_select_"))
async def process_card_selection(callback: CallbackQuery, bot: Bot):
    card_id = int(callback.data.rsplit("_", 1)[-1])
    user_id = callback.from_user.id
    final_spread_file_ids = []

    async with async_session_maker() as cfg_session:
        ai_config = await cfg_session.get(AIConfig, 1)

    try:
        await callback.message.delete()
    except Exception:
        pass

    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, card_id)
        if media:
            if media.file_name == '_back':
                await callback.answer("Эта техническая рубашка не должна выбираться.", show_alert=True)
                return

            caption = f"<b>Твой выбор подтвержден.</b>\n\n{media.description or ''}"
            await callback.message.answer_photo(
                photo=media.file_id,
                caption=caption,
                parse_mode='HTML'
            )
            await callback.answer()

            card_info = f"{media.file_name}: {media.description or 'без описания'}"
            card_system_msg = f"[СИСТЕМА: Пользователь выбрал карту: {card_info}. Дай интерпретацию этой карты.]"
            user = await session.get(User, user_id)
            if user:
                session.add(DBMessage(user_id=user_id, role='user', content=card_system_msg,
                                      dialogue_id=user.current_dialogue_id, topic_id=user.current_topic_id))
                await session.commit()

            typing_task_interp = None
            try:
                async def _typing_loop_card():
                    try:
                        while True:
                            await bot.send_chat_action(chat_id=user_id, action="typing")
                            await asyncio.sleep(4.5)
                    except asyncio.CancelledError:
                        pass

                typing_task_interp = asyncio.create_task(_typing_loop_card())
                interpretation = await ai_integration.generate_response(user_id, card_system_msg)
                typing_task_interp.cancel()
                clean_interpretation = re.sub(r"\[(SEND_AUDIO|RANDOM_IMG|CHOICE_IMG|CHOICE_IMG_HIDDEN|SHOW_IMG|GEN_IMG):.*?\]", "", interpretation).strip()
                if clean_interpretation:
                    html_interp = markdown_to_html(clean_interpretation)
                    for chunk in split_html_text(html_interp):
                        await _safe_send_html(
                            lambda text, pm: bot.send_message(chat_id=user_id, text=text, parse_mode=pm),
                            chunk,
                        )
                    async with async_session_maker() as s2:
                        u2 = await s2.get(User, user_id)
                        s2.add(DBMessage(user_id=user_id, role='assistant', content=interpretation,
                                         dialogue_id=u2.current_dialogue_id, topic_id=u2.current_topic_id))
                        await s2.commit()
            except Exception as e:
                if typing_task_interp:
                    typing_task_interp.cancel()
                provider, model = _resolve_ai_provider_model(ai_config, "chat")
                await _report_ai_failure(
                    bot,
                    title="Сбой интерпретации карт",
                    user=callback.from_user,
                    provider=provider,
                    model=model,
                    stage="message_card_interpretation",
                    details=str(e),
                    exception=e,
                )

            spread = user_spread_state.get(user_id)
            if spread:
                spread.setdefault('selected_file_ids', []).append(media.file_id)
            if spread and spread['rounds_left'] > 0:
                spread['chosen_card_ids'].append(card_id)
                spread['rounds_left'] -= 1
                async with async_session_maker() as s3:
                    spread_coll_ids = await _get_topic_media_ids(s3, spread['topic_id'])
                    spread_decks = await _get_assigned_decks(s3, spread['topic_id']) if spread_coll_ids is None else None
                    exclude_ids = spread['chosen_card_ids']
                    stmt = select(MediaLibrary).where(
                        _media_filter(spread['topic_id'], spread_coll_ids, spread_decks, spread['category']),
                        MediaLibrary.media_type == 'photo',
                        MediaLibrary.file_name != '_back',
                        MediaLibrary.id.notin_(exclude_ids),
                    ).order_by(func.random()).limit(spread['cards_per_round'])
                    res = await s3.execute(stmt)
                    next_cards = res.scalars().all()
                    if next_cards:
                        if spread['hidden']:
                            back_stmt = select(MediaLibrary).where(
                                MediaLibrary.category == spread['category'],
                                MediaLibrary.file_name == '_back',
                            ).limit(1)
                            back_media = await s3.scalar(back_stmt)
                            if back_media:
                                await send_card_album(
                                    bot,
                                    user_id,
                                    [back_media.file_id for _ in next_cards],
                                    context="process_card_selection.hidden_next_cards",
                                )
                        else:
                            await send_card_album(
                                bot,
                                user_id,
                                [c.file_id for c in next_cards],
                                context="process_card_selection.next_cards",
                            )
                        kb = keyboards.card_selection_keyboard(spread['category'], [c.id for c in next_cards])
                        await bot.send_message(chat_id=user_id, text="Выбери следующую карту:", reply_markup=kb)
                    else:
                        user_spread_state.pop(user_id, None)
            elif spread:
                spread['chosen_card_ids'].append(card_id)
                final_spread_file_ids = list(spread.get('selected_file_ids', []))
                user_spread_state.pop(user_id, None)
        else:
            await callback.answer("Ошибка: Карта не найдена.", show_alert=True)

    if len(final_spread_file_ids) > 1:
        await bot.send_message(chat_id=user_id, text="Твой расклад целиком:")
        await send_card_album(
            bot,
            user_id,
            final_spread_file_ids,
            context="process_card_selection.final_spread",
        )


def calculate_signature(*args) -> str:
    return hashlib.md5(':'.join(str(arg) for arg in args).encode()).hexdigest()


def parse_response(request: str) -> dict:
    params = {}
    query = parse.unquote(request)
    for item in query.split('&'):
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        params[key] = value
    return params


def check_signature_result(
    order_number: int,
    received_sum: str,
    received_signature: str,
    password: str
) -> bool:
    signature = calculate_signature(received_sum, order_number, password)
    if signature.lower() == received_signature.lower():
        return True
    return False


def generate_payment_link(
        merchant_login: str,
        merchant_password_1: str,
        cost: decimal.Decimal,
        number: int,
        description: str,
        expiration_date: datetime | None = None,
        is_test=0,
        robokassa_payment_url='https://auth.robokassa.ru/Merchant/Index.aspx',
        recurring: bool = False
) -> str:
    signature = calculate_signature(
        merchant_login,
        cost,
        number,
        merchant_password_1
    )

    data = {
        'MerchantLogin': merchant_login,
        'OutSum': cost,
        'InvId': number,
        'Description': description,
        'SignatureValue': signature,
        'IsTest': is_test
    }

    if recurring:
        data['Recurring'] = 'true'
    if expiration_date:
        msk_expiration = expiration_date.astimezone(timezone(timedelta(hours=3)))
        data['ExpirationDate'] = msk_expiration.strftime('%Y-%m-%dT%H:%M')

    return f'{robokassa_payment_url}?{parse.urlencode(data)}'


def build_robokassa_invoice_access_token(invoice_id: int, user_id: int, password: str) -> str:
    return hashlib.md5(f"{invoice_id}:{user_id}:{password}".encode()).hexdigest()


def build_local_robokassa_redirect_url(payment_id: int, user_id: int, password: str) -> str:
    base_webhook_url = os.environ.get("BASE_WEBHOOK_URL", "").rstrip("/")
    webhook_path_prefix = os.environ.get("WEBHOOK_PATH_PREFIX", "")
    token = build_robokassa_invoice_access_token(payment_id, user_id, password)
    return f"{base_webhook_url}{webhook_path_prefix}/pay/robokassa/{payment_id}?token={token}"


def _fix_unclosed_html_tags(text: str) -> str:
    """Полностью перестраивает HTML-теги для корректной вложенности.

    Обрабатывает ВСЕ виды невалидного HTML:
    - Лишние закрывающие теги (</i> без <i>) → удаляются
    - Незакрытые теги (<b> без </b>) → закрываются в конце
    - Перекрёстная вложенность (<b><i>...</b></i>) → исправляется порядок
    - Дублированные теги (<b><b>...</b></b>) → дубли пропускаются
    """
    allowed_tags = {'b', 'i', 's', 'code', 'pre', 'a', 'blockquote'}
    tag_pattern = re.compile(r'<(/?)([a-z1-6]+)([^>]*)>', re.IGNORECASE)

    segments = []
    last_end = 0
    for match in tag_pattern.finditer(text):
        tag_name = match.group(2).lower()
        if tag_name not in allowed_tags:
            continue
        if match.start() > last_end:
            segments.append(('text', text[last_end:match.start()]))
        is_closing = bool(match.group(1))
        segments.append(('close' if is_closing else 'open', tag_name, match.group(0)))
        last_end = match.end()
    if last_end < len(text):
        segments.append(('text', text[last_end:]))

    result = []
    stack = []

    for seg in segments:
        if seg[0] == 'text':
            result.append(seg[1])
        elif seg[0] == 'open':
            tag_name = seg[1]
            if any(t[0] == tag_name for t in stack):
                continue
            stack.append((tag_name, seg[2]))
            result.append(seg[2])
        elif seg[0] == 'close':
            tag_name = seg[1]
            idx = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == tag_name:
                    idx = i
                    break
            if idx is None:
                continue
            tags_to_reopen = []
            while len(stack) > idx + 1:
                t = stack.pop()
                result.append(f'</{t[0]}>')
                tags_to_reopen.append(t)
            stack.pop()
            result.append(f'</{tag_name}>')
            for t in reversed(tags_to_reopen):
                stack.append(t)
                result.append(t[1])

    while stack:
        t = stack.pop()
        result.append(f'</{t[0]}>')

    return ''.join(result)


def markdown_to_html(text: str) -> str:
    if not text:
        return ""

    # Конвертируем существующие HTML-теги в markdown-эквиваленты
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<pre><code>(.*?)</code></pre>', r'```\1```', text, flags=re.DOTALL)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)

    text = html.escape(text, quote=False)
    placeholders = {}

    def _placeholder(prefix, value):
        key = f"\x01{prefix}{len(placeholders)}\x01"
        placeholders[key] = value
        return key

    # Сохраняем escaped-символы как плейсхолдеры (до обработки markdown)
    def save_escaped(match):
        return _placeholder("ESC", match.group(1))

    text = re.sub(r'\\([*_~`#\[\]()\\>!|-])', save_escaped, text)

    # Сохраняем блоки кода
    def save_code_block(match):
        code = match.group(1)
        # Убираем идентификатор языка из первой строки
        code = re.sub(r'^[a-zA-Z0-9_+-]+\n', '', code)
        return _placeholder("CODEBLOCK", code)

    def save_inline_code(match):
        return _placeholder("INLINE", match.group(1))

    text = re.sub(r'```(.*?)```', save_code_block, text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', save_inline_code, text)

    # Горизонтальные линии
    text = re.sub(r'^\s*[\*_-]{3,}\s*$', '———', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\s*$', '———', text, flags=re.MULTILINE)

    # Blockquotes: собираем последовательные строки с > в <blockquote>
    def _format_quote_content(content):
        """Обрабатываем markdown внутри цитаты (заголовки, списки)."""
        content = re.sub(r'^\s*#{1,6}\s+(.*)', lambda m: '<b>' + m.group(1).replace('***', '').replace('**', '').replace('*', '') + '</b>', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*[-*]\s+', '• ', content, flags=re.MULTILINE)
        return content

    def process_blockquotes(txt):
        lines = txt.split('\n')
        result = []
        quote_lines = []

        def flush_quote():
            if not quote_lines:
                return
            content = '\n'.join(quote_lines).strip()
            if content:
                content = _format_quote_content(content)
                result.append(f'<blockquote>{content}</blockquote>')
            quote_lines.clear()

        for line in lines:
            m = re.match(r'^\s*&gt;\s?(.*)', line)
            if m:
                quote_lines.append(m.group(1))
            else:
                flush_quote()
                result.append(line)
        flush_quote()
        return '\n'.join(result)

    text = process_blockquotes(text)

    # Маркированные списки
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)

    # Заголовки → <b>
    text = re.sub(r'^\s*#{1,6}\s+(.*)', lambda m: '\n\n<b>' + m.group(1).replace('***', '').replace('**', '').replace('*', '') + '</b>\n', text, flags=re.MULTILINE)

    # Жирный + курсив (***text*** / ___text___)
    text = re.sub(r'\*\*\*(?=[^<>]*\*\*\*)((?:(?!\n\n)[^<>])+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'___((?:(?!\n\n)[^<>])+?)___', r'<b><i>\1</i></b>', text)
    # Жирный (**text** / __text__)
    text = re.sub(r'\*\*(?=[^<>]*\*\*)((?:(?!\n\n)[^<>])+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(?=[^<>]*__)((?:(?!\n\n)[^<>])+?)__', r'<b>\1</b>', text)
    # Курсив (*text* / _text_)
    text = re.sub(r'(?<!\w)\*(?!\s)([^<>\n]+?)(?<!\s)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_(?!\s)([^<>\n]+?)(?<!\s)_(?!\w)', r'<i>\1</i>', text)
    # Зачёркнутый
    text = re.sub(r'~~(?=[^<>\n]+~~)([^<>\n]+?)~~', r'<s>\1</s>', text)
    # Ссылки
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)

    # Отступы перед заголовками и списками
    text = re.sub(r'([^\n])\n(<b>)', r'\1\n\n\2', text)
    text = re.sub(r'([^\n])\n(•)', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Чистим оставшийся markdown-мусор
    text = text.replace('**', '').replace('__', '').replace('~~', '')
    text = text.replace('<b></b>', '').replace('<i></i>', '')
    text = re.sub(r'(?<![\w*])\*(?![\w*])', '', text)

    # Восстанавливаем плейсхолдеры
    for key, value in placeholders.items():
        if "CODEBLOCK" in key:
            replacement = f'<pre><code>{value}</code></pre>'
        elif "INLINE" in key:
            replacement = f'<code>{value}</code>'
        else:
            # ESC — escaped-символы, показываем как есть
            replacement = value
        text = text.replace(key, replacement)

    text = text.strip()
    text = _fix_unclosed_html_tags(text)
    return text


def remove_markdown(text: str) -> str:
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'\1', text)
    text = re.sub(r'\*(.*?)\*|_(.*?)_', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    return text


def split_message(text: str, max_length: int = 4096) -> list[str]:
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)

            while len(paragraph) > max_length:
                split_at = paragraph.rfind(' ', 0, max_length)
                if split_at == -1:
                    split_at = max_length

                chunks.append(paragraph[:split_at])
                paragraph = paragraph[split_at:].lstrip()

            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n" + paragraph
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_html_text(text: str, max_length: int = 4090) -> list[str]:
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""
    open_tags = []

    tag_re = re.compile(r'(</?[a-z1-6]+(?: [^>]+)?>)', re.IGNORECASE)
    parts = tag_re.split(text)

    def _suffix():
        return "".join([f"</{t[0]}>" for t in reversed(open_tags)])

    def _reopen():
        return "".join([t[1] for t in open_tags])

    def _close_size():
        return sum(len(t[0]) + 3 for t in open_tags)

    for part in parts:
        if not part:
            continue

        if part.startswith('<'):
            tag_match = re.match(r'<(/?)([a-z1-6]+)', part, re.IGNORECASE)
            if tag_match:
                is_closing = bool(tag_match.group(1))
                tag_name = tag_match.group(2).lower()

                if is_closing:
                    if open_tags and open_tags[-1][0] == tag_name:
                        open_tags.pop()
                elif tag_name not in ['br', 'hr', 'img']:
                    open_tags.append((tag_name, part))

            suffix = _suffix()
            if len(current_chunk) + len(part) + len(suffix) > max_length:
                if current_chunk.strip():
                    chunks.append(current_chunk + suffix)
                current_chunk = _reopen()



            current_chunk += part
        else:
            while len(current_chunk) + len(part) + _close_size() > max_length:
                remaining_space = max_length - len(current_chunk) - _close_size()

                if remaining_space <= 10:
                    suffix = _suffix()
                    if current_chunk.strip():
                        chunks.append(current_chunk + suffix)
                    current_chunk = _reopen()
                    remaining_space = max_length - len(current_chunk) - _close_size()

                split_at = remaining_space

                for separator in ('\n\n', '. ', '! ', '? ', '\n'):
                    found_at = part.rfind(separator, 0, remaining_space)
                    if found_at != -1 and found_at > (remaining_space // 3):
                        split_at = found_at + len(separator.rstrip())
                        break
                else:
                    split_at = part.rfind(' ', 0, remaining_space)
                    if split_at == -1:
                        split_at = remaining_space

                content = part[:split_at]
                suffix = _suffix()

                if (current_chunk + content).strip():
                    chunks.append(current_chunk + content + suffix)

                current_chunk = _reopen()
                part = part[split_at:].lstrip()

            current_chunk += part

    if current_chunk.strip():
        clean_chunk = re.sub(r'(<[a-z1-6][^>]*>)+$', '', current_chunk)
        if clean_chunk.strip():
            suffix = _suffix()
            final_c = clean_chunk + suffix
            if re.sub(r'<[^>]+>', '', final_c).strip():
                chunks.append(final_c)

    return [_fix_unclosed_html_tags(c) for c in chunks if re.sub(r'<[^>]+>', '', c).strip()]


def _strip_all_html_tags(text: str) -> str:
    """Удаляет все HTML-теги, оставляя только текст."""
    return re.sub(r'<[^>]+>', '', text)


async def _safe_send_html(send_coro_factory, chunk: str):
    """Отправляет chunk с parse_mode='HTML'.

    При ошибке парсинга — прогоняет через _fix_unclosed_html_tags и пробует снова.
    Если всё равно не получилось — полностью удаляет теги, html-escape-ит текст
    и отправляет с parse_mode='HTML' (чистый текст, но символы безопасны).
    """
    try:
        await send_coro_factory(chunk, 'HTML')
    except Exception:
        fixed = _fix_unclosed_html_tags(chunk)
        try:
            await send_coro_factory(fixed, 'HTML')
        except Exception:
            plain = html.escape(_strip_all_html_tags(chunk), quote=False)
            if plain.strip():
                await send_coro_factory(plain, 'HTML')


def infer_gender(name: str) -> str:
    name = name.strip().lower()
    if not name:
        return 'unknown'

    male_exceptions = {
        'илья', 'никита', 'лука', 'фома', 'савва', 'кузьма', 'данила',
        'миша', 'лёша', 'лёва', 'гоша'
    }

    if name in male_exceptions:
        return 'male'

    if name.endswith(('а', 'я')):
        return 'female'

    if name.endswith('ь'):
        return 'unknown'

    return 'male'


async def send_temp_notification(target, bot, text, delay=5):
    try:
        msg = await bot.send_message(target, text)
        await asyncio.sleep(delay)
        await bot.delete_message(target, msg.message_id)
    except Exception:
        pass


async def is_admin(user_id: int) -> bool:
    if user_id in OWNER_IDS:
        return True
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        return user.is_admin if user else False


@router.message(Command("help"))
async def cmd_help(message: Message):
    if not await is_admin(message.from_user.id):
        user_help_text = (
            "👋 Здравствуйте! Я ваш персональный ИИ-помощник.\n\n"
            "Просто напишите ваш вопрос в этот чат, и я постараюсь на него ответить. "
            "Вы можете использовать кнопки внизу для навигации по основным разделам или для управления диалогом."
        )
        await message.answer(user_help_text)
        return

    admin_help_text = textwrap.dedent("""
        📜 <b>Руководство для Администратора</b>

        ---

        <b>⭐️ Тарифы и Промокоды</b>
        1.  Перейдите в <code>Админ-панель → ⭐️ Подписки</code>.
        2.  Для создания тарифа нажмите <code>📈 Тарифные планы → ➕ Создать новый тариф</code> и следуйте инструкциям.
        3.  Для создания промокода нажмите <code>🎁 Промокоды → ➕ Создать промокод</code> и заполните все шаги.
        4.  Для редактирования, активации/деактивации или удаления существующих тарифов/промокодов, просто нажмите на нужный элемент в списке.

        ---

        <b>💬 Темы диалогов</b>
        1.  Перейдите в <code>Админ-панель → 💬 Темы диалогов</code>.
        2.  Нажмите <code>➕ Создать новую тему</code>, чтобы добавить новую тему.
        3.  Нажмите на любую существующую тему, чтобы войти в меню редактирования.
        4.  <b>Прикрепление файлов из Базы Знаний (БЗ):</b>
            • В меню редактирования темы нажмите <code>📎 Привязать файлы БЗ</code>.
            • В открывшемся списке вы увидите все файлы из вашей БЗ. Нажимайте на них, чтобы привязать (✅) или отвязать (⭕️) файл от текущей темы.
            • <i>Важно:</i> При активной теме бот будет искать ответы <b>только</b> в привязанных к ней файлах.

        ---

        <b>🎛️ Динамические кнопки</b>
        <i>Динамические кнопки — это кнопки в главном меню клиента, которые показывают контент из раздела "Контент".</i>
        1.  <b>Создание:</b>
            • Перейдите в <code>Админ-панель → 🎛️ Кнопки меню</code>.
            • Нажмите <code>➕ Добавить новую кнопку</code>.
            • <b>Шаг 1:</b> Введите уникальный ключ (ID) на латинице, например, `special_offer`. Этот ключ не виден пользователям.
            • <b>Шаг 2:</b> Введите текст, который будет на кнопке (например, "Специальное предложение").
        2.  <b>Наполнение контентом:</b>
            • После создания кнопки перейдите в <code>Админ-панель → ✏️ Контент</code>.
            • В списке появится новый пункт для редактирования вашей кнопки (например, "✏️ Специальное предложение").
            • Нажмите на него и добавьте нужный текст и медиафайлы.
        3.  <b>Управление:</b>
            • В меню <code>🎛️ Кнопки меню</code> вы можете включать/выключать (✅/❌), переименовывать (✏️) или полностью удалять (🗑️) кнопки.
    """)
    await message.answer(admin_help_text)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, bot: Bot, command: CommandObject = None):
    welcome_bonus_text = ""
    async with async_session_maker() as session:
        user = await session.get(User, message.from_user.id)
        is_new_user = False
        if not user:
            is_new_user = True
            new_user = User(
                id=message.from_user.id,
                username=message.from_user.username,
                first_name=message.from_user.full_name,
                is_admin=await is_admin(message.from_user.id)
            )
            session.add(new_user)
            await session.flush()
            user = new_user
        else:
            user.username = message.from_user.username
            user.first_name = message.from_user.full_name
        await _sync_user_birthdate_from_telegram(bot, user)
        if is_new_user:
            sub_config = await session.get(SubscriptionConfig, 1)
            if sub_config and sub_config.welcome_bonus_days > 0 and sub_config.subscriptions_enabled:
                now = datetime.utcnow()
                end_date = now + timedelta(days=sub_config.welcome_bonus_days)
                new_sub = UserSubscription(
                    user_id=user.id,
                    plan_id=None,
                    start_date=now,
                    end_date=end_date,
                    auto_renewal=False,
                    payment_provider='Trial Welcome',
                    payment_attempt_count=0,
                    discount_percent=0
                )
                session.add(new_sub)
                session.add(TrialUsageHistory(
                    user_id=user.id,
                    plan_id=None,
                    used_at=now
                ))
                welcome_bonus_text = (
                    f"🎁 <b>Вам начислен приветственный бонус!</b>\n"
                    f"Бесплатный доступ ко всем функциям на {sub_config.welcome_bonus_days} дн."
                )
        await session.commit()

        if command and command.args:
            args = command.args

            if args.startswith("topic_"):
                try:
                    topic_id = int(args.split("_")[1])
                    topic = await session.get(Topic, topic_id)
                    if topic and topic.is_active:
                        user.current_topic_id = topic_id
                        ai_config = await session.get(AIConfig, 1)
                        memory_mode = get_memory_mode(ai_config) if ai_config else MEMORY_MODE_RESET
                        restored = await _apply_topic_switch(session, user, topic_id, memory_mode)
                        await session.commit()

                        if topic.start_message:
                            text_to_send = topic.start_message
                        else:
                            text_to_send = _topic_switch_message(topic.name, restored, memory_mode)

                        reply_markup = None
                        if topic.start_button_text and topic.start_button_payload:
                            reply_markup = kb.action_button_keyboard(topic.start_button_text, "topic_action")

                        if not reply_markup:
                            reply_markup = await kb.main_client_keyboard()

                        await message.answer(text_to_send, parse_mode="HTML" if topic.start_message else "Markdown",
                                             reply_markup=reply_markup)
                        if welcome_bonus_text:
                            await message.answer(welcome_bonus_text, parse_mode="HTML")
                        return
                except Exception as e:
                    logging.error(f"Error parsing topic deep link: {e}")

            elif args == "sub":
                if welcome_bonus_text:
                    await message.answer(welcome_bonus_text, parse_mode="HTML")
                await show_subscription_info(message, state, bot)
                return

            elif args == "test":
                if welcome_bonus_text:
                    await message.answer(welcome_bonus_text, parse_mode="HTML")
                await cmd_start_test(message, state)
                return

            elif args.startswith("ref_"):
                raw = args[4:]
                ref_id = None
                try:
                    ref_id = int(raw)
                except ValueError:
                    pass

                if ref_id and ref_id != message.from_user.id and is_new_user:
                    sub_config_ref = await session.get(SubscriptionConfig, 1)
                    if sub_config_ref and sub_config_ref.referral_enabled:
                        referrer = await session.get(User, ref_id)
                        if referrer:
                            user.referred_by = ref_id
                            ref_bonus_days = sub_config_ref.referral_bonus_days_referral
                            ref_days_for_referrer = sub_config_ref.referral_bonus_days_referrer
                            now_r = datetime.utcnow()

                            # Bonus to new user: extend existing sub or create new
                            if ref_bonus_days > 0:
                                existing_sub = await session.scalar(
                                    select(UserSubscription).where(UserSubscription.user_id == user.id)
                                )
                                if existing_sub:
                                    existing_sub.end_date += timedelta(days=ref_bonus_days)
                                else:
                                    session.add(UserSubscription(
                                        user_id=user.id,
                                        plan_id=None,
                                        start_date=now_r,
                                        end_date=now_r + timedelta(days=ref_bonus_days),
                                        auto_renewal=False,
                                        payment_provider='Trial Referral',
                                        payment_attempt_count=0,
                                        discount_percent=0
                                    ))
                                welcome_bonus_text = (
                                    f"🎁 <b>Вам начислено {ref_bonus_days} бонусных дн.</b> "
                                    f"за регистрацию по пригласительной ссылке!"
                                )

                            # Bonus to referrer: extend their sub or create new
                            if ref_days_for_referrer > 0:
                                referrer_sub = await session.scalar(
                                    select(UserSubscription).where(UserSubscription.user_id == ref_id)
                                )
                                if referrer_sub and referrer_sub.end_date > now_r:
                                    referrer_sub.end_date += timedelta(days=ref_days_for_referrer)
                                elif referrer_sub:
                                    referrer_sub.plan_id = None
                                    referrer_sub.start_date = now_r
                                    referrer_sub.end_date = now_r + timedelta(days=ref_days_for_referrer)
                                    referrer_sub.payment_provider = 'Trial Referral Bonus'
                                    referrer_sub.auto_renewal = False
                                    referrer_sub.payment_attempt_count = 0
                                else:
                                    session.add(UserSubscription(
                                        user_id=ref_id,
                                        plan_id=None,
                                        start_date=now_r,
                                        end_date=now_r + timedelta(days=ref_days_for_referrer),
                                        auto_renewal=False,
                                        payment_provider='Trial Referral Bonus',
                                        payment_attempt_count=0,
                                        discount_percent=0
                                    ))

                            await session.commit()

                            if ref_days_for_referrer > 0:
                                try:
                                    await bot.send_message(
                                        ref_id,
                                        f"🎉 По вашей реферальной ссылке зарегистрировался новый пользователь!\n"
                                        f"Вам начислено <b>{ref_days_for_referrer} бонусных дн.</b> к доступу. "
                                        f"Спасибо, что рекомендуете нас!",
                                        parse_mode="HTML"
                                    )
                                except Exception:
                                    pass
                # No return — fall through to show welcome message

            else:
                content_key = args

                stmt_dyn = select(Content).where(Content.key == content_key, Content.is_visible == True).limit(1)
                content_obj_dyn = await session.scalar(stmt_dyn)

                if content_obj_dyn:
                    if welcome_bonus_text:
                        await message.answer(welcome_bonus_text, parse_mode="HTML")

                    content = await get_content_from_db(content_key)
                    text_dyn = content.get('text')
                    media_dyn = content.get('media', [])
                    content_order_dyn = getattr(content_obj_dyn, 'content_order', 'media_top')
                    html_text = text_dyn if text_dyn else ""

                    async def send_text_part():
                        if html_text:
                            text_chunks = split_html_text(html_text, 4000)
                            for chunk in text_chunks:
                                await _safe_send_html(
                                    lambda text, pm: message.answer(text, parse_mode=pm),
                                    chunk,
                                )
                                await asyncio.sleep(0.2)

                    async def send_media_part():
                        if media_dyn:
                            if 1 < len(media_dyn) <= 10:
                                media_group = []
                                for item in media_dyn:
                                    media_to_add = InputMediaPhoto(media=item['file_id']) if item[
                                                                                                 'type'] == 'photo' else InputMediaVideo(
                                        media=item['file_id'])
                                    media_group.append(media_to_add)
                                if media_group:
                                    await message.answer_media_group(media_group)
                            elif len(media_dyn) == 1:
                                item = media_dyn[0]
                                if item['type'] == 'photo':
                                    await message.answer_photo(item['file_id'])
                                elif item['type'] == 'video':
                                    await message.answer_video(item['file_id'])
                            elif len(media_dyn) > 10:
                                for i in range(0, len(media_dyn), 10):
                                    chunk = media_dyn[i:i + 10]
                                    media_group = []
                                    for item in chunk:
                                        media_to_add = InputMediaPhoto(media=item['file_id']) if item[
                                                                                                     'type'] == 'photo' else InputMediaVideo(
                                            media=item['file_id'])
                                        media_group.append(media_to_add)
                                    if media_group:
                                        await message.answer_media_group(media_group)
                                        await asyncio.sleep(0.5)

                    if html_text and media_dyn and len(media_dyn) == 1 and len(html_text) <= 1024:
                        item = media_dyn[0]
                        if item['type'] == 'photo':
                            await message.answer_photo(item['file_id'], caption=html_text, parse_mode="HTML")
                        elif item['type'] == 'video':
                            await message.answer_video(item['file_id'], caption=html_text, parse_mode="HTML")
                        return

                    if content_order_dyn == 'media_top':
                        await send_media_part()
                        await asyncio.sleep(0.2)
                        await send_text_part()
                    else:
                        await send_text_part()
                        await asyncio.sleep(0.2)
                        await send_media_part()
                    return

        stmt = select(Content).where(Content.key == "start_message").options(selectinload(Content.media))
        content_obj = await session.scalar(stmt)

    main_kb = await kb.main_client_keyboard()
    if not content_obj:
        text = "Приветствие не задано."
        await message.answer(text, reply_markup=main_kb, parse_mode="HTML")
        if welcome_bonus_text:
            await message.answer(welcome_bonus_text, parse_mode="HTML")
        return

    text = content_obj.text_content
    media = content_obj.media
    order = content_obj.content_order
    inline_kb = None

    if content_obj.action_btn_text and content_obj.action_btn_payload:
        inline_kb = kb.action_button_keyboard(content_obj.action_btn_text, "start_action")

    async def send_text_func(markup):
        if text:
            await message.answer(text, reply_markup=markup, parse_mode="HTML")
            return True
        return False

    async def send_media_func(markup):
        if media:
            if len(media) == 1 and len(text or "") < 1000 and order == 'media_top' and text:
                item = media[0]
                if item.file_type == 'photo':
                    await message.answer_photo(item.file_id, caption=text, reply_markup=markup, parse_mode="HTML")
                elif item.file_type == 'video':
                    await message.answer_video(item.file_id, caption=text, reply_markup=markup, parse_mode="HTML")
                return True
            if len(media) == 1:
                item = media[0]
                if item.file_type == 'photo':
                    await message.answer_photo(item.file_id, reply_markup=markup)
                elif item.file_type == 'video':
                    await message.answer_video(item.file_id, reply_markup=markup)
                return True
            media_group = []
            for item in media:
                media_to_add = InputMediaPhoto(media=item.file_id) if item.file_type == 'photo' else InputMediaVideo(
                    media=item.file_id)
                media_group.append(media_to_add)
            if media_group:
                await message.answer_media_group(media_group)
                return False
        return False

    sent_combined = False
    if media and len(media) == 1 and len(text or "") < 1000 and order == 'media_top':
        sent_combined = await send_media_func(inline_kb if inline_kb else main_kb)

    if not sent_combined:
        if order == 'media_top':
            await send_media_func(main_kb if not inline_kb else None)
            await asyncio.sleep(0.2)
            await send_text_func(inline_kb if inline_kb else main_kb)
        else:
            await send_text_func(inline_kb if inline_kb else main_kb)
            await asyncio.sleep(0.2)
            await send_media_func(main_kb if not inline_kb else None)

    if inline_kb:
        await asyncio.sleep(0.3)
        if welcome_bonus_text:
            await message.answer(welcome_bonus_text, parse_mode="HTML", reply_markup=main_kb)
        else:
            await message.answer("🔽 Воспользуйтесь меню ниже для навигации:", reply_markup=main_kb)
    elif welcome_bonus_text:
        await asyncio.sleep(0.3)
        await message.answer(welcome_bonus_text, parse_mode="HTML")


@router.message(Command("admin"))
async def cmd_admin(message: Message):
    if not await is_admin(message.from_user.id):
        return
    await message.answer("Добро пожаловать в админ-панель!", reply_markup=kb.admin_panel_keyboard())


@router.message(Command("promo"), StateFilter(None))
async def cmd_promo(message: Message, state: FSMContext):
    user_id = message.from_user.id
    now = datetime.utcnow()

    async with async_session_maker() as session:
        user = await session.get(
            User,
            user_id,
            options=[
                selectinload(User.subscription),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )
        is_admin_user = user_id in OWNER_IDS or (user and user.is_admin)
        trial_conditions = [SubscriptionPlan.is_active == True, SubscriptionPlan.is_trial == True]
        if not is_admin_user:
            trial_conditions.append(SubscriptionPlan.admin_only == False)
        trial_plans_result = await session.execute(
            select(SubscriptionPlan)
            .where(*trial_conditions)
            .options(selectinload(SubscriptionPlan.upgrades_to_plan))
            .order_by(SubscriptionPlan.price.asc())
        )
        all_trial_plans = trial_plans_result.scalars().all()
        user_sub = user.subscription if user else None
        user_promos = user.promo_codes if user else []

        trial_history_result = await session.execute(
            select(TrialUsageHistory).where(TrialUsageHistory.user_id == user_id)
        )
        trial_history = trial_history_result.scalars().all()

    eligible_plans = []
    for plan in all_trial_plans:
        usage_record = next((h for h in trial_history if h.plan_id == plan.id), None)

        if not usage_record:
            planless_usage = next((h for h in trial_history if h.plan_id is None), None)
            if not planless_usage:
                eligible_plans.append(plan)
            continue

        cooldown_days = plan.trial_cooldown_days
        if cooldown_days == 0:
            continue

        if now > (usage_record.used_at + timedelta(days=cooldown_days)):
            eligible_plans.append(plan)

    if not eligible_plans:
        await message.answer("К сожалению, в данный момент нет доступных промо-предложений.")
        return

    global_discount_percent = 0
    if user_sub:
        global_discount_percent = user_sub.discount_percent

    text = (
        "<b>⭐️ Специальные предложения!</b>\n\n"
        "«Оформляя пробную подписку, вы соглашаетесь с условиями выбранного тарифа. "
        "Если для него доступно автопродление, по окончании пробного периода подписка "
        "автоматически перейдет на обычный тариф (отмена в любое время)».\n\n"
    )

    if user_sub and user_sub.end_date > now and user_sub.plan_id is not None:
        text += "<b>У вас уже есть активная подписка.</b> Новый пробный тариф добавится к текущему сроку.\n\n"

    text += "Выберите пробный тариф:"

    await message.answer(
        text,
        reply_markup=kb.promo_plan_selection_keyboard(eligible_plans, global_discount_percent, user_promos)
    )


@router.callback_query(F.data == "admin_panel")
async def back_to_admin_panel(callback: CallbackQuery):
    await callback.message.edit_text("Добро пожаловать в админ-панель!", reply_markup=kb.admin_panel_keyboard())


@router.callback_query(F.data == "admin_stats")
async def admin_stats(callback: CallbackQuery):
    async with async_session_maker() as session:
        total_users = await session.scalar(select(func.count(User.id)))
        total_messages = await session.scalar(select(func.count(DBMessage.id)))

        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = today_start - timedelta(days=7)

        new_users_today = await session.scalar(select(func.count(User.id)).where(User.created_at >= today_start))
        new_users_week = await session.scalar(select(func.count(User.id)).where(User.created_at >= week_ago))

        active_users_today = await session.scalar(
            select(func.count(func.distinct(DBMessage.user_id))).where(DBMessage.timestamp >= today_start))
        messages_today = await session.scalar(
            select(func.count(DBMessage.id)).where(DBMessage.timestamp >= today_start))

        active_subscribers = await session.scalar(
            select(func.count(UserSubscription.id)).where(UserSubscription.end_date > datetime.utcnow())
        )

        total_tests_passed = await session.scalar(
            select(func.count(TestSession.user_id)).where(TestSession.is_finished == True)
        )
        tests_passed_today = await session.scalar(
            select(func.count(TestSession.user_id)).where(
                TestSession.is_finished == True,
                TestSession.created_at >= today_start
            )
        )

    text = f"""
📊 **Статистика бота:**

**Общая:**
- Всего клиентов: {total_users}
- Активных подписчиков: {active_subscribers}
- Прошли тест всего: {total_tests_passed}
- Всего сообщений (клиент+бот): {total_messages}

**Активность сегодня:**
- Новых клиентов: {new_users_today}
- Активных клиентов: {active_users_today}
- Прошли тест сегодня: {tests_passed_today}
- Сообщений за сегодня: {messages_today}

**Новые клиенты за неделю:**
- За последние 7 дней: {new_users_week}
"""
    await callback.message.edit_text(text, parse_mode='Markdown', reply_markup=kb.back_to_admin_panel())


@router.callback_query(F.data.startswith("admin_clients_page_") | F.data.startswith("admin_export_page_"))
async def admin_clients_list(callback: CallbackQuery, state: FSMContext):
    export_mode = callback.data.startswith("admin_export_page_") or (
                await state.get_state() == AdminStates.selecting_for_export)

    try:
        page = int(callback.data.split("_")[-1])
    except (ValueError, IndexError):
        page = 0

    data = await state.get_data()
    search_query = data.get("client_search_query")
    selected_ids = data.get("selected_export_ids", [])

    PAGE_SIZE = 10

    async with async_session_maker() as session:
        base_query = select(User).outerjoin(DBMessage, User.id == DBMessage.user_id)

        if search_query:
            search_filter = or_(
                User.first_name.ilike(f"%{search_query}%"),
                User.name.ilike(f"%{search_query}%"),
                User.username.ilike(f"%{search_query}%"),
                cast(User.id, String).ilike(f"%{search_query}%")
            )
            base_query = base_query.where(search_filter)

        count_stmt = select(func.count(func.distinct(User.id))).select_from(User).outerjoin(DBMessage,
                                                                                            User.id == DBMessage.user_id)
        if search_query:
            count_stmt = count_stmt.where(search_filter)

        total_users = await session.scalar(count_stmt)

        if total_users == 0:
            total_pages = 1
            clients = []
        else:
            total_pages = math.ceil(total_users / PAGE_SIZE)
            page = max(0, min(page, total_pages - 1))

            stmt = base_query.group_by(User.id).order_by(
                func.max(DBMessage.timestamp).desc().nulls_last(),
                User.created_at.desc()
            ).offset(page * PAGE_SIZE).limit(PAGE_SIZE)

            result = await session.execute(stmt)
            clients = result.scalars().all()

    header = f"👥 Список клиентов (Страница {page + 1}/{total_pages})"
    if export_mode:
        header = f"📦 Выберите клиентов для экспорта ({len(selected_ids)} выбрано)"
    elif search_query:
        header = f"🔎 Результаты поиска по запросу: «{html.escape(search_query)}»\n(Стр. {page + 1}/{total_pages})"

    try:
        await callback.message.edit_text(
            header,
            reply_markup=kb.clients_paginator_keyboard(page, total_pages, clients, is_searching=bool(search_query),
                                                       export_mode=export_mode, selected_ids=selected_ids)
        )
    except TelegramBadRequest:
        await callback.answer()


@router.callback_query(F.data == "admin_clients_start_search")
async def start_client_search(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.search_client)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите Имя, Username или ID клиента для поиска:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ Отмена", callback_data="admin_clients_page_0")]
        ])
    )


@router.message(AdminStates.search_client, F.text)
async def process_client_search(message: Message, state: FSMContext, bot: Bot):
    query = message.text.strip()
    await state.update_data(client_search_query=query)

    data = await state.get_data()
    message_id_to_edit = data.get("message_id_to_edit")

    await message.delete()
    await state.set_state(None)


    callback_mock = type('obj', (object,), {
        'message': type('obj', (object,), {
            'chat': message.chat,
            'message_id': message_id_to_edit,
            'edit_text': lambda *args, **kwargs: bot.edit_message_text(chat_id=message.chat.id,
                                                                       message_id=message_id_to_edit, *args, **kwargs),
            'answer': lambda *args, **kwargs: bot.send_message(chat_id=message.chat.id, *args, **kwargs)
        }),
        'data': "admin_clients_page_0",
        'from_user': message.from_user,
        'answer': lambda *args, **kwargs: None
    })()

    await admin_clients_list(callback_mock, state)


@router.callback_query(F.data == "admin_clients_reset_export")
async def admin_clients_reset_export(callback: CallbackQuery, state: FSMContext):
    await state.set_state(None)
    await state.update_data(selected_export_ids=None)
    await admin_clients_list(callback, state)


@router.callback_query(F.data == "admin_clients_reset_search")
async def stop_client_search(callback: CallbackQuery, state: FSMContext):
    await state.update_data(client_search_query=None)
    await admin_clients_list(callback, state)


@router.callback_query(F.data.startswith("view_client_"))
async def view_client_profile(callback: CallbackQuery, state: FSMContext):
    if callback.data.startswith("view_client_"):
        user_id = int(callback.data.split("_")[-1])
    else:
        data = await state.get_data()
        user_id = data.get("viewing_client_id")
        if not user_id:
            await callback.answer("Ошибка: ID клиента потерян. Вернитесь к списку.", show_alert=True)
            return

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
    if not user:
        await callback.answer("Клиент не найден", show_alert=True)
        return

    await state.update_data(viewing_client_id=user_id)

    safe_first_name = html.escape(user.first_name) if user.first_name else "Не указано"
    safe_chosen_name = html.escape(user.name) if user.name else "<i>Не указано</i>"

    if user.username and not user.username.isdigit():
        username_display = f"@{html.escape(user.username)} (<a href='tg://user?id={user.id}'>перейти в профиль</a>)"
    else:
        username_display = f"<a href='tg://user?id={user.id}'>{user.id}</a> (<a href='tg://user?id={user.id}'>перейти в профиль</a>)"

    caller_is_owner = callback.from_user.id in OWNER_IDS
    caller_can_view_history = await check_history_permission(callback.from_user.id)
    birthdate_display = format_birthdate(user.birth_day, user.birth_month, user.birth_year) or "Не указана / недоступна"

    text = (f"<b>Профиль клиента:</b>\n\n"
            f"<b>ID:</b> <code>{user.id}</code>\n"
            f"<b>Имя для общения:</b> {safe_chosen_name}\n"
            f"<b>Имя в Telegram:</b> {safe_first_name}\n"
            f"<b>Username:</b> {username_display}\n"
            f"<b>Дата рождения:</b> {birthdate_display}\n"
            f"<b>Дата регистрации:</b> {user.created_at.strftime('%d-%m-%Y %H:%M')}")

    if user.is_admin and user.id not in OWNER_IDS:
        text += f"\n<b>Доступ к истории:</b> {'✅' if user.can_view_history else '❌'}"

    keyboard = kb.client_profile_keyboard(
        user_id,
        user.is_admin and user.id not in OWNER_IDS,
        user.can_view_history,
        caller_is_owner
    )

    if not caller_can_view_history:
        new_keyboard_buttons = []
        for row in keyboard.inline_keyboard:
            new_row = []
            for button in row:
                if (
                    "client_history_" not in button.callback_data
                    and "download_history_" not in button.callback_data
                    and "admin_delete_client_history_" not in button.callback_data
                ):
                    new_row.append(button)
            if new_row:
                new_keyboard_buttons.append(new_row)
        keyboard = InlineKeyboardMarkup(inline_keyboard=new_keyboard_buttons)

    if not caller_is_owner:
        new_keyboard_buttons = []
        for row in keyboard.inline_keyboard:
            new_row = [button for button in row if "admin_reset_client_" not in button.callback_data]
            if new_row:
                new_keyboard_buttons.append(new_row)
        keyboard = InlineKeyboardMarkup(inline_keyboard=new_keyboard_buttons)

    try:
        await callback.message.edit_text(
            text,
            reply_markup=keyboard,
            disable_web_page_preview=True,
            parse_mode='HTML'
        )
    except TelegramBadRequest as e:
        if "message is not modified" in str(e):
            await callback.answer()
        else:
            logging.error(f"Error editing client profile: {e}")
            await callback.answer("Произошла ошибка при обновлении профиля.", show_alert=True)


@router.callback_query(F.data.startswith("toggle_history_access_"))
async def toggle_history_access(callback: CallbackQuery, state: FSMContext):
    if callback.from_user.id not in OWNER_IDS:
        await callback.answer("Только владелец может делать это.", show_alert=True)
        return

    user_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user or (not user.is_admin and user.id not in OWNER_IDS):
            await callback.answer("Это не администратор.", show_alert=True)
            return

        user.can_view_history = not user.can_view_history
        await session.commit()
        await callback.answer(f"Доступ к истории {'ВЫДАН' if user.can_view_history else 'ЗАБРАН'}.", show_alert=True)

    await view_client_profile(callback, state)


@router.callback_query(F.data == "admin_ai_settings")
async def show_admin_ai_settings(callback: CallbackQuery):
    await admin_ai_settings(callback)


def _build_prompt_block_editor_text(config_key: str, current_value: str) -> str:
    meta = PROMPT_BLOCKS[config_key]
    display_text = current_value or meta["empty_text"]
    if len(display_text) > 3500:
        display_text = display_text[:3500] + "\n\n[...] (Текст обрезан для превью)"

    parts = [
        f"<b>{meta['title']}</b>",
        meta["description"],
        "",
        "<b>Текущее содержимое:</b>",
        f"<code>{html.escape(display_text)}</code>",
    ]
    if meta["placeholders"]:
        parts.extend(["", meta["placeholders"]])
    parts.extend(["", "Отправьте новый текст или загрузите <b>.txt файл</b>."])
    return "\n".join(parts)


async def _open_prompt_block_editor(callback: CallbackQuery, state: FSMContext, config_key: str):
    meta = PROMPT_BLOCKS[config_key]
    await state.set_state(AdminStates.set_prompt_block)
    await state.update_data(
        prompt_block_key=config_key,
        message_id_to_edit=callback.message.message_id,
    )

    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        current_value = getattr(config, meta["db_field"], "") if config else ""

    await callback.message.edit_text(
        _build_prompt_block_editor_text(config_key, current_value),
        reply_markup=kb.prompt_block_keyboard(meta["download_callback"]),
        parse_mode="HTML",
    )


async def _refresh_ai_settings_message(bot: Bot, chat_id: int, message_id: int):
    callback_mock = type('obj', (object,), {
        'message': type('obj', (object,), {
            'chat': type('obj', (object,), {'id': chat_id})(),
            'message_id': message_id,
            'text': "",
            'reply_markup': None,
            'edit_text': lambda *args, **kwargs: bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                *args,
                **kwargs
            ),
        })()
    })()
    await admin_ai_settings(callback_mock)


async def admin_ai_settings(message: Message | CallbackQuery):
    is_callback = isinstance(message, CallbackQuery) or hasattr(message, "message")
    target_message = message.message if is_callback else message

    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            edit_method = target_message.edit_text if is_callback else target_message.answer
            await edit_method("Ошибка: не удалось загрузить конфигурацию ИИ.",
                              reply_markup=kb.back_to_admin_panel())
            return

        provider = config.provider
        if provider == "KIE":
            model_name = getattr(config, "kie_model", "не выбрана")
            model_label = "Активная модель KIE"
        else:
            model_name = getattr(config, f"{provider.lower()}_model", "не выбрана")
            model_label = "Активная модель"

        trans_provider = config.transcription_provider if config.transcription_provider != 'None' else "Выключена"
        vis_provider = config.vision_provider
        vis_model = config.vision_model
        image_gen_provider = getattr(config, 'image_generation_provider', 'Gemini')
        image_gen_model = getattr(config, 'image_generation_model', 'imagen-4.0-generate-001')
        image_edit_provider = getattr(config, 'image_edit_provider', 'Gemini')
        image_edit_model = getattr(config, 'image_edit_model', 'gemini-3-pro-image-preview')
        kie_credit_alert_threshold = getattr(config, 'kie_credit_alert_threshold', 0)
        fb_provider = getattr(config, 'fallback_provider', None)
        fb_model = getattr(config, 'fallback_model', None)
        voice_limit = config.max_voice_duration_sec
        prompt_source = (
            f"Файл: <code>{config.prompt_filename}</code>"
            if config.prompt_mode == 'file' and config.prompt_filename
            else "Текст в БД"
        )
        shared_block_status = "✅ задан" if (getattr(config, 'shared_prompt_block', "") or "").strip() else "❌ пуст"
        service_block_status = "✅ задан" if (getattr(config, 'service_prompt_block', "") or "").strip() else "❌ пуст"

    fb_text = ""
    if fb_provider:
        fb_text = (f"\n🔄 <b>Резервный провайдер (текст):</b>\n"
                   f"▫️ Провайдер: <b>{fb_provider}</b>\n"
                   f"▫️ Модель: <code>{fb_model or 'не задана'}</code>\n")

    text = (f"🤖 <b>Настройки ИИ</b>\n\n"
            f"▫️ Текущий провайдер: <b>{provider}</b>\n"
            f"▫️ {model_label}: <code>{model_name}</code>\n{fb_text}\n"
            f"📝 <b>Промпты:</b>\n"
            f"▫️ Основной промпт: <b>{prompt_source}</b>\n"
            f"▫️ Общий блок: <b>{shared_block_status}</b>\n"
            f"▫️ Служебный блок: <b>{service_block_status}</b>\n\n"
            f"🎙 <b>Аудио:</b>\n"
            f"▫️ Провайдер: <b>{trans_provider}</b>\n"
            f"▫️ Лимит: <b>{voice_limit} сек.</b>\n\n"
            f"🖼 <b>Фото (Vision):</b>\n"
            f"▫️ Провайдер: <b>{vis_provider}</b>\n"
            f"▫️ Модель: <code>{vis_model}</code>\n\n"
            f"🧪 <b>Генерация изображений:</b>\n"
            f"▫️ Провайдер: <b>{image_gen_provider}</b>\n"
            f"▫️ Модель: <code>{image_gen_model}</code>\n\n"
            f"🎨 <b>Редактирование изображений:</b>\n"
            f"▫️ Провайдер: <b>{image_edit_provider}</b>\n"
            f"▫️ Модель: <code>{image_edit_model}</code>\n\n"
            f"💳 <b>KIE кредиты:</b>\n"
            f"▫️ Порог алерта: <b>{kie_credit_alert_threshold}</b>")

    if is_callback:
        try:
            if target_message.text != text or target_message.reply_markup != kb.ai_settings_keyboard(provider):
                await target_message.edit_text(text, reply_markup=kb.ai_settings_keyboard(provider))
        except TelegramBadRequest:
            pass
    else:
        await target_message.answer(text, reply_markup=kb.ai_settings_keyboard(provider))


@router.callback_query(F.data.startswith("ai_provider_"))
async def set_ai_provider(callback: CallbackQuery, bot: Bot):
    provider = callback.data.split("_")[-1]
    async with async_session_maker() as session:
        stmt = update(AIConfig).where(AIConfig.id == 1).values(provider=provider)
        await session.execute(stmt)
        await session.commit()
    await callback.message.edit_text("🤖 Настройки ИИ", reply_markup=kb.ai_settings_keyboard(provider))
    await send_temp_notification(callback.from_user.id, bot, f"✅ Провайдер изменен на {provider}")


@router.callback_query(F.data == "admin_ai_keys")
async def admin_ai_keys_models(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)

    trans_provider = config.transcription_provider if config else 'OpenAI'
    vis_provider = config.vision_provider if config else 'Gemini'
    vis_model = config.vision_model if config else 'gemini-3-flash-preview'
    image_gen_provider = getattr(config, 'image_generation_provider', 'Gemini') if config else 'Gemini'
    image_gen_model = getattr(config, 'image_generation_model', 'imagen-4.0-generate-001') if config else 'imagen-4.0-generate-001'
    image_edit_provider = getattr(config, 'image_edit_provider', 'Gemini') if config else 'Gemini'
    image_edit_model = getattr(config, 'image_edit_model', 'gemini-3-pro-image-preview') if config else 'gemini-3-pro-image-preview'
    kie_credit_alert_threshold = getattr(config, 'kie_credit_alert_threshold', 0) if config else 0
    c_first = config.context_limit_first if config else 2
    c_recent = config.context_limit_recent if config else 10
    temp = getattr(config, 'temperature', 0.7) if config else 0.7
    memory_mode = get_memory_mode(config) if config else MEMORY_MODE_RESET
    fb_provider = getattr(config, 'fallback_provider', None) if config else None
    fb_model = getattr(config, 'fallback_model', None) if config else None

    await callback.message.edit_text(
        "🔑 Настройка ключей, моделей и глубины контекста (памяти)",
        reply_markup=kb.ai_keys_models_keyboard(
            trans_provider,
            c_first,
            c_recent,
            vis_provider,
            vis_model,
            image_gen_provider,
            image_gen_model,
            image_edit_provider,
            image_edit_model,
            kie_credit_alert_threshold,
            temp,
            memory_mode,
            fb_provider,
            fb_model,
        )
    )


@router.callback_query(F.data == "admin_toggle_vision")
async def admin_toggle_vision(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            await callback.answer("Ошибка: Конфигурация ИИ не найдена.", show_alert=True)
            return

        if config.vision_provider == "OpenAI":
            config.vision_provider = "Gemini"
            config.vision_model = "gemini-3-flash-preview"
        elif config.vision_provider == "Gemini":
            config.vision_provider = "KIE"
            config.vision_model = "gemini-2.5-flash"
        elif config.vision_provider == "KIE":
            config.vision_provider = "Claude"
            config.vision_model = getattr(config, "claude_model", "claude-sonnet-4-5-20250929")
        else:
            config.vision_provider = "OpenAI"
            config.vision_model = "gpt-4o"

        new_provider = config.vision_provider
        new_model = config.vision_model
        await session.commit()

    await callback.answer(f"✅ Провайдер: {new_provider}, Модель: {new_model}")
    await admin_ai_keys_models(callback)


@router.callback_query(F.data == "set_context_first")
async def start_set_context_first(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_context_first_limit)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите количество <b>ПЕРВЫХ</b> сообщений диалога, которые бот должен помнить всегда (например, приветствие и знакомство).\n\nОбычно: 2-5.",
        reply_markup=kb.back_to_previous_menu("admin_ai_keys")
    )

@router.callback_query(F.data == "set_context_recent")
async def start_set_context_recent(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_context_recent_limit)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите количество <b>ПОСЛЕДНИХ</b> сообщений диалога, которые бот должен помнить (активная нить разговора).\n\nОбычно: 10-20.",
        reply_markup=kb.back_to_previous_menu("admin_ai_keys")
    )

@router.message(AdminStates.set_context_first_limit, F.text)
async def finish_set_context_first(message: Message, state: FSMContext, bot: Bot):
    try:
        limit = int(message.text)
        if limit < 0: raise ValueError
    except ValueError:
        await message.answer("❌ Введите целое положительное число.")
        return

    async with async_session_maker() as session:
        await session.execute(update(AIConfig).where(AIConfig.id == 1).values(context_limit_first=limit))
        await session.commit()

    data = await state.get_data()
    msg_id = data.get('message_id_to_edit')
    await state.clear()
    await message.delete()

    callback_mock = type('obj', (object,), {
        'message': type('obj', (object,), {
            'chat': message.chat,
            'message_id': msg_id,
            'edit_text': lambda *args, **kwargs: bot.edit_message_text(chat_id=message.chat.id, message_id=msg_id, *args, **kwargs)
        })
    })()
    await admin_ai_keys_models(callback_mock)

@router.message(AdminStates.set_context_recent_limit, F.text)
async def finish_set_context_recent(message: Message, state: FSMContext, bot: Bot):
    try:
        limit = int(message.text)
        if limit < 0: raise ValueError
    except ValueError:
        await message.answer("❌ Введите целое положительное число.")
        return

    async with async_session_maker() as session:
        await session.execute(update(AIConfig).where(AIConfig.id == 1).values(context_limit_recent=limit))
        await session.commit()

    data = await state.get_data()
    msg_id = data.get('message_id_to_edit')
    await state.clear()
    await message.delete()

    callback_mock = type('obj', (object,), {
        'message': type('obj', (object,), {
            'chat': message.chat,
            'message_id': msg_id,
            'edit_text': lambda *args, **kwargs: bot.edit_message_text(chat_id=message.chat.id, message_id=msg_id, *args, **kwargs)
        })
    })()
    await admin_ai_keys_models(callback_mock)


@router.callback_query(F.data == "set_temperature")
async def start_set_temperature(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_temperature)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите температуру генерации ИИ (от <b>0.0</b> до <b>2.0</b>).\n\n"
        "0.0 — строго и предсказуемо\n"
        "0.7 — сбалансированно (по умолчанию)\n"
        "2.0 — максимально творчески",
        reply_markup=kb.back_to_previous_menu("admin_ai_keys")
    )


@router.callback_query(F.data == "set_kie_credit_threshold")
async def start_set_kie_credit_threshold(callback: CallbackQuery, state: FSMContext):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        current_threshold = getattr(config, "kie_credit_alert_threshold", 0) if config else 0

    await state.set_state(AdminStates.set_kie_credit_threshold)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите порог KIE-кредитов, ниже которого всем админам придёт уведомление.\n\n"
        f"Текущее значение: <b>{current_threshold}</b>\n\n"
        "Введите `0`, чтобы отключить уведомления.",
        reply_markup=kb.back_to_previous_menu("admin_ai_keys")
    )


@router.message(AdminStates.set_temperature, F.text)
async def finish_set_temperature(message: Message, state: FSMContext, bot: Bot):
    try:
        temp = float(message.text.strip().replace(',', '.'))
        if not (0.0 <= temp <= 2.0):
            raise ValueError
    except ValueError:
        await message.answer("❌ Введите число от 0.0 до 2.0 (например: 0.7)")
        return

    async with async_session_maker() as session:
        await session.execute(update(AIConfig).where(AIConfig.id == 1).values(temperature=temp))
        await session.commit()

    data = await state.get_data()
    msg_id = data.get('message_id_to_edit')
    await state.clear()
    await message.delete()

    callback_mock = type('obj', (object,), {
        'message': type('obj', (object,), {
            'chat': message.chat,
            'message_id': msg_id,
            'edit_text': lambda *args, **kwargs: bot.edit_message_text(chat_id=message.chat.id, message_id=msg_id, *args, **kwargs)
        })
    })()
    await admin_ai_keys_models(callback_mock)


@router.message(AdminStates.set_kie_credit_threshold, F.text)
async def finish_set_kie_credit_threshold(message: Message, state: FSMContext, bot: Bot):
    try:
        threshold = float(message.text.strip().replace(',', '.'))
        if threshold < 0:
            raise ValueError
    except ValueError:
        await message.answer("❌ Введите число 0 или больше (например: 100)")
        return

    async with async_session_maker() as session:
        await session.execute(
            update(AIConfig).where(AIConfig.id == 1).values(
                kie_credit_alert_threshold=threshold,
                kie_credit_alert_sent=False,
            )
        )
        await session.commit()

    data = await state.get_data()
    msg_id = data.get('message_id_to_edit')
    await state.clear()
    await message.delete()

    callback_mock = type('obj', (object,), {
        'message': type('obj', (object,), {
            'chat': message.chat,
            'message_id': msg_id,
            'edit_text': lambda *args, **kwargs: bot.edit_message_text(chat_id=message.chat.id, message_id=msg_id, *args, **kwargs)
        })
    })()
    await admin_ai_keys_models(callback_mock)


@router.callback_query(F.data == "toggle_preserve_topic_context")
async def toggle_preserve_topic_context(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            await callback.answer("Ошибка: Конфигурация ИИ не найдена.", show_alert=True)
            return
        current_mode = get_memory_mode(config)
        new_value = next_memory_mode(current_mode)
        config.memory_mode = new_value
        config.preserve_topic_context = is_topic_memory_mode(new_value)
        await session.commit()
    await callback.answer(memory_mode_description(new_value), show_alert=False)
    await admin_ai_keys_models(callback)


@router.callback_query(F.data == "admin_toggle_test_btn")
async def admin_toggle_test_btn(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.test_button_enabled = not config.test_button_enabled
        is_enabled = config.test_button_enabled
        await session.commit()

    await callback.answer(f"Кнопка 'Пройти тест' {'включена' if is_enabled else 'выключена'}")
    await callback.message.edit_reply_markup(
        reply_markup=kb.admin_payment_settings_keyboard(config)
    )


@router.callback_query(F.data == "admin_toggle_transcription")
async def admin_toggle_transcription(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            await callback.answer("Ошибка: Конфигурация ИИ не найдена.", show_alert=True)
            return

        if config.transcription_provider == "OpenAI":
            config.transcription_provider = "Gemini"
        elif config.transcription_provider == "Gemini":
            config.transcription_provider = "KIE"
            config.kie_transcription_model = "elevenlabs/speech-to-text"
        elif config.transcription_provider == "KIE":
            config.transcription_provider = "None"
        else:
            config.transcription_provider = "OpenAI"

        new_provider = config.transcription_provider
        await session.commit()

    display_provider = "Выкл" if new_provider == "None" else new_provider
    await callback.answer(f"✅ Провайдер транскрибации изменен на {display_provider}")

    await admin_ai_keys_models(callback)


@router.callback_query(F.data == "admin_toggle_image_generation")
async def admin_toggle_image_generation(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            await callback.answer("Ошибка: Конфигурация ИИ не найдена.", show_alert=True)
            return

        if config.image_generation_provider == "OpenAI":
            config.image_generation_provider = "Gemini"
            config.image_generation_model = "imagen-4.0-generate-001"
        elif config.image_generation_provider == "Gemini":
            config.image_generation_provider = "KIE"
            config.image_generation_model = "seedream/4.5-text-to-image"
        else:
            config.image_generation_provider = "OpenAI"
            config.image_generation_model = "gpt-image-1.5"
        await session.commit()

    await callback.answer(f"✅ Генерация изображений: {config.image_generation_provider}")
    await admin_ai_keys_models(callback)


@router.callback_query(F.data == "admin_toggle_image_edit")
async def admin_toggle_image_edit(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            await callback.answer("Ошибка: Конфигурация ИИ не найдена.", show_alert=True)
            return

        if config.image_edit_provider == "Gemini":
            config.image_edit_provider = "KIE"
            config.image_edit_model = "seedream/4.5-edit"
        else:
            config.image_edit_provider = "Gemini"
            config.image_edit_model = "gemini-3-pro-image-preview"
        await session.commit()

    await callback.answer(f"✅ Редактирование изображений: {config.image_edit_provider}")
    await admin_ai_keys_models(callback)


_FALLBACK_CYCLE = [None, "Deepseek", "Claude", "Gemini", "KIE", "OpenAI"]
_FALLBACK_DEFAULT_MODELS = {
    "Deepseek": "deepseek-chat",
    "Claude": "claude-sonnet-4-5-20250929",
    "Gemini": "gemini-2.0-flash",
    "KIE": "gemini-3-flash",
    "OpenAI": "gpt-4o",
}


@router.callback_query(F.data == "admin_toggle_fallback")
async def admin_toggle_fallback(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        if not config:
            await callback.answer("Ошибка: Конфигурация ИИ не найдена.", show_alert=True)
            return

        current = getattr(config, 'fallback_provider', None)
        try:
            idx = _FALLBACK_CYCLE.index(current)
        except ValueError:
            idx = 0
        next_val = _FALLBACK_CYCLE[(idx + 1) % len(_FALLBACK_CYCLE)]
        config.fallback_provider = next_val
        if next_val:
            config.fallback_model = _FALLBACK_DEFAULT_MODELS.get(next_val, "")
        else:
            config.fallback_model = None
        await session.commit()

    label = next_val if next_val else "выкл"
    await callback.answer(f"✅ Резервный провайдер: {label}")
    await admin_ai_keys_models(callback)


@router.callback_query(F.data == "admin_change_fallback_model")
async def admin_change_fallback_model_list(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        provider = getattr(config, 'fallback_provider', None)

    if not provider:
        await callback.answer("Сначала выберите резервный провайдер.", show_alert=True)
        return

    builder = InlineKeyboardBuilder()
    model_map = {
        "Deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "Claude": ["claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805", "claude-haiku-4-5-20251001"],
        "Gemini": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06"],
        "KIE": ["gemini-3-flash", "gemini-2.5-flash"],
        "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
    }
    models = model_map.get(provider, [])
    for m in models:
        builder.button(text=m, callback_data=f"save_fallback_model_{m}")
    builder.button(text="⬅️ Назад", callback_data="admin_ai_keys")
    builder.adjust(1)

    await callback.message.edit_text(f"Выберите резервную модель для {provider}:", reply_markup=builder.as_markup())


@router.callback_query(F.data.startswith("save_fallback_model_"))
async def save_fallback_model(callback: CallbackQuery):
    model_name = callback.data.replace("save_fallback_model_", "")
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        config.fallback_model = model_name
        await session.commit()

    await callback.answer(f"✅ Резервная модель: {model_name}")
    await admin_ai_keys_models(callback)


@router.message(AdminStates.set_api_key)
@router.message(AdminStates.set_model)
async def process_api_input(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    provider = data['provider']
    current_state = await state.get_state()
    is_key = current_state == AdminStates.set_api_key.state
    column_name = f"{provider.lower()}_api_key" if is_key else f"{provider.lower()}_model"
    value = message.text.strip() if message.text else None

    async with async_session_maker() as session:
        stmt = update(AIConfig).where(AIConfig.id == 1).values({column_name: value})
        await session.execute(stmt)
        await session.commit()

    await state.clear()

    await admin_ai_settings(message)

    notification = "ключ API" if is_key else "модель"
    await send_temp_notification(message.from_user.id, bot, f"✅ Новый {notification} для {provider} сохранен.")
    await message.delete()


@router.callback_query(F.data == "admin_edit_shared_prompt_block")
async def edit_shared_prompt_block(callback: CallbackQuery, state: FSMContext):
    await _open_prompt_block_editor(callback, state, "shared")


@router.callback_query(F.data == "admin_edit_service_prompt_block")
async def edit_service_prompt_block(callback: CallbackQuery, state: FSMContext):
    await _open_prompt_block_editor(callback, state, "service")


@router.callback_query(F.data.in_(["download_shared_prompt_block", "download_service_prompt_block"]))
async def download_prompt_block(callback: CallbackQuery):
    config_key = "shared" if callback.data == "download_shared_prompt_block" else "service"
    meta = PROMPT_BLOCKS[config_key]

    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        prompt_text = getattr(config, meta["db_field"], "") if config else ""

    if not prompt_text and config_key == "service":
        prompt_text = DEFAULT_SERVICE_PROMPT_TEMPLATE

    file_to_send = BufferedInputFile(
        (prompt_text or "").encode('utf-8'),
        filename=meta["filename"]
    )
    await callback.message.answer_document(
        file_to_send,
        caption=f"📄 {meta['title']}."
    )
    await callback.answer()


@router.callback_query(F.data == "admin_edit_system_prompt")
async def edit_system_prompt(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_system_prompt)
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)

    prompt_too_long = False
    if config.prompt_mode == 'file':
        prompt_too_long = True
        prompt_info = f"Текущий промпт загружен из файла: `{config.prompt_filename}`"
        message_text_content = "Отправьте новый текст или .txt файл, чтобы его заменить."
        message_text = f"{prompt_info}\n\n{message_text_content}"
    else:
        prompt_info = "Текущий промпт сохранен как текст."
        prompt_text = config.system_prompt or ""
        display_text = prompt_text
        if len(prompt_text) > 3500:
            prompt_too_long = True
            display_text = prompt_text[:3500] + "\n\n[...] (Промпт слишком длинный для полного отображения)"
        message_text = f"{prompt_info}\n\n`{display_text}`\n\nОтправьте новый текст или .txt файл, чтобы его заменить."

    await callback.message.edit_text(
        message_text,
        parse_mode='Markdown',
        reply_markup=kb.system_prompt_keyboard(prompt_too_long=prompt_too_long)
    )


@router.callback_query(F.data == "download_system_prompt")
async def download_system_prompt(callback: CallbackQuery):
    prompt_text = "Системный промпт не установлен."
    filename = "system_prompt.txt"

    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)

        if config.prompt_mode == 'file' and config.prompt_filename:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(script_dir, "system_prompts", config.prompt_filename)

                with open(file_path, 'r', encoding='utf-8') as f:
                    prompt_text = f.read()
                filename = config.prompt_filename
            except FileNotFoundError:
                prompt_text = f"ОШИБКА: Файл промпта '{config.prompt_filename}' не найден!"
            except Exception as e:
                prompt_text = f"ОШИБКА: Не удалось прочитать файл. {e}"
        else:
            prompt_text = config.system_prompt or "Системный промпт (текст) не установлен."

    file_bytes = prompt_text.encode('utf-8')
    file_to_send = BufferedInputFile(
        file_bytes,
        filename=filename
    )

    await callback.message.answer_document(
        file_to_send,
        caption="📄 Текущий системный промпт."
    )
    await callback.answer()


@router.message(AdminStates.set_system_prompt, (F.text | F.document))
async def process_system_prompt(message: Message, state: FSMContext, bot: Bot):
    async with async_session_maker() as session:
        if message.text:
            stmt = update(AIConfig).where(AIConfig.id == 1).values(
                system_prompt=message.text,
                prompt_mode='text',
                prompt_filename=None
            )
            await session.execute(stmt)
            await session.commit()
            notification_text = "✅ Системный промпт обновлен текстом."


        elif message.document:

            document = message.document

            if not document.file_name.lower().endswith('.txt'):
                await message.answer("❌ Ошибка: Пожалуйста, загрузите файл в формате .txt")

                return


            script_dir = os.path.dirname(os.path.abspath(__file__))

            prompts_dir = os.path.join(script_dir, "system_prompts")


            os.makedirs(prompts_dir, exist_ok=True)

            file_path = os.path.join(prompts_dir, document.file_name)

            await bot.download(document, destination=file_path)

            stmt = update(AIConfig).where(AIConfig.id == 1).values(
                prompt_mode='file',
                prompt_filename=document.file_name
            )
            await session.execute(stmt)
            await session.commit()
            notification_text = f"✅ Системный промпт теперь загружается из файла: `{document.file_name}`."

        config = await session.get(AIConfig, 1)

    await state.clear()
    await message.answer("Настройки ИИ", reply_markup=kb.ai_settings_keyboard(config.provider))
    await send_temp_notification(message.from_user.id, bot, notification_text)


@router.message(AdminStates.set_prompt_block, (F.text | F.document))
async def process_prompt_block(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    config_key = data.get("prompt_block_key")
    meta = PROMPT_BLOCKS.get(config_key or "")
    message_id_to_edit = data.get("message_id_to_edit")

    if not meta:
        await state.clear()
        await message.answer("❌ Ошибка: не удалось определить редактируемый блок.")
        return

    new_value = None
    if message.text is not None:
        new_value = message.text.strip()
    elif message.document:
        if not message.document.file_name.lower().endswith('.txt'):
            temp = await message.answer("❌ Ошибка: Пожалуйста, загрузите файл в формате .txt")
            await asyncio.sleep(4)
            await temp.delete()
            return

        try:
            file_info = await bot.get_file(message.document.file_id)
            file_bytes = await bot.download_file(file_info.file_path)
            new_value = file_bytes.read().decode('utf-8').strip()
        except Exception as e:
            temp = await message.answer(f"❌ Ошибка чтения файла: {e}")
            await asyncio.sleep(4)
            await temp.delete()
            return

    async with async_session_maker() as session:
        await session.execute(
            update(AIConfig).where(AIConfig.id == 1).values({meta["db_field"]: new_value})
        )
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id_to_edit:
        await _refresh_ai_settings_message(bot, message.chat.id, message_id_to_edit)

    temp = await message.answer(f"✅ {meta['title']} обновлен.")
    await asyncio.sleep(3)
    await temp.delete()


async def _display_kb_page(message: Message, page: int = 0, resend: bool = False):
    async with async_session_maker() as session:
        total_files_query = select(func.count(KnowledgeBase.id))
        total_files = await session.scalar(total_files_query)

        if total_files == 0:
            text_to_send = "📚 База знаний пуста."
            reply_markup_to_send = kb.knowledge_base_paginator_keyboard(0, 1, [])
        else:
            total_pages = math.ceil(total_files / KB_PAGE_SIZE)
            page = max(0, min(page, total_pages - 1))
            files_query = select(KnowledgeBase).order_by(
                KnowledgeBase.uploaded_at.desc()
            ).offset(page * KB_PAGE_SIZE).limit(KB_PAGE_SIZE)
            result = await session.execute(files_query)
            files = result.scalars().all()
            text_to_send = f"📚 Управление базой знаний (Страница {page + 1}/{total_pages})\n\n✅ = Файл используется в общем диалоге (без темы).\n⭕️ = Только внутри конкретных тем."
            reply_markup_to_send = kb.knowledge_base_paginator_keyboard(page, total_pages, files)

    if resend:
        try:
            await message.delete()
        except TelegramBadRequest:
            pass
        await message.answer(text=text_to_send, reply_markup=reply_markup_to_send)
    else:
        try:
            await message.edit_text(
                text=text_to_send,
                reply_markup=reply_markup_to_send
            )
        except TelegramBadRequest:
            pass


@router.callback_query(F.data.startswith("admin_kb_page_"))
async def admin_knowledge_base_paginated(callback: CallbackQuery):
    page = int(callback.data.split("_")[-1])
    await _display_kb_page(callback.message, page)
    await callback.answer()


@router.callback_query(F.data == "add_kb_file")
async def add_kb_file(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.upload_kb_file)
    await callback.message.edit_text(
        "Вы вошли в режим добавления файлов в Базу Знаний.\n\n"
        "Отправляйте файлы (txt, md, pdf, docx, xlsx) по одному. "
        "Когда закончите, нажмите кнопку ниже.",
        reply_markup=kb.finish_upload_keyboard()
    )


@router.message(AdminStates.upload_kb_file, F.document)
async def process_kb_file(message: Message, state: FSMContext, bot: Bot):
    document = message.document
    async with async_session_maker() as session:
        new_job = IndexingQueue(
            file_id=document.file_id,
            filename=document.file_name,
            uploader_id=message.from_user.id,
            status='pending'
        )
        session.add(new_job)
        await session.commit()
    await message.answer(f"✅ Файл `{document.file_name}` добавлен в очередь на обработку.")


@router.callback_query(F.data == "admin_balance_refilled")
async def admin_balance_refilled(callback: CallbackQuery):
    async with async_session_maker() as session:
        stmt = update(IndexingQueue).where(IndexingQueue.status == 'paused_balance').values(status='pending')
        await session.execute(stmt)
        await session.commit()
    await callback.message.edit_text("✅ Отлично! Возобновляю обработку файлов из очереди.")


@router.callback_query(F.data == "finish_kb_upload", AdminStates.upload_kb_file)
async def finish_kb_upload(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.answer()
    await _display_kb_page(callback.message, 0, resend=True)


@router.callback_query(F.data == "admin_content")
async def admin_content(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        "✏️ Управление контентом\n\nВыберите раздел для редактирования.",
        reply_markup=await kb.content_management_keyboard()
    )


async def get_content_from_db(key: str) -> dict:
    async with async_session_maker() as session:
        content_obj = await session.get(Content, key, options=[selectinload(Content.media)])
        if content_obj:
            media_list = [
                {'type': media.file_type, 'file_id': media.file_id}
                for media in content_obj.media
            ]
            return {"text": content_obj.text_content, "media": media_list, "is_visible": content_obj.is_visible}
        return {"text": "Контент не найден.", "media": [], "is_visible": False}


@router.message(DynamicButtonFilter(), StateFilter(None))
async def handle_info_buttons(message: Message):
    async with async_session_maker() as session:
        stmt = select(Content).where(Content.button_title == message.text).limit(1)
        content_obj = await session.scalar(stmt)
        content_key = content_obj.key if content_obj else None
    if not content_key:
        return
    content = await get_content_from_db(content_key)
    text = content.get('text')
    media = content.get('media', [])
    content_order = getattr(content_obj, 'content_order', 'media_top')
    html_text = text if text else ""
    async def send_text_part():
        if html_text:
            text_chunks = split_html_text(html_text, 4000)
            for chunk in text_chunks:
                await _safe_send_html(
                    lambda text, pm: message.answer(text, parse_mode=pm),
                    chunk,
                )
                await asyncio.sleep(0.2)
    async def send_media_part():
        if media:
            if 1 < len(media) <= 10:
                media_group = []
                for item in media:
                    media_to_add = InputMediaPhoto(media=item['file_id']) if item['type'] == 'photo' else InputMediaVideo(
                        media=item['file_id'])
                    media_group.append(media_to_add)
                if media_group:
                    await message.answer_media_group(media_group)
            elif len(media) == 1:
                item = media[0]
                if item['type'] == 'photo':
                    await message.answer_photo(item['file_id'])
                elif item['type'] == 'video':
                    await message.answer_video(item['file_id'])
            elif len(media) > 10:
                for i in range(0, len(media), 10):
                    chunk = media[i:i + 10]
                    media_group = []
                    for item in chunk:
                        media_to_add = InputMediaPhoto(media=item['file_id']) if item['type'] == 'photo' else InputMediaVideo(
                            media=item['file_id'])
                        media_group.append(media_to_add)
                    if media_group:
                        await message.answer_media_group(media_group)
                        await asyncio.sleep(0.5)
    if html_text and media and len(media) == 1 and len(html_text) <= 1024:
        item = media[0]
        if item['type'] == 'photo':
            await message.answer_photo(item['file_id'], caption=html_text, parse_mode="HTML")
        elif item['type'] == 'video':
            await message.answer_video(item['file_id'], caption=html_text, parse_mode="HTML")
        return
    if content_order == 'media_top':
        await send_media_part()
        await asyncio.sleep(0.2)
        await send_text_part()
    else:
        await send_text_part()
        await asyncio.sleep(0.2)
        await send_media_part()


@router.callback_query(F.data.startswith("view_models_"))
async def view_models_by_provider(callback: CallbackQuery):
    provider = callback.data.split("_")[-1]
    provider_models = MODELS_INFO.get(provider)

    if not provider_models:
        await callback.answer("Модели для этого провайдера не найдены.", show_alert=True)
        return

    text = f"Выберите модель для <b>{provider}</b>:\n\n"
    for model in provider_models.values():
        if isinstance(model, dict):
            text += f"▪️ <b>{model['name']}</b>: {model['desc']}\n"
    text += f"\n<b>Прайсинг (официальный):</b>\n{provider_models['pricing']}\n\n<i>Цены в рублях являются примерными.</i>"

    await callback.message.edit_text(
        text,
        reply_markup=kb.model_selection_keyboard(provider, {k: v for k, v in provider_models.items() if k != 'pricing'})
    )


@router.callback_query(F.data.startswith("set_key_") | F.data.startswith("set_model_"))
async def process_selection(callback: CallbackQuery, state: FSMContext, bot: Bot):
    parts = callback.data.split("_")
    action_type = parts[0]
    provider = parts[2] if action_type == "set" and len(parts) > 2 else parts[-1]

    if action_type == "set" and parts[1] == "key":
        prompt_text = f"Пожалуйста, отправьте новый API ключ для {provider}."
        await state.set_state(AdminStates.set_api_key)
        await state.update_data(provider=provider)
        await callback.message.edit_text(prompt_text, reply_markup=kb.back_to_previous_menu("admin_ai_keys"))
        return

    if action_type == "set" and parts[1] == "model":
        model_key = "_".join(parts[3:])
        column_name = f"{provider.lower()}_model"
        async with async_session_maker() as session:
            stmt = update(AIConfig).where(AIConfig.id == 1).values({column_name: model_key})
            await session.execute(stmt)
            await session.commit()
        await callback.answer(f"✅ Модель для {provider} установлена на {MODELS_INFO[provider][model_key]['name']}", show_alert=True)
        await admin_ai_keys_models(callback)


async def get_content_display(state: FSMContext):
    data = await state.get_data()
    content_key = data.get('content_key')

    if not content_key:
        return None, None

    text = data.get('text_content', '')
    media_files = data.get('media_files', [])
    content_order = data.get('content_order', 'media_top')

    async with async_session_maker() as session:
        content_obj = await session.get(Content, content_key)
        button_title = content_obj.button_title if content_obj else content_key
        is_visible = content_obj.is_visible if content_obj else True

        btn_text = content_obj.action_btn_text if content_obj else None
        btn_payload = content_obj.action_btn_payload if content_obj else None

    display_name = button_title if button_title else content_key

    content_map = {
        "start_message": "Приветствие (/start)",
        "about_author": "Об авторе",
        "about_method": "О методе",
        "instruction": "Инструкция",
        "disclaimer": "Дисклеймер (Предупреждение)"
    }
    if content_key in content_map:
        display_name = content_map[content_key]

    text_display = "<i>Текст не задан.</i>"
    if text:
        truncated_text = text
        if len(text) > 3500:
            truncated_text = text[:3500] + "\n\n[...] (Текст слишком длинный для полного отображения)"
        text_display = f"<pre><code>{html.escape(truncated_text)}</code></pre>"

    media_display = "<i>Медиафайлы не добавлены.</i>"
    if media_files:
        media_display_lines = []
        for i, media in enumerate(media_files):
            file_type_emoji = "🖼️ Фото" if media['type'] == 'photo' else "📹 Видео"
            media_display_lines.append(f"  {i + 1}. {file_type_emoji}")
        media_display = "\n".join(media_display_lines)

    order_desc = "Сначала медиа, потом текст" if content_order == 'media_top' else "Сначала текст, потом медиа"
    visibility_status = "✅ <b>Виден пользователям</b>" if is_visible else "❌ <b>Скрыт от пользователей</b>"

    btn_info = ""
    if content_key == "start_message":
        btn_info = f"\n<b><u>Кнопка действия:</u></b>\nНазвание: {btn_text or 'Нет'}\nТекст отправки: {btn_payload or 'Нет'}\n"

    response_text = (
        f"📝 <b>Редактирование: '{html.escape(display_name)}'</b>\n\n"
        f"<b><u>Статус:</u></b> {visibility_status}\n"
        f"<b><u>Порядок вывода:</u></b> {order_desc}\n{btn_info}\n"
        f"<b><u>Текущий текст:</u></b>\n{text_display}\n\n"
        f"<b><u>Текущие медиафайлы:</u></b>\n{media_display}\n\n"
        f"Отправьте новый текст (сохранится форматирование), чтобы изменить его, или медиа, чтобы добавить. "
        f"Используйте кнопки ниже для настроек."
    )

    keyboard = kb.content_editing_keyboard(content_key, media_files, content_order, is_visible)
    return response_text, keyboard


@router.callback_query(F.data.startswith("edit_content_"))
async def start_content_edit(callback: CallbackQuery, state: FSMContext):
    if "_btn_text_" in callback.data or "_btn_payload_" in callback.data:
        parts = callback.data.split("_")

        content_key = "start_message"
        action = "text" if "_btn_text_" in callback.data else "payload"

        await state.update_data(content_key=content_key, message_id_to_edit=callback.message.message_id)

        if action == "text":
            await state.set_state(AdminStates.set_content_btn_text)
            text = "Введите название кнопки действия (на самой кнопке):"
        else:
            await state.set_state(AdminStates.set_content_btn_payload)
            text = "Введите текст, который отправится от имени пользователя при нажатии:"

        await callback.message.edit_text(text, reply_markup=kb.back_to_previous_menu(f"edit_content_{content_key}"))
        return

    content_key = callback.data.replace("edit_content_", "")
    await state.set_state(AdminStates.edit_content)

    current_content = await get_content_from_db(content_key)

    async with async_session_maker() as session:
        obj = await session.get(Content, content_key)
        order = getattr(obj, 'content_order', 'media_top')

    await state.update_data(
        content_key=content_key,
        text_content=current_content.get('text'),
        media_files=current_content.get('media', []),
        content_order=order,
        message_id_to_edit=callback.message.message_id
    )

    text, keyboard = await get_content_display(state)
    await callback.message.edit_text(text, reply_markup=keyboard)


@router.message(AdminStates.edit_content, (F.text | F.photo | F.video | F.document))
async def process_content_update(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    if not data.get('content_key'):
        await state.clear()
        await message.delete()
        await message.answer("Произошла ошибка состояния. Пожалуйста, начните редактирование заново из админ-панели.")
        return

    if message.text:
        text_to_save = message.html_text if message.html_text else message.text
        await state.update_data(text_content=text_to_save)
        await send_temp_notification(message.from_user.id, bot, "✅ Текст обновлен (форматирование сохранено).", delay=3)

    elif message.document:
        document = message.document
        if not document.file_name.lower().endswith('.txt'):
            await send_temp_notification(message.from_user.id, bot,
                                         "❌ Ошибка: Для текста принимаются только .txt файлы.", delay=5)
        else:
            try:
                file_info = await bot.get_file(document.file_id)
                file_bytes = await bot.download_file(file_info.file_path)
                text_content = file_bytes.read().decode('utf-8')
                await state.update_data(text_content=text_content)
                await send_temp_notification(message.from_user.id, bot,
                                             f"✅ Текст из файла `{document.file_name}` загружен.", delay=3)
            except UnicodeDecodeError:
                await send_temp_notification(message.from_user.id, bot,
                                             "❌ Ошибка: Не удалось прочитать файл. Убедитесь, что он в кодировке UTF-8.",
                                             delay=5)

    elif message.photo or message.video:
        file_type = 'photo' if message.photo else 'video'
        file_id = message.photo[-1].file_id if message.photo else message.video.file_id

        if message.caption:
            caption_to_save = message.html_caption if message.html_caption else message.caption
            await state.update_data(text_content=caption_to_save)

        media_files = data.get('media_files', [])
        media_files.append({'type': file_type, 'file_id': file_id})
        await state.update_data(media_files=media_files)
        await send_temp_notification(message.from_user.id, bot, f"✅ Медиафайл добавлен (всего: {len(media_files)}).",
                                     delay=3)

    await message.delete()

    try:
        text, keyboard = await get_content_display(state)
        if not text:
            await message.answer("Ошибка: данные сессии утеряны. Вернитесь в админ-панель.")
            return

        msg_id_to_edit = data.get('message_id_to_edit')
        if msg_id_to_edit:
            await bot.edit_message_text(
                text=text,
                chat_id=message.from_user.id,
                message_id=msg_id_to_edit,
                reply_markup=keyboard
            )
    except TelegramBadRequest:
        pass


@router.callback_query(AdminStates.edit_content, F.data.startswith("delete_media_"))
async def handle_media_delete(callback: CallbackQuery, state: FSMContext):
    text, keyboard = await get_content_display(state)
    if not text:
        await callback.answer("Ошибка: сессия истекла или данные не найдены.", show_alert=True)
        await callback.message.delete()
        return

    parts = callback.data.split('_')
    index_to_delete = int(parts[-1])

    data = await state.get_data()
    media_files = data.get('media_files', [])

    if 0 <= index_to_delete < len(media_files):
        media_files.pop(index_to_delete)
        await state.update_data(media_files=media_files)
        await callback.answer(f"Медиафайл #{index_to_delete + 1} удален.")
    else:
        await callback.answer("Ошибка: неверный индекс файла.", show_alert=True)

    text, keyboard = await get_content_display(state)
    if text:
        await callback.message.edit_text(text, reply_markup=keyboard)


@router.callback_query(AdminStates.edit_content, F.data.startswith("save_content_"))
async def save_content(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    content_key = data['content_key']
    new_text = data.get('text_content')
    new_media = data.get('media_files', [])
    new_order = data.get('content_order', 'media_top')

    async with async_session_maker() as session:
        content_obj = await session.get(Content, content_key, options=[selectinload(Content.media)])
        if not content_obj:
            content_obj = Content(key=content_key)
            session.add(content_obj)

        content_obj.text_content = new_text
        content_obj.content_order = new_order

        content_obj.media.clear()
        await session.flush()
        for media_item in new_media:
            content_obj.media.append(ContentMedia(
                file_type=media_item['type'],
                file_id=media_item['file_id']
            ))
        await session.commit()

    await state.clear()

    if content_key in ['test_intro', 'test_results', 'secret_test_outro']:
        await admin_test_menu(callback)
    else:
        await callback.message.edit_text("✅ Раздел успешно обновлен!",
                                         reply_markup=await kb.content_management_keyboard())


@router.callback_query(F.data.startswith("cancel_content_edit_"), StateFilter(AdminStates.edit_content))
async def cancel_content_edit_handler(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    content_key = callback.data.replace("cancel_content_edit_", "")

    if content_key in ['test_intro', 'test_results', 'secret_test_outro']:
        await admin_test_menu(callback)
    else:
        await admin_content(callback, state)


@router.callback_query(AdminStates.edit_content, F.data.startswith("toggle_order_"))
async def toggle_content_order_handler(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    current_order = data.get('content_order', 'media_top')
    new_order = 'text_top' if current_order == 'media_top' else 'media_top'

    await state.update_data(content_order=new_order)

    text, keyboard = await get_content_display(state)
    if text:
        await callback.message.edit_text(text, reply_markup=keyboard)
    await callback.answer(f"Порядок изменен: {new_order}")


@router.callback_query(F.data.startswith("client_history_"))
async def view_client_history(callback: CallbackQuery):
    if not await check_history_permission(callback.from_user.id):
        await callback.answer("У вас нет прав на просмотр истории.", show_alert=True)
        return

    client_id = int(callback.data.split("_")[-1])
    await view_user_history_page(client_id, 0, callback, for_admin_view=True)


@router.callback_query(F.data.startswith("admin_history_"))
async def admin_history_paginator(callback: CallbackQuery):
    parts = callback.data.split("_")
    client_id = int(parts[2])
    page = int(parts[3])
    await view_user_history_page(client_id, page, callback, for_admin_view=True)


async def view_user_history_page(user_id: int, page: int = 0, original_message: Message | CallbackQuery | None = None,
                                 for_admin_view: bool = False):
    async with async_session_maker() as session:
        target_user = await session.get(User, user_id)
        if not target_user:
            await original_message.answer("Пользователь не найден.", show_alert=True)
            return

        topics_res = await session.execute(select(Topic))
        topic_map = {t.id: t.name for t in topics_res.scalars().all()}

        query = select(DBMessage).where(DBMessage.user_id == user_id)
        if not for_admin_view:
            query = query.where(DBMessage.dialogue_id == target_user.current_dialogue_id)

        full_history_stmt = query.order_by(DBMessage.timestamp)
        result = await session.execute(full_history_stmt)
        messages = result.scalars().all()

        if not messages:
            text = "✅ История сообщений этого клиента пуста." if for_admin_view else "Ваша текущая история пуста. Начните диалог!"
            reply_markup = kb.back_to_client_profile(user_id) if for_admin_view else None
            if isinstance(original_message, Message):
                await original_message.answer(text, reply_markup=reply_markup)
            else:
                await original_message.message.edit_text(text, reply_markup=reply_markup)
            return

        SAFE_LIMIT = 3500
        renderable_items = []
        last_dialogue_id = -1

        for msg in messages:
            if for_admin_view and msg.dialogue_id != last_dialogue_id:
                renderable_items.append({'role': 'system', 'content': f"\n--- <b>Диалог №{msg.dialogue_id}</b> ---\n"})
                last_dialogue_id = msg.dialogue_id

            t_name = topic_map.get(msg.topic_id, "Общий")
            header = f"<b>{'👤 (Клиент)' if msg.role == 'user' else '🤖 (Бот)'}</b> [<i>{msg.timestamp.strftime('%d-%m-%Y %H:%M')}</i>] [<i>{t_name}</i>]:\n"

            content = msg.content or ""
            if msg.role == 'assistant':
                try:
                    processed_content = markdown_to_html(content)
                except Exception:
                    processed_content = html.escape(content)
            else:
                processed_content = html.escape(content)

            if len(header) + len(processed_content) > SAFE_LIMIT:
                content_chunks = split_html_text(processed_content, SAFE_LIMIT - len(header) - 50)
                renderable_items.append({'role': msg.role, 'content': header + content_chunks[0]})
                for chunk in content_chunks[1:]:
                    renderable_items.append({'role': msg.role, 'content': f"<i>(продолжение)</i>\n{chunk}"})
            else:
                renderable_items.append({'role': msg.role, 'content': header + processed_content})

        pages = []
        current_page_content = ""
        for item in renderable_items:
            role, content = item['role'], item['content']
            if role == 'assistant':
                entry = f"<blockquote>{content}</blockquote>\n"
            else:
                entry = f"{content}\n"

            if len(current_page_content) + len(entry) > SAFE_LIMIT:
                pages.append(current_page_content)
                current_page_content = entry
            else:
                current_page_content += entry

        if current_page_content:
            pages.append(current_page_content)

        if not pages:
            pages = ["(Нет содержимого для отображения)"]

        total_pages = len(pages)
        page = max(0, min(page, total_pages - 1))

        header_main = f"📜 <b>История клиента:</b> {html.escape(target_user.first_name)} (<a href='https://t.me/@id{target_user.id}'>{target_user.id}</a>)\n" if for_admin_view else "📜 <b>Ваша история сообщений:</b>\n"
        text_to_show = header_main + pages[page]

        reply_markup = kb.user_history_keyboard(
            page, total_pages,
            for_admin_user_id=user_id if for_admin_view else None
        )

        try:
            if isinstance(original_message, Message):
                await original_message.answer(text_to_show, reply_markup=reply_markup)
            else:
                if original_message.message.text == text_to_show:
                    await original_message.answer("Содержимое страницы не изменилось.", show_alert=True)
                else:
                    await original_message.message.edit_text(text_to_show, reply_markup=reply_markup)
        except TelegramBadRequest as e:
            logging.error(f"History render error: {e}")
            safe_text = remove_markdown(text_to_show)
            if isinstance(original_message, Message):
                await original_message.answer(safe_text, reply_markup=reply_markup, parse_mode=None)
            else:
                try:
                    await original_message.message.edit_text(safe_text, reply_markup=reply_markup, parse_mode=None)
                except Exception:
                    pass


@router.message(F.text == "🗑️ Новый диалог")
async def ask_delete_history(message: Message, state: FSMContext):
    await state.clear()

    async with async_session_maker() as session:
        user = await session.get(User, message.from_user.id, options=[selectinload(User.current_topic)])

        if user and user.current_topic_id:
            topic_name = user.current_topic.name if user.current_topic else "Неизвестная тема"

            await message.answer(
                f"Вы находитесь в диалоге: <b>{html.escape(topic_name)}</b>.\n"
                "При начале нового диалога или переходе в основной память ИИ будет очищена.\n"
                "Выберите подходящее действие.",
                reply_markup=kb.topic_reset_options_keyboard(),
                parse_mode="HTML"
            )
        else:
            await message.answer(
                "При начале нового диалога память ИИ будет полностью очищена. Вы уверены?",
                reply_markup=kb.confirm_delete_history_keyboard()
            )


@router.callback_query(F.data == "delete_history_confirm")
async def process_delete_history(callback: CallbackQuery, state: FSMContext, bot: Bot):
    await state.clear()

    async with async_session_maker() as session:
        user = await session.get(User, callback.from_user.id)
        if user:
            ai_config = await session.get(AIConfig, 1)
            memory_mode = get_memory_mode(ai_config) if ai_config else MEMORY_MODE_RESET
            await _new_dialogue_update_state(session, user, user.current_topic_id or 0, memory_mode)

            await session.execute(delete(TestSession).where(TestSession.user_id == user.id))
            await session.commit()

        stmt = select(Content).where(Content.key == "start_message").options(selectinload(Content.media))
        content_obj = await session.scalar(stmt)

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass

    main_kb = await kb.main_client_keyboard()

    if not content_obj:
        await bot.send_message(callback.from_user.id, "✅ Память очищена.", reply_markup=main_kb)
        return

    text = content_obj.text_content
    media = content_obj.media
    order = content_obj.content_order

    inline_kb = None
    if content_obj.action_btn_text and content_obj.action_btn_payload:
        inline_kb = kb.action_button_keyboard(content_obj.action_btn_text, "start_action")

    markup_to_send = inline_kb if inline_kb else main_kb
    chat_id = callback.from_user.id

    async def send_text_func():
        if text:
            await bot.send_message(chat_id, text, reply_markup=markup_to_send)
            return True
        return False

    async def send_media_func():
        if media:
            if len(media) == 1 and len(text or "") < 1000 and order == 'media_top' and text:
                item = media[0]
                if item.file_type == 'photo':
                    await bot.send_photo(chat_id, item.file_id, caption=text, reply_markup=markup_to_send)
                elif item.file_type == 'video':
                    await bot.send_video(chat_id, item.file_id, caption=text, reply_markup=markup_to_send)
                return True

            if len(media) == 1:
                item = media[0]
                if item.file_type == 'photo':
                    await bot.send_photo(chat_id, item.file_id, reply_markup=markup_to_send)
                elif item.file_type == 'video':
                    await bot.send_video(chat_id, item.file_id, reply_markup=markup_to_send)
                return True

            media_group = []
            for item in media:
                media_to_add = InputMediaPhoto(media=item.file_id) if item.file_type == 'photo' else InputMediaVideo(
                    media=item.file_id)
                media_group.append(media_to_add)
            if media_group:
                await bot.send_media_group(chat_id, media_group)
                return False
        return False

    sent_combined = False
    media_sent_with_kb = False
    text_sent_with_kb = False

    if media and len(media) == 1 and len(text or "") < 1000 and order == 'media_top':
        sent_combined = await send_media_func()

    if not sent_combined:
        if order == 'media_top':
            media_sent_with_kb = await send_media_func()
            await asyncio.sleep(0.2)
            text_sent_with_kb = await send_text_func()
        else:
            text_sent_with_kb = await send_text_func()
            await asyncio.sleep(0.2)
            media_sent_with_kb = await send_media_func()

    keyboard_was_sent = sent_combined or media_sent_with_kb or text_sent_with_kb

    if inline_kb:
        await bot.send_message(chat_id, "Главное меню:", reply_markup=main_kb)
    elif not keyboard_was_sent and not inline_kb:
        await bot.send_message(chat_id, "✅ Память очищена.", reply_markup=main_kb)


@router.callback_query(F.data == "delete_history_cancel")
async def cancel_delete_history(callback: CallbackQuery, state: FSMContext):
    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass
    await callback.message.answer("Ок. Продолжаем текущий диалог.")


@router.message(UserStates.awaiting_name)
async def process_user_name(message: Message, state: FSMContext, bot: Bot):
    user_id = message.from_user.id
    user_name = message.text.strip()

    if len(user_name) > 50 or not user_name:
        await message.answer("Пожалуйста, введите корректное имя.")
        return

    async with async_session_maker() as session:
        stmt = update(User).where(User.id == user_id).values(name=user_name)
        await session.execute(stmt)
        await session.commit()

    await message.answer(f"Приятно познакомиться, {html.escape(user_name)}! Укажи свой пол:", reply_markup=kb.gender_selection_keyboard())
    await state.update_data(is_onboarding=True)
    await state.set_state(UserStates.awaiting_gender)


async def get_random_message_by_topic(topic_id: int) -> str | None:
    if not topic_id:
        return None

    async with async_session_maker() as session:
        stmt = select(RandomMessage.content).where(RandomMessage.topic_id == topic_id).order_by(func.random()).limit(1)
        result = await session.scalar(stmt)
        return result


async def process_user_prompt(message: Message, user_id: int, prompt_text: str, bot: Bot):
    user_name = ""
    user_gender = "unknown"
    dialogue_id = 1
    topic_id = None
    async with async_session_maker() as session:
        user = await session.get(User, user_id, options=[selectinload(User.current_topic)])
        if user:
            user_name = user.name
            user_gender = user.gender
            dialogue_id = user.current_dialogue_id
            topic_id = user.current_topic_id
        session.add(
            DBMessage(user_id=user_id, role='user', content=prompt_text, dialogue_id=dialogue_id, topic_id=topic_id))
        await session.commit()

    thinking_msg = await message.answer("🤖 Думаю...")
    ai_prompt_text = prompt_text
    async with async_session_maker() as session:
        ai_config = await session.get(AIConfig, 1)
    if topic_id:
        random_phrase = await get_random_message_by_topic(topic_id)
        if random_phrase:
            ai_prompt_text = (
                f"{prompt_text}\n\n"
                f"[СИСТЕМНАЯ ИНФОРМАЦИЯ: Для ответа используй следующий контекст или метафору (интегрируй это в ответ органично, если уместно): «{random_phrase}»]"
            )
    try:
        response_text = await get_ai_response(user_id, ai_prompt_text, user_name, user_gender)
        
        clean_text, audios, random_imgs, choices, choices_hidden, show_imgs = await handle_ai_media_content(bot, user_id, response_text)

        async with async_session_maker() as session:
            user = await session.get(User, user_id)
            session.add(DBMessage(user_id=user_id, role='assistant', content=response_text,
                                  dialogue_id=user.current_dialogue_id, topic_id=user.current_topic_id))
            await session.commit()
            topic_id = user.current_topic_id if user else None

        # --- 1. Текст AI (сначала предисловие) ---

        image_prompt, text_part = _extract_ai_directive_payload(clean_text, "GEN_IMG")
        if image_prompt:
            if text_part:
                html_text = markdown_to_html(text_part)
                try:
                    await thinking_msg.edit_text(html_text, parse_mode="HTML")
                except Exception:
                    fixed_html = _fix_unclosed_html_tags(html_text)
                    try:
                        await thinking_msg.edit_text(fixed_html, parse_mode="HTML")
                    except Exception:
                        await thinking_msg.delete()
                        for chunk in split_html_text(html_text):
                            await _safe_send_html(
                                lambda text, pm: message.answer(text, parse_mode=pm),
                                chunk,
                            )
                            await asyncio.sleep(0.3)
            else:
                await thinking_msg.delete()

            if image_prompt:
                upload_task = _start_chat_action_loop(bot, user_id, "upload_photo")
                try:
                    image_data = await ai_integration.generate_image(image_prompt)
                    await bot.send_photo(chat_id=user_id, photo=BufferedInputFile(image_data, filename="gen.png"), caption="✨ Готово!")
                except Exception as e:
                    gen_provider, gen_model = _resolve_ai_provider_model(ai_config, "image_generation")
                    await _report_ai_failure(
                        bot,
                        title="Сбой генерации изображения",
                        user=message.from_user,
                        provider=gen_provider,
                        model=gen_model,
                        stage="generate_image",
                        details=str(e),
                        extra={"prompt_len": len(image_prompt)},
                        exception=e,
                    )
                    await bot.send_message(chat_id=user_id, text="😔 Не удалось сгенерировать изображение.")
                finally:
                    upload_task.cancel()
        else:
            if clean_text:
                html_response = markdown_to_html(clean_text)
                try:
                    await thinking_msg.edit_text(html_response, parse_mode="HTML")
                except Exception:
                    fixed_html = _fix_unclosed_html_tags(html_response)
                    try:
                        await thinking_msg.edit_text(fixed_html, parse_mode="HTML")
                    except Exception:
                        await thinking_msg.delete()
                        chunks = split_html_text(html_response, 4000)
                        for chunk in chunks:
                            await _safe_send_html(
                                lambda text, pm: message.answer(text, parse_mode=pm),
                                chunk,
                            )
                            await asyncio.sleep(0.3)
            else:
                await thinking_msg.delete()

        # --- 2. Медиа (карты, аудио) после текста ---

        async with async_session_maker() as session:
            coll_media_ids = await _get_topic_media_ids(session, topic_id) if topic_id else None
            msg_assigned_decks = await _get_assigned_decks(session, topic_id) if topic_id and coll_media_ids is None else None

            for audio_name in audios:
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, msg_assigned_decks),
                    MediaLibrary.file_name == audio_name.strip(),
                    MediaLibrary.media_type == 'audio',
                ).limit(1)
                media = await session.scalar(stmt)
                if media:
                    await bot.send_audio(chat_id=user_id, audio=media.file_id, caption=media.description, parse_mode='HTML')

            for file_name in show_imgs:
                stmt = select(MediaLibrary).where(
                    MediaLibrary.file_name == file_name.strip(),
                    MediaLibrary.media_type == 'photo',
                ).limit(1)
                media = await session.scalar(stmt)
                if media:
                    await send_photo_or_document(
                        bot,
                        user_id,
                        media.file_id,
                        caption=media.description,
                        parse_mode='HTML',
                    )

            drawn_cards_info = []
            all_random_cards = []
            for match in random_imgs:
                cat = match[0].strip() if isinstance(match, tuple) else match.strip()
                r_count = int(match[1]) if isinstance(match, tuple) and match[1] else 1
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, msg_assigned_decks, cat),
                    MediaLibrary.media_type == 'photo',
                    MediaLibrary.file_name != '_back',
                ).order_by(func.random()).limit(r_count)
                res = await session.execute(stmt)
                r_cards = res.scalars().all()
                all_random_cards.extend(r_cards)
            if all_random_cards:
                if len(all_random_cards) == 1:
                    await send_photo_or_document(
                        bot,
                        user_id,
                        all_random_cards[0].file_id,
                        caption=all_random_cards[0].description,
                        parse_mode='HTML',
                    )
                else:
                    await send_card_album(
                        bot,
                        user_id,
                        [c.file_id for c in all_random_cards],
                        context="message_handler.random_cards",
                    )
                for media in all_random_cards:
                    drawn_cards_info.append(f"{media.file_name}: {media.description or 'без описания'}")

            for match in choices:
                cat = match[0].strip()
                cards_per_round = int(match[1])
                rounds = int(match[2]) if match[2] else 1
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, msg_assigned_decks, cat),
                    MediaLibrary.media_type == 'photo',
                    MediaLibrary.file_name != '_back',
                ).order_by(func.random()).limit(cards_per_round)
                res = await session.execute(stmt)
                cards = res.scalars().all()
                if cards:
                    if rounds > 1:
                        user_spread_state[user_id] = {
                            'category': cat, 'topic_id': topic_id,
                            'rounds_left': rounds - 1, 'cards_per_round': cards_per_round,
                            'hidden': False, 'chosen_card_ids': [], 'selected_file_ids': []
                        }
                    await send_card_album(
                        bot,
                        user_id,
                        [c.file_id for c in cards],
                        context="message_handler.choice_spread",
                    )
                    await bot.send_message(chat_id=user_id, text="Выбери карту, которая тебе откликается:", reply_markup=keyboards.card_selection_keyboard(cat, [c.id for c in cards]))

            for match in choices_hidden:
                cat_stripped = match[0].strip()
                cards_per_round = int(match[1])
                rounds = int(match[2]) if match[2] else 1
                stmt = select(MediaLibrary).where(
                    _media_filter(topic_id, coll_media_ids, msg_assigned_decks, cat_stripped),
                    MediaLibrary.media_type == 'photo',
                    MediaLibrary.file_name != '_back',
                ).order_by(func.random()).limit(cards_per_round)
                res = await session.execute(stmt)
                cards = res.scalars().all()
                if cards:
                    if rounds > 1:
                        user_spread_state[user_id] = {
                            'category': cat_stripped, 'topic_id': topic_id,
                            'rounds_left': rounds - 1, 'cards_per_round': cards_per_round,
                            'hidden': True, 'chosen_card_ids': [], 'selected_file_ids': []
                        }
                    back_stmt = select(MediaLibrary).where(
                        MediaLibrary.category == cat_stripped,
                        MediaLibrary.file_name == '_back',
                    ).limit(1)
                    back_media = await session.scalar(back_stmt)
                    if back_media:
                        await send_card_album(
                            bot,
                            user_id,
                            [back_media.file_id for _ in cards],
                            context="message_handler.hidden_choice_spread",
                        )
                    await bot.send_message(chat_id=user_id, text="Выбери карту, которая тебе откликается:", reply_markup=keyboards.card_selection_keyboard(cat_stripped, [c.id for c in cards]))

            if drawn_cards_info:
                cards_text = "; ".join(drawn_cards_info)
                card_system_msg = f"[СИСТЕМА: Случайно выпала карта: {cards_text}. Дай интерпретацию этой карты.]"
                user_obj = await session.get(User, user_id)
                if user_obj:
                    session.add(DBMessage(user_id=user_id, role='user', content=card_system_msg, dialogue_id=user_obj.current_dialogue_id, topic_id=user_obj.current_topic_id))
                    await session.commit()

        if drawn_cards_info:
            typing_task_interp = None
            try:
                async def _typing_loop():
                    try:
                        while True:
                            await bot.send_chat_action(chat_id=user_id, action="typing")
                            await asyncio.sleep(4.5)
                    except asyncio.CancelledError:
                        pass

                typing_task_interp = asyncio.create_task(_typing_loop())
                cards_text = "; ".join(drawn_cards_info)
                interpretation = await ai_integration.generate_response(
                    user_id, f"[СИСТЕМА: Случайно выпала карта: {cards_text}. Дай интерпретацию этой карты.]"
                )
                typing_task_interp.cancel()
                clean_interpretation = re.sub(r"\[(SEND_AUDIO|RANDOM_IMG|CHOICE_IMG|CHOICE_IMG_HIDDEN|SHOW_IMG|GEN_IMG):.*?\]", "", interpretation).strip()
                if clean_interpretation:
                    html_interp = markdown_to_html(clean_interpretation)
                    for chunk in split_html_text(html_interp):
                        await _safe_send_html(
                            lambda text, pm: bot.send_message(chat_id=user_id, text=text, parse_mode=pm),
                            chunk,
                        )
                    async with async_session_maker() as s2:
                        u2 = await s2.get(User, user_id)
                        s2.add(DBMessage(user_id=user_id, role='assistant', content=interpretation, dialogue_id=u2.current_dialogue_id, topic_id=u2.current_topic_id))
                        await s2.commit()
            except Exception as e:
                if typing_task_interp:
                    typing_task_interp.cancel()
                provider, model = _resolve_ai_provider_model(ai_config, "chat")
                await _report_ai_failure(
                    bot,
                    title="Сбой интерпретации карт",
                    user=message.from_user,
                    provider=provider,
                    model=model,
                    stage="message_card_interpretation",
                    details=str(e),
                    exception=e,
                )

    except InsufficientBalanceError as e:
        provider, model = _resolve_ai_provider_model(ai_config, "chat")
        await _report_ai_failure(
            bot,
            title="Критическая ошибка AI API",
            user=message.from_user,
            provider=provider,
            model=model,
            stage="process_user_prompt_balance",
            details=str(e),
            exception=e,
        )
        await thinking_msg.edit_text("К сожалению, сервис временно недоступен из-за технической проблемы.")
    except AIServiceError as e:
        provider, model = _resolve_ai_provider_model(ai_config, "chat")
        await _report_ai_failure(
            bot,
            title="Сбой AI-сервиса",
            user=message.from_user,
            provider=provider,
            model=model,
            stage="process_user_prompt",
            details=str(e),
            extra={"prompt_len": len(ai_prompt_text)},
            exception=e,
        )
        await thinking_msg.edit_text(
            "Упс... У нас что-то сломалось. Мы уже сообщили нашим создателям. Попробуйте вернуться и повторить через несколько минут."
        )
    except Exception as e:
        provider, model = _resolve_ai_provider_model(ai_config, "chat")
        await _report_ai_failure(
            bot,
            title="Непредвиденная ошибка AI-потока",
            user=message.from_user,
            provider=provider,
            model=model,
            stage="process_user_prompt_unexpected",
            details=str(e),
            extra={"prompt_len": len(ai_prompt_text)},
            exception=e,
        )
        await thinking_msg.edit_text("Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")


@router.callback_query(F.data == "disclaimer_accepted", UserStates.awaiting_disclaimer_acceptance)
async def disclaimer_accepted_handler(callback: CallbackQuery, state: FSMContext, bot: Bot):
    user_id = callback.from_user.id

    async with async_session_maker() as session:
        stmt = update(User).where(User.id == user_id).values(accepted_disclaimer=True)
        await session.execute(stmt)
        await session.commit()

    data = await state.get_data()
    prompt_text = data.get('initial_prompt')

    await state.clear()

    await callback.message.delete()

    if prompt_text:
        await process_user_prompt(callback.message, user_id, prompt_text, bot)
    else:
        await callback.message.answer("Спасибо! Теперь вы можете задать свой вопрос.")


@router.callback_query(F.data == "admin_clients")
async def admin_clients_entry(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await admin_clients_list(callback, state)


@router.callback_query(F.data.startswith("download_history_"))
async def download_client_history(callback: CallbackQuery, state: FSMContext):
    if not await check_history_permission(callback.from_user.id):
        await callback.answer("У вас нет прав на скачивание истории.", show_alert=True)
        return

    client_id = int(callback.data.split("_")[-1])
    await state.update_data(export_single_id=client_id)

    await callback.message.edit_text(
        f"Выберите формат и параметры экспорта для пользователя <code>{client_id}</code>:",
        reply_markup=kb.single_export_options_keyboard(client_id)
    )


@router.callback_query(F.data.startswith("run_single_"))
async def process_single_export(callback: CallbackQuery, bot: Bot):
    parts = callback.data.split("_")
    fmt = parts[2]
    anonymize = parts[3] == "yes"
    user_id = int(parts[4])

    thinking_msg = await callback.message.answer("⏳ Формирую файл...")

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            await thinking_msg.edit_text("❌ Ошибка: Пользователь не найден.")
            return

        topics_res = await session.execute(select(Topic))
        topic_map = {t.id: t.name for t in topics_res.scalars().all()}

        msg_stmt = select(DBMessage).where(DBMessage.user_id == user.id).order_by(DBMessage.timestamp)
        messages = (await session.execute(msg_stmt)).scalars().all()

        if not messages:
            await thinking_msg.edit_text("✅ История сообщений пуста.")
            return

        user_label = "user_1" if anonymize else str(user.id)

        if fmt == "txt":
            header = f"History: {user_label}\n" if anonymize else f"History: {user.name or user.first_name} (ID: {user.id}, @{user.username})\n"
            content_str = header + "=" * 50 + "\n"
            for m in messages:
                t_name = topic_map.get(m.topic_id, "General")
                role = "Client" if m.role == "user" else "Bot"
                text = remove_markdown(m.content) if m.role == 'assistant' else m.content
                content_str += f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}] [{t_name}] {role}: {text}\n\n"
        else:
            history_data = []
            for m in messages:
                history_data.append({
                    "timestamp": m.timestamp.isoformat(),
                    "topic": topic_map.get(m.topic_id, "General"),
                    "role": m.role,
                    "content": m.content
                })
            content_str = json.dumps(history_data, ensure_ascii=False, indent=2)

        file_bytes = content_str.encode("utf-8")

        if len(file_bytes) > 50 * 1024 * 1024:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f"{user_label}.{fmt}", file_bytes)
            zip_buffer.seek(0)
            file_to_send = BufferedInputFile(zip_buffer.read(), filename=f"history_{user_label}.zip")
            caption = "📦 Файл превысил 50МБ и был заархивирован."
        else:
            file_to_send = BufferedInputFile(file_bytes, filename=f"{user_label}.{fmt}")
            caption = f"📋 История пользователя {user_label}"

        await callback.message.answer_document(file_to_send, caption=caption)
        await bot.delete_message(callback.message.chat.id, thinking_msg.message_id)
        await callback.answer()


@router.callback_query(F.data.startswith("delete_kb_"))
async def prompt_delete_kb_file(callback: CallbackQuery):
    file_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        kb_file = await session.get(KnowledgeBase, file_id)
        if not kb_file:
            await callback.answer("Файл не найден.", show_alert=True)
            return

    await callback.message.edit_text(
        f"Вы уверены, что хотите удалить файл `{kb_file.filename}` из Базы Знаний?\n\n"
        "Это действие необратимо и также удалит все связанные с ним данные для AI.",
        reply_markup=kb.confirm_delete_kb_keyboard(file_id)
    )


@router.callback_query(F.data.startswith("confirm_delete_kb_"))
async def confirm_delete_kb_file(callback: CallbackQuery, bot: Bot):
    file_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        kb_file = await session.get(KnowledgeBase, file_id)
        if not kb_file:
            await callback.answer("Файл уже был удален.", show_alert=True)
            await _display_kb_page(callback.message, 0)
            return

        filename = kb_file.filename
        stmt = delete(KnowledgeBase).where(KnowledgeBase.id == file_id)
        await session.execute(stmt)
        await session.commit()

    delete_document_vectors(file_id)

    await send_temp_notification(
        callback.from_user.id, bot, f"✅ Файл `{filename}` и его векторы успешно удалены."
    )
    await _display_kb_page(callback.message, 0, resend=True)
    await callback.answer()


@router.callback_query(F.data.startswith("cancel_state_"))
async def cancel_handler(callback: CallbackQuery, state: FSMContext):
    await state.clear()

    target_menu_callback_data = callback.data.split("_", 2)[2]

    callback_mock = type('obj', (object,), {
        'message': callback.message,
        'data': target_menu_callback_data,
        'from_user': callback.from_user,
        'bot': callback.bot,
        'answer': (lambda *args, **kwargs: callback.answer(*args, **kwargs))
    })()

    if target_menu_callback_data == "admin_ai_settings":
        await admin_ai_settings(callback)
    elif target_menu_callback_data == "admin_ai_keys":
        await admin_ai_keys_models(callback_mock)
    elif target_menu_callback_data == "admin_plans":
        await admin_plans_list(callback_mock)
    elif target_menu_callback_data.startswith("admin_edit_plan_"):
        await admin_edit_plan_menu(callback_mock, state)
    elif target_menu_callback_data == "admin_promocodes":
        await admin_promocodes_list(callback_mock)
    elif target_menu_callback_data.startswith("admin_edit_promo_"):
        await admin_edit_promo_menu(callback_mock)
    elif target_menu_callback_data == "admin_payment_settings":
        await admin_payment_settings_menu(callback_mock)
    elif target_menu_callback_data == "admin_payment_keys_menu":
        await admin_payment_keys_menu(callback_mock)
    elif target_menu_callback_data.startswith("admin_topics_page_"):
        await admin_topics_paginated(callback_mock)
    elif target_menu_callback_data.startswith("edit_topic_"):
        await topic_editor_router(callback_mock, state)
    elif target_menu_callback_data == "admin_manage_admins":
        await admin_manage_admins_menu(callback_mock)
    elif target_menu_callback_data == "admin_manage_buttons":
        await admin_manage_buttons(callback)
    elif target_menu_callback_data.startswith("admin_case_studies_page_"):
        await admin_case_studies_list(callback_mock)
    elif target_menu_callback_data == "admin_test_menu":
        await admin_test_menu(callback)
    elif target_menu_callback_data == "admin_secret_questions":
        await admin_secret_questions_menu(callback_mock)
    elif target_menu_callback_data == "admin_test_links":
        await admin_test_links_menu(callback_mock)
    elif target_menu_callback_data.startswith("edit_content_"):
        await start_content_edit(callback_mock, state)

    await callback.answer()


async def _show_admin_edit_plan_menu(bot: Bot, chat_id: int, message_id: int, plan_id: int):
    async with async_session_maker() as session:
        plan = await session.get(
            SubscriptionPlan,
            plan_id,
            options=[selectinload(SubscriptionPlan.upgrades_to_plan)]
        )

    if not plan:
        await bot.edit_message_text("Тариф не найден.", chat_id=chat_id, message_id=message_id, reply_markup=None)
        return

    duration_unit_text = "дн." if plan.duration_unit == 'days' else "мес."

    trial_status = "❌ Нет"
    if plan.is_trial:
        if plan.upgrades_to_plan:
            trial_status = f"✅ Да (переход на «{plan.upgrades_to_plan.name}»)"
        else:
            trial_status = "✅ Да (переход НЕ НАСТРОЕН!)"

    allow_auto_renewal = getattr(plan, 'allow_auto_renewal', True)
    plan_admin_only = getattr(plan, 'admin_only', False)
    text = (
        f"<b>Редактирование тарифа: «{html.escape(plan.name)}»</b>\n\n"
        f"<b>Описание:</b> {html.escape(plan.description)}\n"
        f"<b>Цена:</b> {plan.price} руб.\n"
        f"<b>Длительность:</b> {plan.duration_value} {duration_unit_text}\n"
        f"<b>Статус:</b> {'🟢 Активен' if plan.is_active else '⚪️ Неактивен'}\n"
        f"<b>Видимость:</b> {'🔒 Только для админов' if plan_admin_only else '🔓 Виден всем'}\n"
        f"<b>Автопродление:</b> {'✅ Разрешено' if allow_auto_renewal else '❌ Запрещено (разовая оплата)'}\n"
        f"<b>Пробный период:</b> {trial_status}"
    )
    if plan.is_trial:
        cooldown_text = "нельзя" if plan.trial_cooldown_days == 0 else f"{plan.trial_cooldown_days} дн."
        text += f"\n<b>Повторное подключение:</b> {cooldown_text}"

    try:
        await bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=kb.admin_edit_plan_keyboard(plan_id, plan.is_active, plan.is_trial, allow_auto_renewal, plan_admin_only)
        )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("admin_edit_plan_"))
async def admin_edit_plan_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    plan_id = int(callback.data.split("_")[-1])
    await _show_admin_edit_plan_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        plan_id=plan_id
    )


@router.callback_query(F.data == "show_subscription_info_from_chat")
async def show_subscription_info_from_chat(callback: CallbackQuery, state: FSMContext, bot: Bot):
    await callback.message.delete()
    await _send_subscription_info(callback.from_user.id, callback.message.chat.id, bot, state)


async def _apply_topic_switch(session, user, topic_key: int, memory_mode: str) -> bool:
    if is_global_memory_mode(memory_mode):
        return True
    if is_topic_memory_mode(memory_mode):
        state_rec = await session.get(UserTopicState, (user.id, topic_key))
        if state_rec:
            user.current_dialogue_id = state_rec.dialogue_id
            return True
        user.current_dialogue_id += 1
        session.add(UserTopicState(user_id=user.id, topic_id=topic_key, dialogue_id=user.current_dialogue_id))
    else:
        user.current_dialogue_id += 1
    return False


def _topic_switch_message(topic_name: str, restored: bool, memory_mode: str) -> str:
    if restored:
        return f"✅ Продолжаем тему: **{topic_name}**."
    if is_global_memory_mode(memory_mode):
        return (
            f"✅ Отлично! Мы переключились на тему: **{topic_name}**.\n\n"
            f"Контекст диалога сохранен. Дальше бот будет использовать промпт текущей темы."
        )
    return (
        f"✅ Отлично! Мы переключились на тему: **{topic_name}**.\n\n"
        f"Память диалога была очищена. Можете задавать свой вопрос."
    )


async def _new_dialogue_update_state(session, user, topic_key: int, memory_mode: str):
    user.current_dialogue_id += 1
    if is_global_memory_mode(memory_mode):
        return
    if is_topic_memory_mode(memory_mode):
        state_rec = await session.get(UserTopicState, (user.id, topic_key))
        if state_rec:
            state_rec.dialogue_id = user.current_dialogue_id
        else:
            session.add(UserTopicState(user_id=user.id, topic_id=topic_key, dialogue_id=user.current_dialogue_id))


@router.message(F.text == "📚 Темы диалога")
@router.message(TopicsButtonFilter())
async def select_topic_menu(message: Message):
    async with async_session_maker() as session:
        user = await session.get(User, message.from_user.id, options=[selectinload(User.current_topic)])

        current_status = "в <b>Основном диалоге</b>"
        if user and user.current_topic:
            current_status = f"в диалоге: <b>{html.escape(user.current_topic.name)}</b>"

        is_admin_user = message.from_user.id in OWNER_IDS or (user and user.is_admin)
        topic_conditions = [Topic.is_active == True, Topic.show_in_list == True]
        if not is_admin_user:
            topic_conditions.append(Topic.admin_only == False)
        topics_result = await session.execute(
            select(Topic).where(*topic_conditions).order_by(Topic.sort_order.asc(), Topic.id.asc())
        )
        active_topics = topics_result.scalars().all()

        user_topic_id = user.current_topic_id if user else None

    if not active_topics:
        await message.answer("К сожалению, сейчас нет доступных тем для диалога.")
        return

    text = (
        f"Вы находитесь {current_status}.\n"
        "Выберите подходящую тему для общения.\n"
        "Бот будет использовать специальные знания и инструкции для ответов по выбранной теме."
    )

    await message.answer(
        text,
        reply_markup=kb.select_topic_keyboard(active_topics, user_topic_id),
        parse_mode="HTML"
    )


@router.callback_query(F.data.startswith("select_topic_"))
async def process_topic_selection(callback: CallbackQuery, bot: Bot):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        user = await session.get(User, callback.from_user.id)
        restored = False
        already_in_topic = bool(user and user.current_topic_id == topic_id)
        if user and not already_in_topic:
            user.current_topic_id = topic_id
            ai_config = await session.get(AIConfig, 1)
            memory_mode = get_memory_mode(ai_config) if ai_config else MEMORY_MODE_RESET
            restored = await _apply_topic_switch(session, user, topic_id, memory_mode)
            await session.commit()

        topic = await session.get(Topic, topic_id)

    if already_in_topic:
        try:
            await callback.message.delete()
        except TelegramBadRequest:
            pass
        await callback.answer()
        return

    await callback.message.delete()

    if topic.start_message:
        text_to_send = topic.start_message
    else:
        text_to_send = _topic_switch_message(topic.name, restored, memory_mode)

    reply_markup = None
    if topic.start_button_text and topic.start_button_payload:
        reply_markup = kb.action_button_keyboard(topic.start_button_text, "topic_action")

    await bot.send_message(
        callback.from_user.id,
        text_to_send,
        parse_mode="HTML" if topic.start_message else "Markdown",
        reply_markup=reply_markup
    )


@router.callback_query(F.data == "reset_topic")
async def process_topic_reset(callback: CallbackQuery, bot: Bot):
    async with async_session_maker() as session:
        user = await session.get(User, callback.from_user.id)
        if user:
            user.current_topic_id = None
            ai_config = await session.get(AIConfig, 1)
            memory_mode = get_memory_mode(ai_config) if ai_config else MEMORY_MODE_RESET
            await _apply_topic_switch(session, user, 0, memory_mode)
            await session.commit()

        stmt = select(Content).where(Content.key == "start_message").options(selectinload(Content.media))
        content_obj = await session.scalar(stmt)

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass

    main_kb = await kb.main_client_keyboard()
    chat_id = callback.from_user.id

    if not content_obj:
        await bot.send_message(chat_id, "✅ Тема сброшена. Мы вернулись в общий режим диалога.", reply_markup=main_kb)
        return

    text = content_obj.text_content
    media = content_obj.media
    order = content_obj.content_order

    inline_kb = None
    if content_obj.action_btn_text and content_obj.action_btn_payload:
        inline_kb = kb.action_button_keyboard(content_obj.action_btn_text, "start_action")

    markup_to_send = inline_kb if inline_kb else main_kb

    async def send_text_func():
        if text:
            await bot.send_message(chat_id, text, reply_markup=markup_to_send)
            return True
        return False

    async def send_media_func():
        if media:
            if len(media) == 1 and len(text or "") < 1000 and order == 'media_top' and text:
                item = media[0]
                if item.file_type == 'photo':
                    await bot.send_photo(chat_id, item.file_id, caption=text, reply_markup=markup_to_send)
                elif item.file_type == 'video':
                    await bot.send_video(chat_id, item.file_id, caption=text, reply_markup=markup_to_send)
                return True

            if len(media) == 1:
                item = media[0]
                if item.file_type == 'photo':
                    await bot.send_photo(chat_id, item.file_id, reply_markup=markup_to_send)
                elif item.file_type == 'video':
                    await bot.send_video(chat_id, item.file_id, reply_markup=markup_to_send)
                return True

            media_group = []
            for item in media:
                media_to_add = InputMediaPhoto(media=item.file_id) if item.file_type == 'photo' else InputMediaVideo(
                    media=item.file_id)
                media_group.append(media_to_add)
            if media_group:
                await bot.send_media_group(chat_id, media_group)
                return False
        return False

    sent_combined = False
    media_sent_with_kb = False
    text_sent_with_kb = False

    if media and len(media) == 1 and len(text or "") < 1000 and order == 'media_top':
        sent_combined = await send_media_func()

    if not sent_combined:
        if order == 'media_top':
            media_sent_with_kb = await send_media_func()
            await asyncio.sleep(0.2)
            text_sent_with_kb = await send_text_func()
        else:
            text_sent_with_kb = await send_text_func()
            await asyncio.sleep(0.2)
            media_sent_with_kb = await send_media_func()

    keyboard_was_sent = sent_combined or media_sent_with_kb or text_sent_with_kb

    if inline_kb:
        await bot.send_message(chat_id, "Главное меню:", reply_markup=main_kb)
    elif not keyboard_was_sent:
        await bot.send_message(chat_id, "✅ Тема сброшена. Мы вернулись в общий режим диалога.", reply_markup=main_kb)


@router.callback_query(F.data == "topic_select_cancel")
async def process_topic_select_cancel(callback: CallbackQuery):
    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass
    await callback.answer()


async def show_topics_admin_list(callback: CallbackQuery, page: int = 0):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)

        total_topics_q = await session.execute(select(func.count(Topic.id)))
        total_topics = total_topics_q.scalar_one()

        if total_topics == 0:
            await callback.message.edit_text(
                "💬 Управление темами диалогов\n\nВы еще не создали ни одной темы.",
                reply_markup=keyboards.topics_admin_list_keyboard([], 0, 0, config)
            )
            return

        total_pages = math.ceil(total_topics / PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))

        topics_q = await session.execute(
            select(Topic).order_by(Topic.sort_order.asc(), Topic.id.asc()).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
        )
        topics = topics_q.scalars().all()

    await callback.message.edit_text(
        f"💬 Управление темами диалогов (Стр. {page + 1}/{total_pages})",
        reply_markup=keyboards.topics_admin_list_keyboard(topics, page, total_pages, config)
    )


@router.callback_query(F.data.startswith("move_topic_"))
async def admin_move_topic(callback: CallbackQuery):
    parts = callback.data.split("_")
    direction = parts[2]
    topic_id = int(parts[3])
    page = int(parts[4])

    async with async_session_maker() as session:
        stmt = select(Topic).order_by(Topic.sort_order.asc(), Topic.id.asc())
        result = await session.execute(stmt)
        topics = result.scalars().all()

        for idx, t in enumerate(topics):
            t.sort_order = idx

        current_idx = -1
        for idx, t in enumerate(topics):
            if t.id == topic_id:
                current_idx = idx
                break

        if current_idx == -1:
            await callback.answer("Тема не найдена.")
            return

        target_idx = -1
        if direction == "up" and current_idx > 0:
            target_idx = current_idx - 1
        elif direction == "down" and current_idx < len(topics) - 1:
            target_idx = current_idx + 1

        if target_idx != -1:
            topics[current_idx].sort_order, topics[target_idx].sort_order = target_idx, current_idx
            await session.commit()
            await callback.answer("Порядок изменен.")
        else:
            await callback.answer("Дальше перемещать нельзя.")
            return

    await show_topics_admin_list(callback, page)


@router.callback_query(F.data.startswith("admin_topics_page_"))
async def admin_topics_paginated(callback: CallbackQuery):
    page = int(callback.data.split("_")[-1])
    await show_topics_admin_list(callback, page)


@router.callback_query(F.data == "create_topic")
async def admin_create_topic_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_topic_name)
    await state.update_data(topic_id=None, message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text("Введите название для новой темы:",
                                     reply_markup=kb.back_to_previous_menu("admin_topics_page_0"))


async def show_edit_topic_menu(callback: CallbackQuery, topic_id: int):
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id, options=[selectinload(Topic.knowledge_base_files)])
        if not topic:
            await callback.answer("Тема не найдена", show_alert=True)
            return

    kb_files_count = len(topic.knowledge_base_files)
    prompt_status = "✅ Задан" if topic.system_prompt else "❌ Не задан (используется общий)"
    active_status = "🟢 Активна" if topic.is_active else "⚪️ Неактивна"

    text = (
        f"Редактирование темы: <b>{html.escape(topic.name)}</b>\n\n"
        f"<b>ID темы (для рассылок):</b> <code>{topic.id}</code>\n"
        f"<b>Ссылка для перехода:</b> <code>https://t.me/{(await callback.bot.get_me()).username}?start=topic_{topic.id}</code>\n\n"
        f"<b>Статус:</b> {active_status}\n"
        f"<b>Системный промпт:</b> {prompt_status}\n"
        f"<b>Привязано файлов БЗ:</b> {kb_files_count}"
    )

    try:
        await callback.message.edit_text(text, reply_markup=kb.edit_topic_keyboard(topic_id, topic.is_active, admin_only=getattr(topic, 'admin_only', False)))
    except TelegramBadRequest:
        await callback.answer()


@router.callback_query(F.data.startswith("toggle_topic_activity_"), StateFilter('*'))
async def admin_toggle_topic(callback: CallbackQuery):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            topic.is_active = not topic.is_active
            await session.commit()
    await _show_edit_topic_menu(callback.bot, callback.message.chat.id, callback.message.message_id, topic_id)
    await callback.answer()


@router.callback_query(F.data.startswith("toggle_topic_admin_only_"), StateFilter('*'))
async def admin_toggle_topic_admin_only(callback: CallbackQuery):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            topic.admin_only = not getattr(topic, 'admin_only', False)
            await session.commit()
    await _show_edit_topic_menu(callback.bot, callback.message.chat.id, callback.message.message_id, topic_id)
    await callback.answer()


@router.callback_query(F.data.startswith("edit_topic_"), StateFilter('*'))
async def topic_editor_router(callback: CallbackQuery, state: FSMContext):
    await state.clear()

    parts = callback.data.split('_')
    topic_id = int(parts[-1])

    action_key = parts[2]

    if action_key == 'name':
        await state.set_state(AdminStates.set_topic_name)
        await state.update_data(topic_id=topic_id, message_id_to_edit=callback.message.message_id)
        await callback.message.edit_text(
            "Введите новое название для темы:",
            reply_markup=kb.back_to_previous_menu(f"edit_topic_{topic_id}")
        )

    elif action_key == 'prompt':
        async with async_session_maker() as session:
            topic = await session.get(Topic, topic_id)

        await state.set_state(AdminStates.set_topic_prompt)
        await state.update_data(topic_id=topic_id, message_id_to_edit=callback.message.message_id)

        prompt_text = topic.system_prompt or "<i>Собственный промпт для темы не задан. Будет использоваться общий.</i>"

        safe_prompt_text = html.escape(prompt_text) if topic.system_prompt else prompt_text

        if len(safe_prompt_text) > 3500:
            safe_prompt_text = safe_prompt_text[:3500] + "\n\n[...] (Промпт слишком длинный)"

        await callback.message.edit_text(
            f"<b>Текущий промпт:</b>\n{safe_prompt_text}\n\n"
            f"Отправьте новый текст или .txt файл.",
            reply_markup=kb.topic_prompt_keyboard(topic_id)
        )

    elif action_key == 'intro':
        await state.set_state(AdminStates.set_topic_intro_msg)
        await state.update_data(topic_id=topic_id, message_id_to_edit=callback.message.message_id)
        await callback.message.edit_text(
            "Введите текст приветственного сообщения (инструкции) для этой темы.\n"
            "Оно будет показано сразу после переключения.",
            reply_markup=kb.back_to_previous_menu(f"edit_topic_{topic_id}")
        )

    elif action_key == 'btn':
        sub_type = parts[3]

        if sub_type == 'text':
            await state.set_state(AdminStates.set_topic_btn_text)
            await state.update_data(topic_id=topic_id, message_id_to_edit=callback.message.message_id)
            await callback.message.edit_text(
                "Введите название кнопки (то, что написано НА кнопке). Например: 'Начать диалог'.",
                reply_markup=kb.back_to_previous_menu(f"edit_topic_{topic_id}")
            )
        elif sub_type == 'payload':
            await state.set_state(AdminStates.set_topic_btn_payload)
            await state.update_data(topic_id=topic_id, message_id_to_edit=callback.message.message_id)
            await callback.message.edit_text(
                "Введите текст, который отправится от имени пользователя при нажатии. Например: 'Привет!'.",
                reply_markup=kb.back_to_previous_menu(f"edit_topic_{topic_id}")
            )

    else:
        await _show_edit_topic_menu(
            bot=callback.bot,
            chat_id=callback.message.chat.id,
            message_id=callback.message.message_id,
            topic_id=topic_id
        )


@router.message(AdminStates.set_topic_name, F.text)
async def admin_edit_topic_name_process(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    topic_id = data.get('topic_id')
    message_id_to_edit = data.get('message_id_to_edit')
    new_name = message.text.strip()

    await message.delete()
    await state.clear()

    if not new_name:
        temp_msg = await message.answer("Название не может быть пустым. Попробуйте еще раз.")
        asyncio.create_task(delete_message_after_delay(temp_msg, 4))
        return

    async with async_session_maker() as session:
        if topic_id:
            stmt = update(Topic).where(Topic.id == topic_id).values(name=new_name)
            await session.execute(stmt)
            await session.commit()
            action_text = f"✅ Название темы обновлено на «{new_name}»."

            if message_id_to_edit:
                await _update_topic_edit_menu(bot, message.chat.id, message_id_to_edit, topic_id)

            temp_msg = await message.answer(action_text)
            asyncio.create_task(delete_message_after_delay(temp_msg, 3))

        else:
            new_topic = Topic(name=new_name)
            session.add(new_topic)
            await session.commit()
            action_text = f"✅ Тема «{new_name}» создана."

            if message_id_to_edit:
                try:
                    config = await session.get(SubscriptionConfig, 1)
                    total_topics_q = await session.execute(select(func.count(Topic.id)))
                    total_topics = total_topics_q.scalar_one()
                    total_pages = math.ceil(total_topics / PAGE_SIZE) or 1
                    page = total_pages - 1

                    topics_q = await session.execute(
                        select(Topic).order_by(Topic.name).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
                    )
                    topics = topics_q.scalars().all()

                    await bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=message_id_to_edit,
                        text=f"💬 Управление темами диалогов (Стр. {page + 1}/{total_pages})",
                        reply_markup=kb.topics_admin_list_keyboard(topics, page, total_pages, config)
                    )
                except TelegramBadRequest:
                    pass

            temp_msg = await message.answer(action_text)
            asyncio.create_task(delete_message_after_delay(temp_msg, 3))


@router.message(AdminStates.set_topic_prompt, (F.text | F.document))
async def admin_edit_topic_prompt_process(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    topic_id = data['topic_id']
    message_id_to_edit = data.get('message_id_to_edit')
    new_prompt = None
    notification_text = ""

    if message.text:
        new_prompt = message.text if message.text != '-' else None
        notification_text = "✅ Промпт для темы обновлен текстом."

    elif message.document:
        if not message.document.file_name.lower().endswith('.txt'):
            await message.delete()
            temp_msg = await message.answer("❌ Ошибка: Пожалуйста, загрузите файл в формате .txt")
            await delete_message_after_delay(temp_msg, 5)
            return
        try:
            file_info = await bot.get_file(message.document.file_id)
            file_bytes = await bot.download_file(file_info.file_path)
            new_prompt = file_bytes.read().decode('utf-8')
            notification_text = f"✅ Промпт для темы обновлен из файла `{message.document.file_name}`."
        except Exception as e:
            await message.delete()
            temp_msg = await message.answer(f"❌ Ошибка при чтении файла: {e}")
            await delete_message_after_delay(temp_msg, 5)
            return

    async with async_session_maker() as session:
        stmt = update(Topic).where(Topic.id == topic_id).values(system_prompt=new_prompt)
        await session.execute(stmt)
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id_to_edit:
        await _show_edit_topic_menu(
            bot=bot,
            chat_id=message.chat.id,
            message_id=message_id_to_edit,
            topic_id=topic_id
        )

    temp_msg = await message.answer(notification_text)
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.callback_query(F.data.startswith("delete_topic_"), StateFilter('*'))
async def admin_delete_topic_start(callback: CallbackQuery):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
    if not topic:
        await callback.answer("Тема уже удалена.", show_alert=True)
        return
    await callback.message.edit_text(
        f"Вы уверены, что хотите удалить тему «{topic.name}»? Это действие необратимо.",
        reply_markup=kb.confirm_delete_topic_keyboard(topic_id)
    )


@router.callback_query(F.data.startswith("delete_topic_"), StateFilter('*'))
async def admin_delete_topic_start(callback: CallbackQuery):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
    if not topic:
        await callback.answer("Тема уже удалена.", show_alert=True)
        return
    await callback.message.edit_text(
        f"Вы уверены, что хотите удалить тему «{topic.name}»? Это действие необратимо.",
        reply_markup=kb.confirm_delete_topic_keyboard(topic_id)
    )


@router.callback_query(F.data.startswith("confirm_delete_topic_"))
async def admin_delete_topic_confirm(callback: CallbackQuery):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        stmt = delete(Topic).where(Topic.id == topic_id)
        await session.execute(stmt)
        await session.commit()
    await callback.answer("Тема удалена.", show_alert=True)
    await show_topics_admin_list(callback, 0)


async def _show_assign_kb_to_topic_menu(bot: Bot, chat_id: int, message_id: int, topic_id: int, page: int = 0):
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id, options=[selectinload(Topic.knowledge_base_files)])
        assigned_file_ids = {f.id for f in topic.knowledge_base_files}

        total_files = await session.scalar(select(func.count(KnowledgeBase.id)))
        total_pages = math.ceil(total_files / PAGE_SIZE)

        all_files_q = await session.execute(
            select(KnowledgeBase).order_by(KnowledgeBase.filename).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
        )
        all_files = all_files_q.scalars().all()

    await bot.edit_message_text(
        text=f"Привязка файлов к теме «{topic.name}» (Стр. {page + 1}/{total_pages})\n"
             f"Нажмите на файл, чтобы добавить или убрать его из темы.",
        chat_id=chat_id,
        message_id=message_id,
        reply_markup=kb.assign_kb_to_topic_keyboard(topic_id, all_files, assigned_file_ids, page, total_pages)
    )

@router.callback_query(F.data.startswith("assign_kb_topic_"), StateFilter('*'))
async def admin_assign_kb_to_topic(callback: CallbackQuery):
    parts = callback.data.split("_")
    topic_id = int(parts[3])
    page = int(parts[5])
    await _show_assign_kb_to_topic_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        topic_id=topic_id,
        page=page
    )


@router.callback_query(F.data.startswith("kb_topic_"))
async def admin_toggle_kb_for_topic(callback: CallbackQuery):
    parts = callback.data.split("_")
    action = parts[2]
    topic_id = int(parts[3])
    kb_id = int(parts[4])
    page = int(parts[5])

    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id, options=[selectinload(Topic.knowledge_base_files)])
        kb_file = await session.get(KnowledgeBase, kb_id)

        if action == "add" and kb_file not in topic.knowledge_base_files:
            topic.knowledge_base_files.append(kb_file)
        elif action == "remove" and kb_file in topic.knowledge_base_files:
            topic.knowledge_base_files.remove(kb_file)

        await session.commit()

    await callback.answer(f"Файл '{kb_file.filename}' {'добавлен' if action == 'add' else 'удален'}.")

    await _show_assign_kb_to_topic_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        topic_id=topic_id,
        page=page
    )


# ────────── Привязка медиа-колод к топикам ──────────

DECK_PAGE_SIZE = 20


async def _show_assign_deck_to_topic_menu(bot: Bot, chat_id: int, message_id: int, topic_id: int, page: int = 0):
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if not topic:
            return

        # Все уникальные категории (колоды) из media_library
        all_decks_res = await session.execute(
            select(MediaLibrary.category)
            .where(MediaLibrary.category != None, MediaLibrary.category != '')
            .group_by(MediaLibrary.category)
            .order_by(MediaLibrary.category)
        )
        all_deck_names = [r[0] for r in all_decks_res.all()]
        total_decks = len(all_deck_names)
        total_pages = max(1, math.ceil(total_decks / DECK_PAGE_SIZE))
        page_decks = all_deck_names[page * DECK_PAGE_SIZE: (page + 1) * DECK_PAGE_SIZE]

        # Колоды, привязанные к этому топику
        assigned_res = await session.execute(
            select(TopicMediaDeck.deck_name).where(TopicMediaDeck.topic_id == topic_id)
        )
        assigned_decks = {r[0] for r in assigned_res.all()}

        # Подсчёт файлов в каждой колоде
        count_res = await session.execute(
            select(MediaLibrary.category, func.count(MediaLibrary.id))
            .where(MediaLibrary.category.in_(page_decks))
            .group_by(MediaLibrary.category)
        )
        deck_counts = dict(count_res.all())

    deck_info = ", ".join(f"{d} ({deck_counts.get(d, 0)})" for d in page_decks)
    text = (
        f"🃏 Привязка колод к теме: <b>{topic.name}</b>\n"
        f"Нажми на колоду, чтобы добавить или убрать.\n\n"
        f"Колоды: {deck_info}"
    )

    await bot.edit_message_text(
        text=text,
        chat_id=chat_id,
        message_id=message_id,
        reply_markup=kb.assign_decks_to_topic_keyboard(topic_id, page_decks, assigned_decks, page, total_pages),
        parse_mode='HTML'
    )


@router.callback_query(F.data.startswith("assign_deck_topic_"), StateFilter('*'))
async def admin_assign_deck_to_topic(callback: CallbackQuery):
    parts = callback.data.split("_")
    # assign_deck_topic_{topic_id}_page_{page}
    topic_id = int(parts[3])
    page = int(parts[5])
    await _show_assign_deck_to_topic_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        topic_id=topic_id,
        page=page
    )


@router.callback_query(F.data.startswith("deck_topic_"))
async def admin_toggle_deck_for_topic(callback: CallbackQuery):
    parts = callback.data.split("_")
    # deck_topic_{action}_{topic_id}_{deck_name}_{page}
    action = parts[2]
    topic_id = int(parts[3])
    deck_name = parts[4]
    page = int(parts[5])

    async with async_session_maker() as session:
        if action == "add":
            existing = await session.execute(
                select(TopicMediaDeck).where(
                    TopicMediaDeck.topic_id == topic_id,
                    TopicMediaDeck.deck_name == deck_name
                )
            )
            if not existing.scalar_one_or_none():
                session.add(TopicMediaDeck(topic_id=topic_id, deck_name=deck_name))
        elif action == "remove":
            await session.execute(
                delete(TopicMediaDeck).where(
                    TopicMediaDeck.topic_id == topic_id,
                    TopicMediaDeck.deck_name == deck_name
                )
            )
        await session.commit()

    await callback.answer(f"Колода '{deck_name}' {'добавлена' if action == 'add' else 'убрана'}.")

    await _show_assign_deck_to_topic_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        topic_id=topic_id,
        page=page
    )


# ─────────── Медиа-коллекции: CRUD ───────────

COLL_PAGE_SIZE = 8
COLL_FILES_PAGE_SIZE = 10


async def _show_collections_list(bot: Bot, chat_id: int, message_id: int, page: int = 0):
    async with async_session_maker() as session:
        # all collections with file counts
        stmt = (
            select(MediaCollection.id, MediaCollection.name, func.count(media_collection_items.c.media_id))
            .outerjoin(media_collection_items, media_collection_items.c.collection_id == MediaCollection.id)
            .group_by(MediaCollection.id)
            .order_by(MediaCollection.name)
        )
        res = await session.execute(stmt)
        all_colls = [{'id': r[0], 'name': r[1], 'count': r[2]} for r in res.all()]

    total = len(all_colls)
    total_pages = max(1, math.ceil(total / COLL_PAGE_SIZE))
    page_colls = all_colls[page * COLL_PAGE_SIZE: (page + 1) * COLL_PAGE_SIZE]

    text = f"🎨 <b>Медиа-коллекции</b> ({total})\nНажмите на коллекцию для управления."
    await bot.edit_message_text(
        text=text, chat_id=chat_id, message_id=message_id,
        reply_markup=kb.admin_collections_list_keyboard(page_colls, page, total_pages),
        parse_mode='HTML'
    )


@router.callback_query(F.data.startswith("admin_collections_page_"), StateFilter('*'))
async def admin_collections_page(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    page = int(callback.data.split("_")[-1])
    await _show_collections_list(callback.bot, callback.message.chat.id, callback.message.message_id, page)


@router.callback_query(F.data.startswith("admin_coll_view_"), StateFilter('*'))
async def admin_coll_view(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    coll_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        coll = await session.get(MediaCollection, coll_id)
        if not coll:
            await callback.answer("Коллекция не найдена", show_alert=True)
            return
        count_res = await session.execute(
            select(func.count()).select_from(media_collection_items).where(
                media_collection_items.c.collection_id == coll_id
            )
        )
        file_count = count_res.scalar() or 0
        # topics using this collection
        topics_res = await session.execute(
            select(Topic.name).join(
                topic_collection_association,
                topic_collection_association.c.topic_id == Topic.id
            ).where(topic_collection_association.c.collection_id == coll_id)
        )
        topic_names = [r[0] for r in topics_res.all()]

    topics_text = ", ".join(topic_names) if topic_names else "нет"
    text = (
        f"📂 <b>{coll.name}</b>\n\n"
        f"Файлов: {file_count}\n"
        f"Привязана к темам: {topics_text}"
    )
    await callback.message.edit_text(
        text=text, parse_mode='HTML',
        reply_markup=kb.admin_collection_detail_keyboard(coll_id)
    )


@router.callback_query(F.data == "admin_coll_create", StateFilter('*'))
async def admin_coll_create(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminCollectionState.waiting_for_name)
    await callback.message.edit_text(
        "Введите название для новой коллекции:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data="admin_collections_page_0")]
        ])
    )


@router.message(AdminCollectionState.waiting_for_name, F.text)
async def admin_coll_create_name(message: Message, state: FSMContext):
    name = message.text.strip()
    if not name:
        await message.answer("Название не может быть пустым. Попробуйте ещё раз:")
        return
    async with async_session_maker() as session:
        existing = await session.execute(
            select(MediaCollection).where(MediaCollection.name == name)
        )
        if existing.scalar_one_or_none():
            await message.answer(f"Коллекция «{name}» уже существует. Введите другое название:")
            return
        coll = MediaCollection(name=name)
        session.add(coll)
        await session.commit()
        coll_id = coll.id

    await state.clear()
    sent = await message.answer(
        f"✅ Коллекция «{name}» создана.",
        reply_markup=kb.admin_collection_detail_keyboard(coll_id)
    )


@router.callback_query(F.data.startswith("admin_coll_rename_"), StateFilter('*'))
async def admin_coll_rename(callback: CallbackQuery, state: FSMContext):
    coll_id = int(callback.data.split("_")[-1])
    await state.set_state(AdminCollectionState.waiting_for_rename)
    await state.update_data(coll_id=coll_id)
    await callback.message.edit_text(
        "Введите новое название коллекции:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data=f"admin_coll_view_{coll_id}")]
        ])
    )


@router.message(AdminCollectionState.waiting_for_rename, F.text)
async def admin_coll_rename_done(message: Message, state: FSMContext):
    data = await state.get_data()
    coll_id = data['coll_id']
    name = message.text.strip()
    if not name:
        await message.answer("Название не может быть пустым.")
        return
    async with async_session_maker() as session:
        existing = await session.execute(
            select(MediaCollection).where(MediaCollection.name == name, MediaCollection.id != coll_id)
        )
        if existing.scalar_one_or_none():
            await message.answer(f"Коллекция «{name}» уже существует.")
            return
        coll = await session.get(MediaCollection, coll_id)
        if coll:
            coll.name = name
            await session.commit()
    await state.clear()
    await message.answer(
        f"✅ Коллекция переименована в «{name}».",
        reply_markup=kb.admin_collection_detail_keyboard(coll_id)
    )


@router.callback_query(F.data.startswith("admin_coll_delete_"), StateFilter('*'))
async def admin_coll_delete(callback: CallbackQuery, state: FSMContext):
    coll_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        coll = await session.get(MediaCollection, coll_id)
        if coll:
            await session.delete(coll)
            await session.commit()
            await callback.answer(f"Коллекция «{coll.name}» удалена.", show_alert=True)
        else:
            await callback.answer("Коллекция не найдена.", show_alert=True)
    await _show_collections_list(callback.bot, callback.message.chat.id, callback.message.message_id, 0)


@router.callback_query(F.data.startswith("admin_coll_files_"), StateFilter('*'))
async def admin_coll_files(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    parts = callback.data.split("_")
    # admin_coll_files_{coll_id}_{page}
    coll_id = int(parts[3])
    page = int(parts[4])
    async with async_session_maker() as session:
        coll = await session.get(MediaCollection, coll_id)
        if not coll:
            await callback.answer("Коллекция не найдена", show_alert=True)
            return
        # assigned file IDs
        assigned_res = await session.execute(
            select(media_collection_items.c.media_id).where(
                media_collection_items.c.collection_id == coll_id
            )
        )
        assigned_ids = {r[0] for r in assigned_res.all()}
        # all media files
        all_media_res = await session.execute(
            select(MediaLibrary).order_by(MediaLibrary.file_name, MediaLibrary.id)
        )
        all_media = list(all_media_res.scalars().all())

    total = len(all_media)
    total_pages = max(1, math.ceil(total / COLL_FILES_PAGE_SIZE))
    page_media = all_media[page * COLL_FILES_PAGE_SIZE: (page + 1) * COLL_FILES_PAGE_SIZE]

    await callback.message.edit_text(
        f"📎 Файлы коллекции «{coll.name}» (отмечено {len(assigned_ids)} из {total})\n"
        f"Нажмите, чтобы добавить/убрать файл.",
        reply_markup=kb.admin_collection_files_keyboard(coll_id, page_media, assigned_ids, page, total_pages)
    )


@router.callback_query(F.data.startswith("coll_file_"), StateFilter('*'))
async def admin_coll_toggle_file(callback: CallbackQuery):
    parts = callback.data.split("_")
    # coll_file_{action}_{coll_id}_{media_id}_{page}
    action = parts[2]
    coll_id = int(parts[3])
    media_id = int(parts[4])
    page = int(parts[5])

    async with async_session_maker() as session:
        if action == "add":
            try:
                await session.execute(
                    media_collection_items.insert().values(collection_id=coll_id, media_id=media_id)
                )
                await session.commit()
            except Exception:
                await session.rollback()
        elif action == "remove":
            await session.execute(
                media_collection_items.delete().where(
                    media_collection_items.c.collection_id == coll_id,
                    media_collection_items.c.media_id == media_id
                )
            )
            await session.commit()

    await callback.answer("✅")
    # refresh the files page
    async with async_session_maker() as session:
        coll = await session.get(MediaCollection, coll_id)
        if not coll:
            return
        assigned_res = await session.execute(
            select(media_collection_items.c.media_id).where(
                media_collection_items.c.collection_id == coll_id
            )
        )
        assigned_ids = {r[0] for r in assigned_res.all()}
        all_media_res = await session.execute(
            select(MediaLibrary).order_by(MediaLibrary.file_name, MediaLibrary.id)
        )
        all_media = list(all_media_res.scalars().all())

    total = len(all_media)
    total_pages = max(1, math.ceil(total / COLL_FILES_PAGE_SIZE))
    page_media = all_media[page * COLL_FILES_PAGE_SIZE: (page + 1) * COLL_FILES_PAGE_SIZE]

    try:
        await callback.message.edit_text(
            f"📎 Файлы коллекции «{coll.name}» (отмечено {len(assigned_ids)} из {total})\n"
            f"Нажмите, чтобы добавить/убрать файл.",
            reply_markup=kb.admin_collection_files_keyboard(coll_id, page_media, assigned_ids, page, total_pages)
        )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("admin_coll_upload_"), StateFilter('*'))
async def admin_coll_upload(callback: CallbackQuery, state: FSMContext):
    coll_id = int(callback.data.split("_")[-1])
    await state.set_state(AdminCollectionState.waiting_for_upload_file)
    await state.update_data(upload_coll_id=coll_id)
    await callback.message.edit_text(
        "📤 Отправьте медиа-файл (фото, видео, аудио, документ) для добавления в коллекцию.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data=f"admin_coll_view_{coll_id}")]
        ])
    )


@router.message(AdminCollectionState.waiting_for_upload_file)
async def admin_coll_upload_file(message: Message, state: FSMContext):
    data = await state.get_data()
    coll_id = data.get('upload_coll_id')
    if not coll_id:
        await state.clear()
        return

    file_id = None
    media_type = None
    file_name = None

    if message.photo:
        file_id = message.photo[-1].file_id
        media_type = "image"
        file_name = f"photo_{message.photo[-1].file_unique_id}"
    elif message.video:
        file_id = message.video.file_id
        media_type = "video"
        file_name = message.video.file_name or f"video_{message.video.file_unique_id}"
    elif message.audio:
        file_id = message.audio.file_id
        media_type = "audio"
        file_name = message.audio.file_name or f"audio_{message.audio.file_unique_id}"
    elif message.document:
        file_id = message.document.file_id
        mime = message.document.mime_type or ""
        if mime.startswith("image"):
            media_type = "image"
        elif mime.startswith("video"):
            media_type = "video"
        elif mime.startswith("audio"):
            media_type = "audio"
        else:
            media_type = "document"
        file_name = message.document.file_name or f"doc_{message.document.file_unique_id}"
    else:
        await message.answer("Отправьте фото, видео, аудио или документ.")
        return

    async with async_session_maker() as session:
        media = MediaLibrary(
            media_type=media_type,
            file_id=file_id,
            file_name=file_name,
            category="",
        )
        session.add(media)
        await session.flush()
        await session.execute(
            media_collection_items.insert().values(collection_id=coll_id, media_id=media.id)
        )
        await session.commit()

    await state.clear()
    await message.answer(
        f"✅ Файл «{file_name}» добавлен в коллекцию.",
        reply_markup=kb.admin_collection_detail_keyboard(coll_id)
    )


# ─────────── Привязка коллекций к темам ───────────

async def _show_assign_collections_to_topic(bot: Bot, chat_id: int, message_id: int, topic_id: int, page: int = 0):
    PAGE_SIZE = 10
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if not topic:
            return
        # all collections with counts
        stmt = (
            select(MediaCollection.id, MediaCollection.name, func.count(media_collection_items.c.media_id))
            .outerjoin(media_collection_items, media_collection_items.c.collection_id == MediaCollection.id)
            .group_by(MediaCollection.id)
            .order_by(MediaCollection.name)
        )
        res = await session.execute(stmt)
        all_colls = [{'id': r[0], 'name': f"{r[1]} ({r[2]})"} for r in res.all()]
        # assigned
        assigned_res = await session.execute(
            select(topic_collection_association.c.collection_id).where(
                topic_collection_association.c.topic_id == topic_id
            )
        )
        assigned_ids = {r[0] for r in assigned_res.all()}

    total = len(all_colls)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page_colls = all_colls[page * PAGE_SIZE: (page + 1) * PAGE_SIZE]

    text = (
        f"🎨 Привязка коллекций к теме: <b>{topic.name}</b>\n"
        f"Нажмите, чтобы добавить или убрать коллекцию."
    )
    await bot.edit_message_text(
        text=text, chat_id=chat_id, message_id=message_id,
        reply_markup=kb.assign_collections_to_topic_keyboard(topic_id, page_colls, assigned_ids, page, total_pages),
        parse_mode='HTML'
    )


@router.callback_query(F.data.startswith("assign_coll_topic_"), StateFilter('*'))
async def admin_assign_coll_to_topic(callback: CallbackQuery, state: FSMContext):
    # assign_coll_topic_{topic_id}_page_{page}
    parts = callback.data.split("_")
    topic_id = int(parts[3])
    page = int(parts[5])
    await _show_assign_collections_to_topic(
        callback.bot, callback.message.chat.id, callback.message.message_id, topic_id, page
    )


@router.callback_query(F.data.startswith("topcoll_"), StateFilter('*'))
async def admin_toggle_coll_for_topic(callback: CallbackQuery):
    parts = callback.data.split("_")
    # topcoll_{action}_{topic_id}_{coll_id}_{page}
    action = parts[1]
    topic_id = int(parts[2])
    coll_id = int(parts[3])
    page = int(parts[4])

    async with async_session_maker() as session:
        if action == "add":
            try:
                await session.execute(
                    topic_collection_association.insert().values(topic_id=topic_id, collection_id=coll_id)
                )
                await session.commit()
            except Exception:
                await session.rollback()
        elif action == "remove":
            await session.execute(
                topic_collection_association.delete().where(
                    topic_collection_association.c.topic_id == topic_id,
                    topic_collection_association.c.collection_id == coll_id
                )
            )
            await session.commit()

    await callback.answer("✅")
    await _show_assign_collections_to_topic(
        callback.bot, callback.message.chat.id, callback.message.message_id, topic_id, page
    )


async def _send_subscription_info(user_id: int, chat_id: int, bot: Bot, state: FSMContext):
    async with async_session_maker() as session:
        user_sub_result = await session.execute(
            select(UserSubscription).options(
                selectinload(UserSubscription.plan).selectinload(SubscriptionPlan.upgrades_to_plan)
            )
            .where(UserSubscription.user_id == user_id)
        )
        user_sub = user_sub_result.scalar_one_or_none()

        user = await session.get(
            User,
            user_id,
            options=[
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )
        user_promos = user.promo_codes if user else []

        sub_config_kb = await session.get(SubscriptionConfig, 1)
        referral_info = None
        if sub_config_kb and sub_config_kb.referral_enabled:
            referral_info = {
                'enabled': True,
                'sub_btn_name': sub_config_kb.referral_sub_btn_name,
            }

    sub_info = None
    text = ""
    MSK = timezone(timedelta(hours=3))
    now = datetime.utcnow()

    if user_sub and user_sub.end_date > now and user_sub.plan_id is not None:
        end_date_msk = user_sub.end_date.astimezone(MSK)

        plan = user_sub.plan
        plan_name_with_duration = "Неизвестный тариф"
        if plan:
            duration_unit_text = "дн." if plan.duration_unit == 'days' else "мес."
            plan_name_with_duration = f"{plan.name} ({plan.duration_value} {duration_unit_text})"

        plan_allows_renewal = getattr(plan, 'allow_auto_renewal', True) if plan else True
        renewal_line = ""
        if plan_allows_renewal:
            renewal_line = f"\n<b>Автопродление:</b> {'✅ Включено' if user_sub.auto_renewal else '❌ Выключено'}"
        text = (
            f"<b>⭐️ Ваша подписка активна</b>\n\n"
            f"<b>Тариф:</b> {plan_name_with_duration}\n"
            f"<b>Действует до:</b> {end_date_msk.strftime('%d.%m.%Y %H:%M')} МСК"
            f"{renewal_line}"
        )

        plan_to_charge = None
        price_line = ""

        target_plan_allows_renewal = False
        if plan.is_trial and plan.upgrades_to_plan:
            plan_to_charge = plan.upgrades_to_plan
            target_plan_allows_renewal = getattr(plan_to_charge, 'allow_auto_renewal', True)
        elif not plan.is_trial:
            plan_to_charge = plan
            target_plan_allows_renewal = getattr(plan_to_charge, 'allow_auto_renewal', True)
            if target_plan_allows_renewal:
                price_line = "\n(применится при автопродлении)"

        if plan_to_charge:
            plan_discount_percent = user_sub.discount_percent
            plan_to_charge_id = plan_to_charge.id

            plan_specific_promo = next((
                p for p in user_promos
                if not p.applies_to_all_plans and any(ap.id == plan_to_charge_id for ap in p.applicable_plans)
            ), None)

            if plan_specific_promo and plan_specific_promo.discount_percent > plan_discount_percent:
                plan_discount_percent = plan_specific_promo.discount_percent

            if not plan_specific_promo:
                all_plans_promo = next((
                    p for p in user_promos if p.applies_to_all_plans
                ), None)
                if all_plans_promo and all_plans_promo.discount_percent > plan_discount_percent:
                    plan_discount_percent = all_plans_promo.discount_percent

            if plan_discount_percent > 0:
                text += f"\n<b>Ваша скидка:</b> {plan_discount_percent}%"

            final_price_to_charge = plan_to_charge.price * (1 - plan_discount_percent / 100)

            if plan.is_trial:
                up_duration_unit = "дн" if plan_to_charge.duration_unit == 'days' else "мес"
                price_line = f"\n<b>Стоимость основного тарифа"
                if plan_discount_percent > 0:
                    price_line += " (со скидкой)"
                price_line += f":</b> {final_price_to_charge:.2f} руб за {plan_to_charge.duration_value} {up_duration_unit}"
                if target_plan_allows_renewal:
                    price_line += " (спишется при включенном автопродлении)"
                else:
                    price_line += " (оформляется вручную после окончания пробного периода)"
            else:
                price_line = f"\n<b>Стоимость"
                if plan_discount_percent > 0:
                    price_line += " (со скидкой)"
                price_line += f":</b> {final_price_to_charge:.2f} руб."

            text += price_line

        sub_info = {'auto_renewal': user_sub.auto_renewal, 'is_trial': plan.is_trial if plan else False, 'allow_auto_renewal': getattr(plan, 'allow_auto_renewal', True) if plan else True}

    else:
        is_retry_mode = (
            user_sub
            and user_sub.end_date <= now
            and user_sub.plan_id is not None
            and user_sub.auto_renewal
            and user_sub.payment_method_id
            and (
                user_sub.payment_attempt_count < 3
                or (
                    user_sub.payment_provider == 'Robokassa'
                    and user_sub.pending_robokassa_invoice_id is not None
                )
            )
        )
        if is_retry_mode:
            plan = user_sub.plan
            plan_name = plan.name if plan else "текущий тариф"
            is_pending_robokassa = (
                user_sub.payment_provider == 'Robokassa'
                and user_sub.pending_robokassa_invoice_id is not None
            )

            plan_to_charge = plan.upgrades_to_plan if (plan and plan.is_trial and plan.upgrades_to_plan) else plan
            current_discount = user_sub.discount_percent
            if plan_to_charge and user_promos:
                best_promo = next((p for p in user_promos if not p.applies_to_all_plans and any(
                    ap.id == plan_to_charge.id for ap in p.applicable_plans)), None)
                if best_promo and best_promo.discount_percent > current_discount:
                    current_discount = best_promo.discount_percent
                elif not best_promo:
                    global_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
                    if global_promo and global_promo.discount_percent > current_discount:
                        current_discount = global_promo.discount_percent

            final_price = plan_to_charge.price * (1 - current_discount / 100) if plan_to_charge else 0
            duration_text = ""
            if plan_to_charge:
                unit = "дн." if plan_to_charge.duration_unit == 'days' else "мес."
                duration_text = f"{plan_to_charge.duration_value} {unit}"

            price_line = f"\n<b>Сумма к списанию:</b> {final_price:.2f} руб." if plan_to_charge else ""
            duration_line = f"\n<b>Период:</b> {duration_text}" if duration_text else ""
            attempt_text = f"\n<b>Попыток списания:</b> {user_sub.payment_attempt_count} из 3" if user_sub.payment_attempt_count > 0 else ""
            if is_pending_robokassa:
                text = (
                    f"<b>⚠️ Подписка истекла, ожидаем результат автопродления</b>\n\n"
                    f"<b>Тариф:</b> {plan_name}{duration_line}{price_line}{attempt_text}\n\n"
                    f"Запрос на списание уже отправлен в Robokassa. Можете проверить статус "
                    f"или отменить автопродление и оформить подписку заново."
                )
                reply_markup = kb.subscription_pending_keyboard()
            else:
                text = (
                    f"<b>⚠️ Подписка истекла, ожидает оплаты по автопродлению</b>\n\n"
                    f"<b>Тариф:</b> {plan_name}{duration_line}{price_line}{attempt_text}\n\n"
                    f"К вашей карте привязан метод оплаты. Можете попробовать списание прямо сейчас "
                    f"или отменить автопродление и оформить новую подписку."
                )
                reply_markup = kb.subscription_retry_keyboard()
            await bot.send_message(chat_id, text, reply_markup=reply_markup)
            return

        if user_sub and user_sub.end_date > now and user_sub.plan_id is None and user_sub.payment_provider in ['Trial Promo', 'Trial Welcome']:
            remaining_time = user_sub.end_date - now
            remaining_seconds = remaining_time.total_seconds()

            remaining_days_display = 0
            remaining_hours = 0

            if remaining_seconds <= 0:
                remaining_days_display = 0
                remaining_hours = 0
            else:
                remaining_days_display = math.ceil(remaining_seconds / 86400)
                remaining_hours = int(remaining_time.total_seconds() / 3600)

            text = "У вас нет активной подписки.\n"

            if remaining_days_display > 1:
                text += f"\nДоступные бонусные дни: {remaining_days_display}\n"
            elif remaining_days_display == 1:
                if remaining_hours > 0:
                    text += f"\nДоступные бонусные часы: ~{remaining_hours}\n"
                else:
                    text += f"\nБонусный доступ скоро закончится.\n"
            elif remaining_days_display == 0:
                text += f"\nБонусный доступ скоро закончится.\n"

            if user_sub.discount_percent > 0:
                text += f"🔥 У вас есть скидка <b>{user_sub.discount_percent}%</b>, которая <b>сгорит</b>, если не оформить подписку до окончания бонусных дней!\n"

            text += "\nОформите подписку, чтобы получить доступ ко всем возможностям бота!"
            sub_info = None

        else:
            text = "У вас нет активной подписки.\n"

            discount_percent = 0
            if user_sub and user_sub.discount_percent > 0:
                discount_percent = user_sub.discount_percent

            for p in user_promos:
                if p.applies_to_all_plans and p.discount_percent > discount_percent:
                    discount_percent = p.discount_percent

            if discount_percent > 0:
                text += f"\nДоступная скидка: {discount_percent} % (применяется к подходящим тарифам)\n"

            text += "\nОформите ее, чтобы получить доступ ко всем возможностям бота!"
            sub_info = None

    await bot.send_message(chat_id, text, reply_markup=kb.subscription_info_keyboard(sub_info, referral_info=referral_info))


@router.message(F.text == "⭐️ Подписка")
async def show_subscription_info(message: Message, state: FSMContext, bot: Bot):
    await _send_subscription_info(message.from_user.id, message.chat.id, bot, state)


@router.callback_query(F.data == "back_to_sub_info")
async def back_to_subscription_info(callback: CallbackQuery, state: FSMContext, bot: Bot):
    await state.clear()
    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass
    await _send_subscription_info(callback.from_user.id, callback.message.chat.id, bot, state)


@router.callback_query(F.data == "sub_toggle_renewal")
async def toggle_subscription_renewal(callback: CallbackQuery, state: FSMContext, bot: Bot):
    async with async_session_maker() as session:
        user_sub = await session.scalar(
            select(UserSubscription)
            .where(UserSubscription.user_id == callback.from_user.id)
            .options(selectinload(UserSubscription.plan))
        )
        if not user_sub:
            await callback.answer("Не удалось найти вашу подписку.", show_alert=True)
            return

        user_sub.auto_renewal = not user_sub.auto_renewal
        if not user_sub.auto_renewal:
            user_sub.payment_attempt_count = 0
            user_sub.last_payment_attempt = None
            user_sub.pending_robokassa_invoice_id = None
        await session.commit()

        status_text = "включено" if user_sub.auto_renewal else "отменено"
        user_ref = f"{callback.from_user.first_name or ''}"
        if callback.from_user.username:
            user_ref += f" (@{callback.from_user.username})"
        user_ref += f" [id=<code>{callback.from_user.id}</code>]"
        plan_name = user_sub.plan.name if user_sub.plan else (user_sub.payment_provider or "Trial")
        end_date_msk = user_sub.end_date.astimezone(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M')
        if user_sub.auto_renewal:
            plog.info(f"ВКЛЮЧЕНИЕ_АВТОПРОДЛ | {user_ref} | {plan_name} | до {end_date_msk}")
        else:
            plog.info(f"ОТМЕНА_АВТОПРОДЛ | {user_ref} | {plan_name} | до {end_date_msk}")
        await callback.answer(f"Автопродление подписки {status_text}.", show_alert=True)

    await _send_subscription_info(callback.from_user.id, callback.message.chat.id, bot, state)
    await callback.message.delete()


@router.callback_query(F.data == "sub_cancel_retry")
async def handle_sub_cancel_retry(callback: CallbackQuery, state: FSMContext, bot: Bot):
    async with async_session_maker() as session:
        user_sub = await session.scalar(
            select(UserSubscription)
            .where(UserSubscription.user_id == callback.from_user.id)
            .options(selectinload(UserSubscription.plan))
        )
        if user_sub:
            user_ref = f"{callback.from_user.first_name or ''}"
            if callback.from_user.username:
                user_ref += f" (@{callback.from_user.username})"
            user_ref += f" [id=<code>{callback.from_user.id}</code>]"
            plan_name = user_sub.plan.name if user_sub.plan else (user_sub.payment_provider or "Trial")
            end_date_msk = user_sub.end_date.astimezone(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M')
            user_sub.auto_renewal = False
            user_sub.payment_attempt_count = 0
            user_sub.last_payment_attempt = None
            user_sub.pending_robokassa_invoice_id = None
            await session.commit()
            plog.info(f"ОТМЕНА_АВТОПРОДЛ | {user_ref} | {plan_name} | до {end_date_msk}")
    await callback.answer("Автопродление отменено.", show_alert=False)
    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass
    await _send_subscription_info(callback.from_user.id, callback.message.chat.id, bot, state)


@router.callback_query(F.data == "sub_retry_now")
async def handle_sub_retry_now(callback: CallbackQuery, state: FSMContext, bot: Bot):
    user_id = callback.from_user.id
    now = datetime.utcnow()
    MSK = timezone(timedelta(hours=3))

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        user_sub = await session.scalar(
            select(UserSubscription).options(
                selectinload(UserSubscription.plan).selectinload(SubscriptionPlan.upgrades_to_plan)
            ).where(UserSubscription.user_id == user_id)
        )

        has_pending_robokassa = bool(
            user_sub
            and user_sub.payment_provider == 'Robokassa'
            and user_sub.pending_robokassa_invoice_id
        )

        if (
            not user_sub
            or not user_sub.auto_renewal
            or not user_sub.payment_method_id
            or (user_sub.payment_attempt_count >= 3 and not has_pending_robokassa)
        ):
            plog.info(
                "РУЧНОЙ_РЕТРАЙ_ОТКЛОНЕН | %s | provider=%s | auto_renewal=%s | payment_method=%s | attempts=%s | pending_inv=%s",
                user_id,
                user_sub.payment_provider if user_sub else "none",
                user_sub.auto_renewal if user_sub else False,
                bool(user_sub.payment_method_id) if user_sub else False,
                user_sub.payment_attempt_count if user_sub else "none",
                user_sub.pending_robokassa_invoice_id if user_sub else "none",
            )
            await callback.answer("Невозможно выполнить списание.", show_alert=True)
            return

        if user_sub.pending_robokassa_invoice_id and config and config.robokassa_password_2:
            pending_inv_id = user_sub.pending_robokassa_invoice_id
            op_state = await check_robokassa_op_state(config, pending_inv_id)
            plog.info(
                "РУЧНОЙ_РЕТРАЙ_РЕЗУЛЬТАТ | %s | Robokassa | pending_inv=%s | op_state=%s",
                user_id,
                pending_inv_id,
                op_state,
            )
            if op_state == 'success':
                plan_ok = user_sub.plan
                if plan_ok:
                    ptc_ok = plan_ok.upgrades_to_plan if (plan_ok.is_trial and plan_ok.upgrades_to_plan) else plan_ok
                    user_sub.end_date = extend_subscription_end_date(
                        user_sub.end_date,
                        now,
                        ptc_ok.duration_value,
                        ptc_ok.duration_unit,
                    )
                    user_sub.plan_id = ptc_ok.id
                pending_payment_ok = await session.get(RobokassaPayment, pending_inv_id)
                if pending_payment_ok:
                    pending_payment_ok.status = 'completed'
                user_sub.payment_attempt_count = 0
                user_sub.last_payment_attempt = None
                user_sub.pending_robokassa_invoice_id = None
                await session.commit()
                await bot.send_message(user_id, f"✅ Подписка продлена до {user_sub.end_date.astimezone(MSK).strftime('%d.%m.%Y %H:%M')} МСК.")
                try:
                    await callback.message.delete()
                except TelegramBadRequest:
                    pass
                await _send_subscription_info(user_id, callback.message.chat.id, bot, state)
                return
            elif op_state == 'pending':
                await callback.answer("Запрос уже в обработке, ожидайте подтверждения.", show_alert=True)
                return
            elif op_state == 'failed':
                # Явный отказ провайдера: очищаем pending и, если это 3-я неудача, отключаем автопродление.
                pending_payment_fail = await session.get(RobokassaPayment, pending_inv_id)
                if pending_payment_fail:
                    pending_payment_fail.status = 'failed'
                if user_sub.payment_attempt_count >= 3:
                    user_ref = f"{callback.from_user.first_name or ''}"
                    if callback.from_user.username:
                        user_ref += f" (@{callback.from_user.username})"
                    user_ref += f" [id=<code>{user_id}</code>]"
                    await disable_auto_renewal_after_failed_attempts(
                        session,
                        bot,
                        user_sub,
                        user_ref,
                        user_sub.plan.name if user_sub.plan else "Unknown",
                        InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="💳 Оформить подписку", callback_data="show_subscription_info_from_chat")]
                        ]),
                        config,
                        await get_all_admin_ids(),
                    )
                    try:
                        await callback.message.delete()
                    except TelegramBadRequest:
                        pass
                    return
                user_sub.pending_robokassa_invoice_id = None
                await session.commit()
                await callback.answer(
                    f"Банк отклонил списание. Подписка пока активна до {user_sub.end_date.astimezone(MSK).strftime('%d.%m.%Y %H:%M МСК')}.",
                    show_alert=True
                )
                try:
                    await callback.message.delete()
                except TelegramBadRequest:
                    pass
                return
            else:
                # unknown в первые часы после запроса считаем "ещё не отражён в Robokassa":
                # иначе можно преждевременно отключить автопродление, хотя провайдер позже проведёт платёж.
                if not has_robokassa_pending_timed_out(user_sub, now):
                    await callback.answer(
                        "Предыдущий запрос ещё не отражён в Robokassa. Подождите до 3 часов.",
                        show_alert=True
                    )
                    return

                if user_sub.payment_attempt_count >= 3:
                    user_ref = f"{callback.from_user.first_name or ''}"
                    if callback.from_user.username:
                        user_ref += f" (@{callback.from_user.username})"
                    user_ref += f" [id=<code>{user_id}</code>]"
                    await disable_auto_renewal_after_failed_attempts(
                        session,
                        bot,
                        user_sub,
                        user_ref,
                        user_sub.plan.name if user_sub.plan else "Unknown",
                        InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="💳 Оформить подписку", callback_data="show_subscription_info_from_chat")]
                        ]),
                        config,
                        await get_all_admin_ids(),
                    )
                    try:
                        await callback.message.delete()
                    except TelegramBadRequest:
                        pass
                    return

                pending_payment_timeout = await session.get(RobokassaPayment, pending_inv_id)
                if pending_payment_timeout and pending_payment_timeout.status == 'pending':
                    pending_payment_timeout.status = 'timeout'
                user_sub.pending_robokassa_invoice_id = None
                await session.commit()
                await callback.answer(
                    "Статус платежа в Robokassa не подтвердился. Новый запрос сейчас не отправлялся.",
                    show_alert=True
                )
                try:
                    await callback.message.delete()
                except TelegramBadRequest:
                    pass
                return

        if not config:
            await callback.answer("Ошибка конфигурации.", show_alert=True)
            return

        if user_sub.payment_provider == 'Robokassa' and not can_retry_manually(user_sub.payment_attempt_count):
            plog.info(f"РУЧНОЙ_РЕТРАЙ_ЗАБЛОКИРОВАН | {callback.from_user.id} | Robokassa | исчерпан лимит попыток")
            await callback.answer(
                "Повторное списание недоступно: исчерпан лимит из 3 попыток.",
                show_alert=True
            )
            return

        plan = user_sub.plan
        if not plan:
            await callback.answer("Тариф не найден.", show_alert=True)
            return

        plan_to_charge = plan.upgrades_to_plan if (plan.is_trial and plan.upgrades_to_plan) else plan

        user = await session.get(User, user_id, options=[
            selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
        ])
        user_promos = user.promo_codes if user else []
        current_discount = user_sub.discount_percent

        best_promo = next((p for p in user_promos if not p.applies_to_all_plans and any(
            ap.id == plan_to_charge.id for ap in p.applicable_plans)), None)
        if best_promo and best_promo.discount_percent > current_discount:
            current_discount = best_promo.discount_percent
        elif not best_promo:
            global_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
            if global_promo and global_promo.discount_percent > current_discount:
                current_discount = global_promo.discount_percent

        final_price = plan_to_charge.price * (1 - current_discount / 100)

        user_ref = (user.first_name or "") if user else ""
        if user and user.username:
            user_ref += f" (@{user.username})"
        user_ref += f" [id=<code>{user_id}</code>]"
        plog.info(
            "РУЧНОЙ_РЕТРАЙ_ЗАПРОС | %s | provider=%s | attempts=%s | auto_renewal=%s | pending_inv=%s",
            user_id,
            user_sub.payment_provider,
            user_sub.payment_attempt_count,
            user_sub.auto_renewal,
            user_sub.pending_robokassa_invoice_id or "none",
        )

        await callback.answer("Отправляем запрос на списание...", show_alert=False)

        if user_sub.payment_provider == 'Yookassa':
            if not can_retry_now(
                user_sub.payment_attempt_count,
                user_sub.last_payment_attempt,
                now,
            ):
                next_retry_at = get_next_retry_at(user_sub.payment_attempt_count, user_sub.last_payment_attempt)
                next_retry_str = format_msk(next_retry_at, '%d.%m.%Y %H:%M МСК') if next_retry_at else "позже"
                plog.info(
                    "РУЧНОЙ_РЕТРАЙ_ЗАБЛОКИРОВАН | %s | Yookassa | attempts=%s | last_attempt=%s | next_retry_at=%s",
                    user_id,
                    user_sub.payment_attempt_count,
                    format_msk(user_sub.last_payment_attempt, '%d.%m.%Y %H:%M:%S МСК') if user_sub.last_payment_attempt else "none",
                    next_retry_str,
                )
                await callback.answer(
                    f"Повторное списание пока недоступно. Следующая попытка после {next_retry_str}.",
                    show_alert=True
                )
                return
            attempt_started_at = user_sub.last_payment_attempt or now
            plog.info(
                "РУЧНОЙ_РЕТРАЙ_ОТПРАВКА | %s | Yookassa | attempts=%s | attempt_started_at=%s | amount=%.2f | payment_method_id=%s",
                user_id,
                user_sub.payment_attempt_count,
                format_msk(attempt_started_at, '%d.%m.%Y %H:%M:%S МСК'),
                final_price,
                user_sub.payment_method_id,
            )
            res, yk_payment_id, yk_payment_status = await process_recurring_payment(
                bot, user_sub, plan_to_charge, final_price, config, attempt_started_at
            )
            plog.info(
                "РУЧНОЙ_РЕТРАЙ_РЕЗУЛЬТАТ | %s | Yookassa | result=%s | payment_id=%s | payment_status=%s",
                user_id,
                res,
                yk_payment_id or "none",
                yk_payment_status or "none",
            )
            existing_yk_payment = await session.get(YookassaPayment, yk_payment_id) if yk_payment_id else None
            yk_payment_already_processed = bool(
                existing_yk_payment
                and existing_yk_payment.status == 'completed'
                and existing_yk_payment.processed_at
            )
            if yk_payment_id and not yk_payment_already_processed:
                await session.merge(YookassaPayment(
                    payment_id=yk_payment_id,
                    user_id=user_id,
                    plan_id=plan_to_charge.id,
                    amount=final_price,
                    status='completed' if res is True else (yk_payment_status or 'failed'),
                    payment_method_id=user_sub.payment_method_id,
                    is_recurring=True,
                    processed_at=now if res is True else None
                ))

            if res is True and yk_payment_already_processed:
                user_sub.payment_attempt_count = 0
                user_sub.last_payment_attempt = None
                await session.commit()
                await callback.answer("Платёж уже обработан.", show_alert=True)
                try:
                    await callback.message.delete()
                except TelegramBadRequest:
                    pass
                await _send_subscription_info(user_id, callback.message.chat.id, bot, state)
                return

            if res is True:
                user_sub.end_date = extend_subscription_end_date(
                    user_sub.end_date,
                    now,
                    plan_to_charge.duration_value,
                    plan_to_charge.duration_unit,
                )
                user_sub.plan_id = plan_to_charge.id
                user_sub.payment_attempt_count = 0
                user_sub.last_payment_attempt = None
                await session.commit()
                pay_id_suffix = f" | PayId={yk_payment_id}" if yk_payment_id else ""
                plog.info(f"ПРОДЛЕНИЕ | Yookassa | {user_ref} | {plan_to_charge.name} | {final_price:.2f} руб{pay_id_suffix}")
                await bot.send_message(user_id,
                                       f"✅ Подписка продлена до {user_sub.end_date.astimezone(MSK).strftime('%d.%m.%Y %H:%M')} МСК.")
                cfg = await session.get(SubscriptionConfig, 1)
                if cfg and cfg.notifications_enabled:
                    for admin_id in await get_all_admin_ids():
                        try:
                            await bot.send_message(
                                admin_id,
                                f"🔔 Автопродление (YooKassa)!\n\nПользователь: {user_ref}\nТариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\nДо: {user_sub.end_date.astimezone(MSK).strftime('%d.%m.%Y %H:%M')} МСК" + (f"\nPayId: {yk_payment_id}" if yk_payment_id else "")
                            )
                        except Exception:
                            pass
            elif res == 'deactivate':
                plog.warning(f"АВТОПРОДЛ_ОТКЛ | {user_ref} | причина: deactivate | {plan_to_charge.name}")
                user_sub.auto_renewal = False
                await session.commit()
                await bot.send_message(user_id,
                                       "Не удалось списать средства (YooKassa). Автопродление отключено.\n\nОформите подписку вручную.")
                cfg = await session.get(SubscriptionConfig, 1)
                if cfg and cfg.notifications_enabled:
                    for admin_id in await get_all_admin_ids():
                        try:
                            await bot.send_message(
                                admin_id,
                                f"🚫 Автопродление отключено (отказ провайдера)\nПользователь: {user_ref}\nПровайдер: Yookassa"
                            )
                        except Exception:
                            pass
            elif res == 'provider_error':
                user_sub.last_payment_attempt = attempt_started_at
                await session.commit()
                await bot.send_message(
                    user_id,
                    "ЮKassa временно недоступна. Эта ошибка не засчитана как попытка списания.\n\n"
                    "Попробуйте снова позже."
                )
            elif res == 'integration_error':
                user_sub.last_payment_attempt = attempt_started_at
                await session.commit()
                await bot.send_message(
                    user_id,
                    "ЮKassa вернула ошибку интеграции. Эта ошибка не засчитана как попытка списания.\n\n"
                    "Попробуйте снова позже."
                )
            elif res == 'pending':
                user_sub.last_payment_attempt = attempt_started_at
                await session.commit()
                await bot.send_message(
                    user_id,
                    "Запрос в ЮKassa принят и ожидает подтверждения оплаты. Пока не отправляйте повторное списание."
                )
            else:
                attempt_num = user_sub.payment_attempt_count + 1
                pay_id_suffix = f" | PayId={yk_payment_id}" if yk_payment_id else ""
                plog.warning(f"ОШИБКА_СПИСАНИЯ | Yookassa | {user_ref} | попытка {attempt_num} | {plan_to_charge.name}{pay_id_suffix}")
                user_sub.payment_attempt_count += 1
                user_sub.last_payment_attempt = attempt_started_at
                if attempt_num >= 3:
                    await disable_auto_renewal_after_failed_attempts(
                        session,
                        bot,
                        user_sub,
                        user_ref,
                        plan_to_charge.name,
                        InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="💳 Оформить подписку", callback_data="show_subscription_info_from_chat")]
                        ]),
                        config,
                        await get_all_admin_ids(),
                    )
                    try:
                        await callback.message.delete()
                    except TelegramBadRequest:
                        pass
                    return
                await session.commit()
                await bot.send_message(user_id,
                                       "Не удалось списать средства (YooKassa). Проверьте состояние карты и попробуйте позже.")
                cfg = await session.get(SubscriptionConfig, 1)
                if cfg and cfg.notifications_enabled:
                    for admin_id in await get_all_admin_ids():
                        try:
                            await bot.send_message(
                                admin_id,
                                f"⚠️ Ошибка автосписания [{attempt_num}/3]\nПользователь: {user_ref}\nТариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\nПровайдер: Yookassa" + (f"\nPayId: {yk_payment_id}" if yk_payment_id else "")
                            )
                        except Exception:
                            pass

        elif user_sub.payment_provider == 'Robokassa':
            new_payment = RobokassaPayment(user_id=user_id, plan_id=plan_to_charge.id, amount=final_price)
            session.add(new_payment)
            await session.commit()
            plog.info(
                "РУЧНОЙ_РЕТРАЙ_ОТПРАВКА | %s | Robokassa | attempts=%s | amount=%.2f | parent_inv=%s | new_inv=%s",
                user_id,
                user_sub.payment_attempt_count,
                final_price,
                user_sub.payment_method_id or "none",
                new_payment.id,
            )

            robokassa_res = await process_recurring_robokassa_payment(
                config, plan_to_charge, final_price, user_sub.payment_method_id, new_payment.id
            )
            plog.info(
                "РУЧНОЙ_РЕТРАЙ_РЕЗУЛЬТАТ | %s | Robokassa | result=%s | new_inv=%s",
                user_id,
                robokassa_res,
                new_payment.id,
            )

            if robokassa_res is True:
                plog.info(f"ЗАПРОС_ПРОДЛЕНИЯ | Robokassa | {user_ref} | {plan_to_charge.name} | {final_price:.2f} руб | InvId={new_payment.id}")
                user_sub.pending_robokassa_invoice_id = new_payment.id
                user_sub.payment_attempt_count += 1
                user_sub.last_payment_attempt = now
                await session.commit()
                await bot.send_message(user_id,
                                       f"⏳ Запрос на списание отправлен. Ожидайте подтверждения оплаты.")
            elif robokassa_res == 'deactivate':
                plog.warning(f"АВТОПРОДЛ_ОТКЛ | {user_ref} | причина: deactivate | {plan_to_charge.name}")
                user_sub.auto_renewal = False
                user_sub.payment_attempt_count = 0
                new_payment.status = 'request_deactivated'
                await session.commit()
                await bot.send_message(user_id,
                                       "Не удалось списать средства (Robokassa). Автопродление отключено.\n\nОформите подписку вручную.")
            elif robokassa_res == 'provider_error':
                new_payment.status = 'request_provider_error'
                user_sub.last_payment_attempt = now
                await session.commit()
                await bot.send_message(
                    user_id,
                    "Robokassa временно недоступна. Эта ошибка не засчитана как попытка списания.\n\n"
                    "Попробуйте снова позже."
                )
            else:
                attempt_num = user_sub.payment_attempt_count + 1
                plog.warning(f"ОШИБКА_СПИСАНИЯ | Robokassa | {user_ref} | попытка {attempt_num} | {plan_to_charge.name}")
                user_sub.payment_attempt_count += 1
                user_sub.last_payment_attempt = now
                new_payment.status = 'request_failed'
                if attempt_num >= 3:
                    plog.warning(f"АВТОПРОДЛ_ОТКЛ | {user_ref} | причина: 3 попытки | {plan_to_charge.name}")
                    user_sub.auto_renewal = False
                await session.commit()
                if attempt_num >= 3:
                    await bot.send_message(
                        user_id,
                        "Ваша подписка истекла. Не удалось списать средства после 3 попыток — автопродление отключено.\n\nПродлите подписку вручную в меню."
                    )
                    try:
                        await callback.message.delete()
                    except TelegramBadRequest:
                        pass
                    return
                await bot.send_message(
                    user_id,
                    "Не удалось списать средства (Robokassa). Проверьте состояние карты и попробуйте позже."
                )

        else:
            await callback.answer("Провайдер не поддерживает ручное списание.", show_alert=True)
            return

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass
    await _send_subscription_info(user_id, callback.message.chat.id, bot, state)


@router.callback_query(F.data == "sub_enter_promo")
async def enter_promo_code(callback: CallbackQuery, state: FSMContext):
    await state.set_state(UserStates.awaiting_promo_code)
    await callback.message.edit_text(
        "Пожалуйста, введите ваш промокод:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_sub_info")]
        ])
    )


@router.message(UserStates.awaiting_promo_code, F.text)
async def process_promo_code(message: Message, state: FSMContext, bot: Bot):
    if message.text.startswith('/'):
        await state.clear()
        if message.text == "/start":
            await cmd_start(message)
        else:
            await message.answer("Ввод промокода отменен. Вы можете продолжить общение.")
        return

    user_id = message.from_user.id
    code_text = message.text.upper().strip()
    now = datetime.utcnow()

    back_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_sub_info")]
    ])

    async with async_session_maker() as session:
        user = await session.get(User, user_id,
                                 options=[selectinload(User.promo_codes), selectinload(User.subscription)])
        if not user:
            await state.clear()
            await message.answer("Произошла ошибка, не удалось найти ваш профиль.", reply_markup=back_kb)
            return

        promo_result = await session.execute(
            select(PromoCode).where(
                func.lower(PromoCode.code) == code_text.lower(),
                PromoCode.is_active == True,
                PromoCode.times_used < PromoCode.max_uses
            ).options(selectinload(PromoCode.applicable_plans))
        )
        promo = promo_result.scalar_one_or_none()

        if not promo:
            await message.answer("❌ Промокод не найден, истёк или недействителен. Попробуйте ещё раз или нажмите «Назад».", reply_markup=back_kb)
            return

        if promo in user.promo_codes:
            await state.clear()
            await message.answer("❌ Вы уже активировали этот промокод.", reply_markup=back_kb)
            return

        if promo.discount_percent == 0 and promo.free_days == 0:
            await state.clear()
            await message.answer("❌ Этот промокод неактивен (0% скидки и 0 дней). Обратитесь к администратору.", reply_markup=back_kb)
            return

        trial_activated = False
        discount_banked = False

        user_sub = user.subscription
        is_active_sub = user_sub and user_sub.end_date > now and user_sub.plan_id is not None

        if not user_sub:
            user_sub = UserSubscription(user_id=user_id, end_date=now)
            session.add(user_sub)

        MSK = timezone(timedelta(hours=3))

        base_trial_date = now
        if user_sub.end_date > now and user_sub.plan_id is None:
            base_trial_date = user_sub.end_date

        new_end_date = base_trial_date + timedelta(days=promo.free_days)

        if promo.free_days > 0:
            if is_active_sub:
                if promo.discount_percent == 0:
                    await state.clear()
                    await message.answer(
                        f"ℹ️ Этот промокод даёт пробный период ({promo.free_days} дн.). "
                        f"Активируйте его после окончания текущей подписки.", reply_markup=back_kb)
                    return
                await message.answer(
                    f"✅ Скидка {promo.discount_percent}% сохранена, но пробный период ({promo.free_days} дн.) нельзя активировать, пока у вас есть другая активная платная подписка.")
            else:
                user_sub.plan_id = None
                user_sub.start_date = user_sub.start_date if (
                            user_sub.start_date and user_sub.start_date < now) else now
                user_sub.end_date = new_end_date
                user_sub.auto_renewal = False
                user_sub.payment_provider = 'Trial Promo'
                user_sub.payment_method_id = None
                user_sub.pending_robokassa_invoice_id = None
                user_sub.last_payment_attempt = None
                user_sub.payment_attempt_count = 0
                trial_activated = True

        if promo.discount_percent > 0:
            if promo.applies_to_all_plans:
                user_sub.discount_percent = promo.discount_percent
            discount_banked = True

        if trial_activated or discount_banked:
            await session.execute(
                update(PromoCode)
                .where(PromoCode.id == promo.id, PromoCode.times_used < PromoCode.max_uses)
                .values(times_used=PromoCode.times_used + 1)
            )
            await session.refresh(promo)
            user.promo_codes.append(promo)

            if trial_activated:
                new_trial_record = TrialUsageHistory(
                    user_id=user_id,
                    plan_id=None,
                    used_at=now
                )
                session.add(new_trial_record)

            await session.commit()

            user_ref_pc = f"{message.from_user.first_name or ''}"
            if message.from_user.username:
                user_ref_pc += f" (@{message.from_user.username})"
            user_ref_pc += f" [id=<code>{user_id}</code>]"
            plog.info(f"ПРОМОКОД | {user_ref_pc} | код={code_text} | скидка={promo.discount_percent}% | дней={promo.free_days}")

            _cfg_pc = await session.get(SubscriptionConfig, 1)
            if _cfg_pc and _cfg_pc.notifications_enabled:
                for admin_id in await get_all_admin_ids():
                    try:
                        await bot.send_message(
                            admin_id,
                            f"🎁 Активирован промокод «{code_text}»\nПользователь: {user_ref_pc}\nСкидка: {promo.discount_percent}%\nДней: {promo.free_days}"
                        )
                    except Exception:
                        pass

            await state.clear()

            end_date_msk_str = new_end_date.astimezone(MSK).strftime('%d.%m.%Y %H:%M')

            if trial_activated and discount_banked:
                await message.answer(
                    f"✅ Вам начислен пробный период: <b>{promo.free_days} дн.</b> (до {end_date_msk_str} МСК).\n\n"
                    f"🔥 Также вам назначена скидка <b>{promo.discount_percent}%</b>! "
                    f"Она сохранится для всех автоплатежей, <b>если вы оформите подписку до окончания пробного периода</b>."
                )
            elif trial_activated:
                await message.answer(
                    f"✅ Пробный период успешно активирован!\n"
                    f"Вам начислено: <b>{promo.free_days} бесплатных дней</b>.\n"
                    f"Доступ активен до: {end_date_msk_str} МСК."
                )
            elif discount_banked:
                if not is_active_sub:
                    await message.answer(
                        f"✅ Скидка <b>{promo.discount_percent}%</b> сохранена!\n"
                        "Она будет автоматически применена при выборе тарифа и всех последующих автоплатежах."
                    )
                else:
                    await message.answer(
                        f"✅ Скидка <b>{promo.discount_percent}%</b> сохранена! Она будет применена при <b>следующем</b> продлении или смене тарифа.")

            if discount_banked and not is_active_sub and not trial_activated:
                await show_plans_for_subscription(message, state)
            else:
                await _send_subscription_info(user_id, message.chat.id, bot, state)
        else:
            await state.clear()
            await message.answer("Произошла системная ошибка при активации промокода. Обратитесь в поддержку.")


async def show_plans_for_subscription(message_or_callback, state: FSMContext):
    user_id = message_or_callback.from_user.id
    now = datetime.utcnow()

    async with async_session_maker() as session:
        user = await session.get(
            User,
            user_id,
            options=[
                selectinload(User.subscription),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )
        is_admin_user = user_id in OWNER_IDS or (user and user.is_admin)
        plan_conditions = [SubscriptionPlan.is_active == True]
        if not is_admin_user:
            plan_conditions.append(SubscriptionPlan.admin_only == False)
        all_active_plans = (await session.execute(
            select(SubscriptionPlan)
            .where(*plan_conditions)
            .options(selectinload(SubscriptionPlan.upgrades_to_plan))
            .order_by(SubscriptionPlan.price.asc())
        )).scalars().all()

        user_sub = user.subscription if user else None
        user_promos = user.promo_codes if user else []

        trial_history_result = await session.execute(
            select(TrialUsageHistory).where(TrialUsageHistory.user_id == user_id)
        )
        trial_history = trial_history_result.scalars().all()

    eligible_plans = []
    for plan in all_active_plans:
        if not plan.is_trial:
            eligible_plans.append(plan)
            continue

        usage_record = next((h for h in trial_history if h.plan_id == plan.id), None)

        if not usage_record:
            planless_usage = next((h for h in trial_history if h.plan_id is None), None)
            if not planless_usage:
                eligible_plans.append(plan)
            continue

        cooldown_days = plan.trial_cooldown_days
        if cooldown_days == 0:
            continue

        if now > (usage_record.used_at + timedelta(days=cooldown_days)):
            eligible_plans.append(plan)

    global_discount_percent = 0
    if user_sub:
        global_discount_percent = user_sub.discount_percent

    text = "Выберите подходящий тариф:"

    if user_sub and user_sub.end_date > now and user_sub.plan_id is not None:
        text += "\n\n<b>При смене тарифа срок оплаты нового тарифа добавится к текущему (прибавятся неиспользуемые дни).</b>"

    has_any_discount = global_discount_percent > 0 or any(p.discount_percent > 0 for p in user_promos)
    if has_any_discount:
        text += f"\n\n<i>У вас есть активные скидки! Они будут применены к подходящим тарифам.</i>"

    if not eligible_plans:
        text = "К сожалению, сейчас нет доступных тарифных планов."

    if isinstance(message_or_callback, CallbackQuery):
        await message_or_callback.message.edit_text(text,
                                                    reply_markup=kb.plan_selection_keyboard(eligible_plans,
                                                                                             global_discount_percent,
                                                                                             user_promos))
    else:
        await message_or_callback.answer(text, reply_markup=kb.plan_selection_keyboard(eligible_plans,
                                                                                       global_discount_percent,
                                                                                       user_promos))


@router.callback_query(F.data == "sub_select_plan")
async def select_plan_callback(callback: CallbackQuery, state: FSMContext):
    await show_plans_for_subscription(callback, state)


@router.callback_query(F.data.startswith("sub_pay_"))
async def choose_payment_provider(callback: CallbackQuery, state: FSMContext):
    plan_id = int(callback.data.split("_")[2])
    async with async_session_maker() as session:
        plan = await session.get(
            SubscriptionPlan,
            plan_id,
            options=[selectinload(SubscriptionPlan.upgrades_to_plan)]
        )

        user = await session.get(
            User,
            callback.from_user.id,
            options=[
                selectinload(User.subscription),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )
        user_sub = user.subscription if user else None
        user_promos = user.promo_codes if user else []

    global_discount_percent = user_sub.discount_percent if user_sub else 0
    plan_discount_percent = global_discount_percent

    plan_specific_promo = next((
        p for p in user_promos
        if not p.applies_to_all_plans and any(ap.id == plan_id for ap in p.applicable_plans)
    ), None)

    if plan_specific_promo:
        plan_discount_percent = plan_specific_promo.discount_percent
    elif not plan_specific_promo and plan_discount_percent == 0:
        all_plans_promo = next((
            p for p in user_promos if p.applies_to_all_plans
        ), None)
        if all_plans_promo:
            plan_discount_percent = all_plans_promo.discount_percent

    final_price = plan.price
    if plan_discount_percent > 0 and not plan.is_trial:
        final_price *= (1 - plan_discount_percent / 100)

    duration_unit_text = "дн." if plan.duration_unit == 'days' else "мес."
    text = (
        f"<b>Тариф:</b> {plan.name} ({plan.duration_value} {duration_unit_text})\n"
        f"<b>Стоимость:</b> {final_price:.2f} руб.\n"
    )
    if plan.description:
        text += f"{html.escape(plan.description)}\n"
    text += "\n"

    if plan.is_trial and plan.upgrades_to_plan:
        upgrade_plan = plan.upgrades_to_plan
        upgrade_plan_id = upgrade_plan.id
        upgrade_price = upgrade_plan.price
        upgrade_plan_allows_renewal = getattr(upgrade_plan, 'allow_auto_renewal', True)

        upgrade_plan_discount_percent = global_discount_percent

        upgrade_specific_promo = next((
            p for p in user_promos
            if not p.applies_to_all_plans and any(ap.id == upgrade_plan_id for ap in p.applicable_plans)
        ), None)

        if upgrade_specific_promo:
            upgrade_plan_discount_percent = upgrade_specific_promo.discount_percent
        elif not upgrade_specific_promo and upgrade_plan_discount_percent == 0:
            all_plans_promo = next((
                p for p in user_promos if p.applies_to_all_plans
            ), None)
            if all_plans_promo:
                upgrade_plan_discount_percent = all_plans_promo.discount_percent

        if upgrade_plan_discount_percent > 0:
            upgrade_price = upgrade_price * (1 - upgrade_plan_discount_percent / 100)

        upgrade_duration_unit_text = "дн." if upgrade_plan.duration_unit == 'days' else "мес."
        if upgrade_plan_allows_renewal:
            text += (
                f"<b>Далее:</b> {upgrade_price:.2f} руб. / "
                f"{upgrade_plan.duration_value} {upgrade_duration_unit_text}\n"
                f"(автопереход на «{upgrade_plan.name}»)\n\n"
            )
        else:
            text += (
                f"<b>После пробного периода:</b> {upgrade_price:.2f} руб. / "
                f"{upgrade_plan.duration_value} {upgrade_duration_unit_text}\n"
                f"(тариф «{upgrade_plan.name}», оформление вручную)\n\n"
            )

    text += "Выберите способ оплаты:"

    await callback.message.edit_text(
        text,
        reply_markup=await kb.payment_provider_keyboard(plan_id, final_price)
    )


@router.callback_query(F.data.startswith("pay_yookassa_"))
async def create_yookassa_invoice(callback: CallbackQuery, state: FSMContext):
    from yookassa.domain.exceptions import UnauthorizedError

    parts = callback.data.split("_")
    plan_id = int(parts[2])

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        plan = await session.get(SubscriptionPlan, plan_id)
        user = await session.get(
            User, callback.from_user.id,
            options=[
                selectinload(User.subscription),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )

    if not plan:
        await callback.answer("Тариф не найден.", show_alert=True)
        return

    user_sub = user.subscription if user else None
    user_promos = user.promo_codes if user else []
    discount_percent = user_sub.discount_percent if user_sub else 0

    plan_specific_promo = next((
        p for p in user_promos
        if not p.applies_to_all_plans and any(ap.id == plan_id for ap in p.applicable_plans)
    ), None)
    if plan_specific_promo:
        discount_percent = plan_specific_promo.discount_percent
    elif discount_percent == 0:
        all_plans_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
        if all_plans_promo:
            discount_percent = all_plans_promo.discount_percent

    price = plan.price
    if discount_percent > 0 and not plan.is_trial:
        price *= (1 - discount_percent / 100)

    if not config.yookassa_shop_id or not config.yookassa_secret_key:
        await callback.message.edit_text(
            "❌ Платёжная система временно недоступна. Администратор не настроил ключи API.")
        return

    Configuration.account_id = config.yookassa_shop_id
    Configuration.secret_key = config.yookassa_secret_key

    idempotence_key = str(uuid4())
    payload = {
        "amount": {
            "value": f"{price:.2f}",
            "currency": "RUB"
        },
        "confirmation": {
            "type": "redirect",
            "return_url": f"https://t.me/{(await callback.bot.get_me()).username}"
        },
        "capture": True,
        "description": f"Оплата подписки на тариф «{plan.name}»",
        "metadata": {
            "user_id": callback.from_user.id,
            "plan_id": plan_id
        },
        "merchant_customer_id": str(callback.from_user.id),
        "save_payment_method": True
    }

    try:
        _plog_yookassa_tech(
            "TECH_INVOICE_REQUEST",
            Method="POST",
            Endpoint="/v3/payments",
            UserId=callback.from_user.id,
            PlanId=plan_id,
            IdempotenceKey=idempotence_key,
            Payload=_encode_log_json(payload),
        )
        payment = await asyncio.to_thread(Payment.create, payload, idempotence_key)
        _plog_yookassa_tech(
            "TECH_INVOICE_RESPONSE",
            PaymentId=payment.id,
            Status=payment.status,
            Body=_encode_log_json(_serialize_yookassa_payment(payment)),
        )

        payment_url = payment.confirmation.confirmation_url
        async with async_session_maker() as session:
            session.add(YookassaPayment(
                payment_id=payment.id,
                user_id=callback.from_user.id,
                plan_id=plan_id,
                amount=price,
                status=payment.status,
                payment_method_id=payment.payment_method.id if payment.payment_method else None,
                is_recurring=False
            ))
            await session.commit()

        privacy_url = config.privacy_policy_url or "#"
        offer_url = config.offer_agreement_url or "#"
        user_ref = f"{callback.from_user.first_name or ''}"
        if callback.from_user.username:
            user_ref += f" (@{callback.from_user.username})"
        user_ref += f" [id=<code>{callback.from_user.id}</code>]"
        plog.info(f"СЧЕТ_СОЗДАН | Yookassa | {user_ref} | {plan.name} | {price:.2f} руб | PayId={payment.id}")

        plan_allows_renewal = getattr(plan, 'allow_auto_renewal', True)
        payment_type_line = (
            "Регулярная оплата, можно отключить в любой момент"
            if (plan_allows_renewal or plan.is_trial)
            else "Разовая оплата"
        )

        text = (
            "Ваша ссылка на оплату готова.\n\n"
            f"Нажимая «Оплатить», я даю согласие на <a href='{privacy_url}'>обработку персональных данных</a> и принимаю <a href='{offer_url}'>договор оферты</a>.\n\n"
            f"<b>Сумма:</b> {price:.2f} руб.\n"
            f"{payment_type_line}"
        )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💳 Оплатить через ЮKassa", url=payment_url)],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data=f"sub_pay_{plan_id}_{price}")]
        ])

        await callback.message.edit_text(text, reply_markup=keyboard, disable_web_page_preview=True)

    except UnauthorizedError:
        logging.error("YooKassa UnauthorizedError: Invalid Shop ID or Secret Key.")
        _plog_yookassa_tech(
            "TECH_INVOICE_ERROR",
            ErrorClass="UnauthorizedError",
            Body="Invalid Shop ID or Secret Key",
        )
        await callback.message.edit_text(
            "❌ Не удалось создать платеж. Похоже, возникла проблема с настройками платежной системы. Мы уже работаем над этим."
        )
        for owner_id in OWNER_IDS:
            try:
                await callback.bot.send_message(
                    owner_id,
                    "🚨 <b>Критическая ошибка ЮKassa!</b>\n\nНе удалось создать платеж из-за неверных учетных данных (Shop ID или Secret Key). Пожалуйста, срочно проверьте их в админ-панели."
                )
            except Exception:
                pass
    except Exception as e:
        logging.error(f"An unexpected error occurred with YooKassa: {e}")
        error_content = getattr(e, "content", None)
        _plog_yookassa_tech(
            "TECH_INVOICE_ERROR",
            ErrorClass=type(e).__name__,
            Body=_encode_log_json(error_content) if isinstance(error_content, dict) else str(e),
        )
        await notify_admins_about_error(
            callback.bot,
            title="Сбой создания платежа YooKassa",
            user_id=callback.from_user.id,
            username=callback.from_user.username,
            full_name=callback.from_user.full_name,
            provider="YooKassa",
            stage="create_payment",
            details=str(e),
            extra={"plan_id": plan_id, "price": f"{price:.2f}"},
            exception=e,
            logger=log,
        )
        await callback.message.edit_text(
            "❌ Произошла непредвиденная ошибка при создании платежа. Пожалуйста, попробуйте позже.")

    await callback.answer()


@router.callback_query(F.data.startswith("pay_tg_"))
async def create_telegram_pay_invoice(callback: CallbackQuery, state: FSMContext, bot: Bot):
    parts = callback.data.split("_")
    plan_id = int(parts[2])

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        plan = await session.get(SubscriptionPlan, plan_id)
        user = await session.get(
            User, callback.from_user.id,
            options=[
                selectinload(User.subscription),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )

    if not plan:
        await callback.answer("Тариф не найден.", show_alert=True)
        return

    user_sub = user.subscription if user else None
    user_promos = user.promo_codes if user else []
    discount_percent = user_sub.discount_percent if user_sub else 0

    plan_specific_promo = next((
        p for p in user_promos
        if not p.applies_to_all_plans and any(ap.id == plan_id for ap in p.applicable_plans)
    ), None)
    if plan_specific_promo:
        discount_percent = plan_specific_promo.discount_percent
    elif discount_percent == 0:
        all_plans_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
        if all_plans_promo:
            discount_percent = all_plans_promo.discount_percent

    price = plan.price
    if discount_percent > 0 and not plan.is_trial:
        price *= (1 - discount_percent / 100)

    if not config.telegram_pay_token:
        await callback.message.answer("❌ Оплата через Telegram Pay временно недоступна. Администратор не настроил токен.")
        await callback.answer()
        return

    await bot.send_invoice(
        chat_id=callback.from_user.id,
        title=f"Подписка на тариф «{plan.name}»",
        description=plan.description,
        payload=f"sub-plan-{plan_id}-user-{callback.from_user.id}",
        provider_token=config.telegram_pay_token,
        currency="RUB",
        prices=[LabeledPrice(label=f"Тариф «{plan.name}»", amount=int(price * 100))]
    )
    await callback.answer()


@router.callback_query(F.data == "sub_cancel_renewal")
async def cancel_subscription_renewal(callback: CallbackQuery, state: FSMContext):
    user_ref_cr = callback.from_user.first_name or ""
    if callback.from_user.username:
        user_ref_cr += f" (@{callback.from_user.username})"
    user_ref_cr += f" [id=<code>{callback.from_user.id}</code>]"

    plan_name_cr = None
    end_date_msk_cr = None
    notifications_enabled_cr = False

    async with async_session_maker() as session:
        sub_cr = await session.scalar(
            select(UserSubscription)
            .where(UserSubscription.user_id == callback.from_user.id)
            .options(selectinload(UserSubscription.plan))
        )
        if sub_cr:
            plan_name_cr = sub_cr.plan.name if sub_cr.plan else (sub_cr.payment_provider or "Trial")
            MSK_cr = timezone(timedelta(hours=3))
            end_date_msk_cr = sub_cr.end_date.astimezone(MSK_cr).strftime('%d.%m.%Y %H:%M')

        config_cr = await session.get(SubscriptionConfig, 1)
        notifications_enabled_cr = bool(config_cr and config_cr.notifications_enabled)

        result = await session.execute(
            update(UserSubscription)
            .where(UserSubscription.user_id == callback.from_user.id)
            .values(auto_renewal=False)
            .returning(UserSubscription.id)
        )
        await session.commit()

    if result.scalar_one_or_none():
        plog.info(f"ОТМЕНА_АВТОПРОДЛ | {user_ref_cr} | {plan_name_cr} | до {end_date_msk_cr}")
        await callback.answer("Автопродление подписки отменено.", show_alert=True)
        if notifications_enabled_cr:
            for admin_id in await get_all_admin_ids():
                try:
                    await callback.bot.send_message(
                        admin_id,
                        f"🔕 Пользователь отключил автопродление\nПользователь: {user_ref_cr}\nТариф: {plan_name_cr}\nПодписка до: {end_date_msk_cr} МСК"
                    )
                except Exception:
                    pass
        await show_subscription_info(callback.message, state)
    else:
        await callback.answer("Не удалось найти активную подписку.", show_alert=True)


@router.callback_query(F.data == "admin_subscriptions")
async def admin_subscriptions_menu(callback: CallbackQuery):
    await callback.message.edit_text(
        "⭐️ Управление подписками",
        reply_markup=kb.admin_subscriptions_keyboard()
    )


_SUBS_FILTERS = {
    "all":   "Все",
    "paid":  "💰 Платные",
    "bonus": "🎁 Бонусные",
}

async def _render_subscribers_list(callback: CallbackQuery, page: int, filter_key: str):
    if filter_key not in _SUBS_FILTERS:
        filter_key = "all"

    SUBS_PAGE_SIZE = 10
    now = datetime.utcnow()
    MSK = timezone(timedelta(hours=3))

    base_where = [UserSubscription.end_date > now]
    if filter_key == "paid":
        base_where.append(UserSubscription.plan_id.isnot(None))
    elif filter_key == "bonus":
        base_where.append(UserSubscription.plan_id.is_(None))

    async with async_session_maker() as session:
        total_count = await session.scalar(
            select(func.count()).select_from(UserSubscription).where(*base_where)
        )
        total_pages = max(1, math.ceil(total_count / SUBS_PAGE_SIZE))
        page = min(page, total_pages - 1)

        rows = (await session.execute(
            select(UserSubscription, User, SubscriptionPlan)
            .join(User, User.id == UserSubscription.user_id)
            .outerjoin(SubscriptionPlan, SubscriptionPlan.id == UserSubscription.plan_id)
            .where(*base_where)
            .order_by(UserSubscription.end_date.asc())
            .offset(page * SUBS_PAGE_SIZE)
            .limit(SUBS_PAGE_SIZE)
        )).all()

    SEP = "━━━━━━━━━━━━━━━━"
    cards = []
    for sub, user, plan in rows:
        is_trial = sub.plan_id is None
        status_icon = "⏳" if is_trial else "🟢"

        name_part = f"<b>{html.escape(user.first_name or '')}</b>"
        if user.username:
            name_part += f" (@{user.username})"
        name_part += f" [id=<code>{user.id}</code>]"

        if is_trial:
            provider = sub.payment_provider or ""
            if "Promo" in provider:
                plan_line = "🎁 Промокод (бонусные дни)"
            else:
                plan_line = "⏳ Пробный период"
        else:
            plan_line = f"💳 {html.escape(plan.name if plan else '')}"

        end_date_local = sub.end_date.astimezone(MSK)
        end_date_str = end_date_local.strftime('%d.%m.%y %H:%M')
        date_line = f"📅 до {end_date_str} МСК"

        if not is_trial:
            renewal_str = "🔄 Автопродление ВКЛ" if sub.auto_renewal else "⏸ Без продления"
            date_line += f"  {renewal_str}"

        cards.append(f"{SEP}\n{status_icon} {name_part}\n{plan_line}\n{date_line}")

    filter_label = _SUBS_FILTERS[filter_key]
    text = f"👥 Подписчики · {filter_label}: {total_count} · стр. {page + 1}/{total_pages}\n"
    if cards:
        text += "\n".join(cards)
    else:
        text += f"\n{SEP}\nНет записей."

    builder = InlineKeyboardBuilder()
    filter_row = []
    for fk, flabel in _SUBS_FILTERS.items():
        btn_text = f"· {flabel} ·" if fk == filter_key else flabel
        filter_row.append(InlineKeyboardButton(text=btn_text, callback_data=f"admin_subs_0_{fk}"))
    builder.row(*filter_row)

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_subs_{page - 1}_{filter_key}"))
    nav_row.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav_row.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_subs_{page + 1}_{filter_key}"))
    if nav_row:
        builder.row(*nav_row)
    builder.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_subscriptions"))

    await callback.message.edit_text(text, reply_markup=builder.as_markup())
    await callback.answer()


@router.callback_query(F.data.startswith("admin_subs_"))
async def admin_subscribers_list(callback: CallbackQuery):
    parts = callback.data.split("_")
    page = int(parts[2])
    filter_key = parts[3] if len(parts) > 3 else "all"
    await _render_subscribers_list(callback, page, filter_key)


@router.callback_query(F.data.startswith("admin_subscribers_page_"))
async def admin_subscribers_list_legacy(callback: CallbackQuery):
    await _render_subscribers_list(callback, 0, "all")


_PLOG_FILTERS = {
    "all":    (None,                                    "Все"),
    "pay":    (["ОПЛАТА"],                              "💰 Оплаты"),
    "renew":  (["ПРОДЛЕНИЕ"],                           "🔄 Продления"),
    "err":    (["ОШИБКА_СПИСАНИЯ", "АВТОПРОДЛ_ОТКЛ"],   "❌ Ошибки"),
    "promo":  (["ПРОМОКОД"],                            "🎁 Промокоды"),
    "cancel": (["ОТМЕНА_АВТОПРОДЛ", "ИСТЕЧЕНИЕ"],       "⏸ Отмены"),
}

_PLOG_ENTRY_MAX_LEN = 320
_PLOG_MESSAGE_MAX_LEN = 3800


def _shorten_payment_log_entry(line: str, limit: int = _PLOG_ENTRY_MAX_LEN) -> str:
    normalized_line = re.sub(r"</?[^>]+>", "", line.strip())
    if len(normalized_line) <= limit:
        return normalized_line
    return f"{normalized_line[:limit - 3]}..."

async def _render_payment_log(callback: CallbackQuery, page: int, filter_key: str):
    if filter_key not in _PLOG_FILTERS:
        filter_key = "all"

    LOG_PAGE_SIZE = 20
    MAX_LOG_LINES = 400
    app_port = os.environ.get("APP_PORT", "8080")
    log_path = os.path.join(os.path.dirname(__file__), "logs", f"payment_events_{app_port}.log")

    if not os.path.exists(log_path):
        await callback.message.edit_text(
            "📋 Лог пуст — платёжных событий ещё не было.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_subscriptions")]
            ])
        )
        await callback.answer()
        return

    with open(log_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    recent_lines = all_lines[-MAX_LOG_LINES:]
    recent_lines.reverse()

    keywords, filter_label = _PLOG_FILTERS[filter_key]
    if keywords:
        recent_lines = [l for l in recent_lines if any(kw in l for kw in keywords)]

    if not recent_lines:
        filtered_text = f"📋 Лог платежей · {filter_label}\n\nЗаписей по этому фильтру нет."
        total_pages = 1
    else:
        total_pages = max(1, math.ceil(len(recent_lines) / LOG_PAGE_SIZE))
        page = min(page, total_pages - 1)
        start = page * LOG_PAGE_SIZE
        page_lines = recent_lines[start:start + LOG_PAGE_SIZE]
        separator = "─" * 38
        entries = []
        for line in page_lines:
            if not line.strip():
                continue
            entries.append(_shorten_payment_log_entry(line))
        log_body = f"\n{separator}\n".join(entries)
        max_body_len = _PLOG_MESSAGE_MAX_LEN - 120
        if len(log_body) > max_body_len:
            log_body = f"{log_body[:max_body_len - 4]} ..."
        filtered_text = (
            f"📋 Лог платежей · {filter_label} · стр. {page + 1}/{total_pages}\n\n"
            f"<pre>{html.escape(log_body)}</pre>"
        )

    builder = InlineKeyboardBuilder()
    row1, row2 = [], []
    for i, (fk, (_, flabel)) in enumerate(_PLOG_FILTERS.items()):
        btn_text = f"· {flabel} ·" if fk == filter_key else flabel
        btn = InlineKeyboardButton(text=btn_text, callback_data=f"admin_plog_0_{fk}")
        if i < 3:
            row1.append(btn)
        else:
            row2.append(btn)
    builder.row(*row1)
    builder.row(*row2)
    if recent_lines:
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_plog_{page - 1}_{filter_key}"))
        nav_row.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_plog_{page + 1}_{filter_key}"))
        if nav_row:
            builder.row(*nav_row)
    builder.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_subscriptions"))

    await callback.message.edit_text(filtered_text, reply_markup=builder.as_markup(), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("admin_plog_"))
async def admin_payment_log(callback: CallbackQuery):
    parts = callback.data.split("_")
    page = int(parts[2])
    filter_key = parts[3] if len(parts) > 3 else "all"
    await _render_payment_log(callback, page, filter_key)


@router.callback_query(F.data.startswith("admin_payment_log_page_"))
async def admin_payment_log_legacy(callback: CallbackQuery):
    await _render_payment_log(callback, 0, "all")


@router.callback_query(F.data == "noop")
async def noop_callback(callback: CallbackQuery):
    await callback.answer()


async def _show_admin_plans_list(bot: Bot, chat_id: int, message_id: int):
    async with async_session_maker() as session:
        plans = (await session.execute(
            select(SubscriptionPlan).options(selectinload(SubscriptionPlan.upgrades_to_plan)).order_by(SubscriptionPlan.price.asc())
        )).scalars().all()
    await bot.edit_message_text(
        text="📈 Управление тарифными планами",
        chat_id=chat_id,
        message_id=message_id,
        reply_markup=kb.admin_plans_keyboard(plans)
    )


@router.callback_query(F.data == "admin_plans")
async def admin_plans_list(callback: CallbackQuery):
    await _show_admin_plans_list(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )

async def _show_admin_promocodes_list(bot: Bot, chat_id: int, message_id: int):
    async with async_session_maker() as session:
        codes = (await session.execute(select(PromoCode))).scalars().all()
    await bot.edit_message_text(
        text="🎁 Управление промокодами",
        chat_id=chat_id,
        message_id=message_id,
        reply_markup=kb.admin_promocodes_keyboard(codes)
    )

@router.callback_query(F.data == "admin_promocodes")
async def admin_promocodes_list(callback: CallbackQuery):
    await _show_admin_promocodes_list(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )


@router.callback_query(F.data == "admin_payment_settings")
async def admin_payment_settings_menu(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if not config:
            config = SubscriptionConfig(id=1)
            session.add(config)
            await session.commit()

    await callback.message.edit_text(
        "⚙️ Настройки платежных систем и уведомлений",
        reply_markup=kb.admin_payment_settings_keyboard(config)
    )

@router.callback_query(F.data == "admin_toggle_sub_notifs")
async def admin_toggle_sub_notifications(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.notifications_enabled = not config.notifications_enabled
        await session.commit()

        await callback.answer(f"Уведомления {'включены' if config.notifications_enabled else 'выключены'}")
        await callback.message.edit_reply_markup(
            reply_markup=kb.admin_payment_settings_keyboard(config)
        )


@router.callback_query(F.data == "admin_toggle_subscriptions")
async def admin_toggle_subscriptions_enabled(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.subscriptions_enabled = not config.subscriptions_enabled
        await session.commit()

        await callback.answer(f"Система подписок {'включена' if config.subscriptions_enabled else 'выключена (бот бесплатный)'}")
        await callback.message.edit_reply_markup(
            reply_markup=kb.admin_payment_settings_keyboard(config)
        )


@router.callback_query(F.data == "admin_toggle_topics")
async def admin_toggle_topics_enabled(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.topics_enabled = not config.topics_enabled
        is_enabled = config.topics_enabled
        await session.commit()

    await callback.answer(f"Темы диалогов {'включены' if is_enabled else 'выключены'}")
    await show_topics_admin_list(callback, 0)


@router.callback_query(F.data == "admin_create_plan")
async def admin_create_plan_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_plan_name)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите название для нового тарифа (например, «Базовый месяц»):",
        reply_markup=kb.back_to_previous_menu("admin_plans")
    )


@router.message(AdminStates.set_plan_name, F.text)
async def process_plan_name(message: Message, state: FSMContext, bot: Bot):
    await state.update_data(name=message.text)
    await state.set_state(AdminStates.set_plan_description)

    data = await state.get_data()
    message_id_to_edit = data['message_id_to_edit']

    await message.delete()
    await bot.edit_message_text(
        "Отлично. Теперь введите краткое описание тарифа:",
        chat_id=message.chat.id,
        message_id=message_id_to_edit
    )


@router.message(AdminStates.set_plan_description, F.text)
async def process_plan_description(message: Message, state: FSMContext, bot: Bot):
    await state.update_data(description=message.text)
    await state.set_state(AdminStates.set_plan_price)

    data = await state.get_data()
    message_id_to_edit = data['message_id_to_edit']

    await message.delete()
    await bot.edit_message_text(
        "Теперь введите стоимость тарифа в рублях (например, 990.50):",
        chat_id=message.chat.id,
        message_id=message_id_to_edit
    )


@router.message(AdminStates.set_plan_price, F.text)
async def process_plan_price(message: Message, state: FSMContext, bot: Bot):
    try:
        price = float(message.text.replace(',', '.'))
        await state.update_data(price=price)
        await state.set_state(AdminStates.set_plan_duration_unit)

        data = await state.get_data()
        message_id_to_edit = data['message_id_to_edit']

        await message.delete()
        await bot.edit_message_text(
            "Отлично. Теперь выберите единицу измерения длительности:",
            chat_id=message.chat.id,
            message_id=message_id_to_edit,
            reply_markup=kb.admin_select_duration_unit_keyboard()
        )
    except ValueError:
        await message.delete()
        temp_msg = await message.answer("❌ Ошибка: Введите корректное число для цены.")
        await asyncio.sleep(4)
        await temp_msg.delete()


@router.message(AdminStates.set_plan_duration_value, F.text)
async def process_plan_duration_value(message: Message, state: FSMContext, bot: Bot):
    try:
        duration_value = int(message.text)
        data = await state.get_data()
        message_id_to_edit = data.get('message_id_to_edit')
        new_plan_id = None

        async with async_session_maker() as session:
            new_plan = SubscriptionPlan(
                name=data['name'],
                description=data['description'],
                price=data['price'],
                duration_value=duration_value,
                duration_unit=data['duration_unit']
            )
            session.add(new_plan)
            await session.commit()
            new_plan_id = new_plan.id

        await state.clear()
        await message.delete()

        if message_id_to_edit and new_plan_id:
            try:
                await _show_admin_edit_plan_menu(
                    bot=bot,
                    chat_id=message.chat.id,
                    message_id=message_id_to_edit,
                    plan_id=new_plan_id
                )
            except TelegramBadRequest:
                pass

        temp_msg = await message.answer(f"✅ Новый тариф «{data['name']}» создан! Теперь можно настроить его.")
        await asyncio.sleep(4)
        await temp_msg.delete()

    except ValueError:
        await message.delete()
        temp_msg = await message.answer("❌ Ошибка: Введите целое число для длительности.")
        await asyncio.sleep(4)
        await temp_msg.delete()


@router.callback_query(AdminStates.set_plan_duration_unit, F.data.startswith("set_duration_unit_"))
async def process_plan_duration_unit(callback: CallbackQuery, state: FSMContext, bot: Bot):
    duration_unit = callback.data.split("_")[-1]
    await state.update_data(duration_unit=duration_unit)
    await state.set_state(AdminStates.set_plan_duration_value)

    data = await state.get_data()
    message_id_to_edit = data.get('message_id_to_edit')

    unit_text = "дней" if duration_unit == 'days' else "месяцев"

    await bot.edit_message_text(
        f"И последнее: введите длительность тарифа (число {unit_text}, например, 30 или 1):",
        chat_id=callback.message.chat.id,
        message_id=message_id_to_edit,
        reply_markup=kb.back_to_previous_menu("admin_plans")
    )
    await callback.answer()


@router.callback_query(F.data.startswith("toggle_plan_activity_"))
async def admin_toggle_plan_activity(callback: CallbackQuery, state: FSMContext):
    plan_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.is_active = not plan.is_active
            await session.commit()
    await admin_edit_plan_menu(callback, state)


@router.callback_query(F.data.startswith("toggle_plan_admin_only_"))
async def admin_toggle_plan_admin_only(callback: CallbackQuery, state: FSMContext):
    plan_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.admin_only = not getattr(plan, 'admin_only', False)
            await session.commit()
    await _show_admin_edit_plan_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        plan_id=plan_id
    )
    await callback.answer()


@router.callback_query(F.data.startswith("toggle_plan_allow_auto_renewal_"))
async def admin_toggle_plan_allow_auto_renewal(callback: CallbackQuery, state: FSMContext):
    plan_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.allow_auto_renewal = not getattr(plan, 'allow_auto_renewal', True)
            await session.commit()
    await admin_edit_plan_menu(callback, state)


@router.callback_query(F.data.startswith("edit_plan_field_"))
async def admin_edit_plan_field_start(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    plan_id = int(parts[-1])
    field_to_edit = "_".join(parts[3:-1])

    field_map = {
        "name": "название",
        "description": "описание",
        "price": "цену (в рублях)",
        "duration_value": "длительность (число)",
        "duration_unit": "единицу длительности",
        "trial_cooldown_days": "кулдаун (в днях, 0 = нельзя повторно)"
    }

    await state.update_data(
        plan_id=plan_id,
        field=field_to_edit,
        message_id_to_edit=callback.message.message_id
    )

    if field_to_edit == "duration_unit":
        await state.set_state(AdminStates.edit_plan_field)
        await callback.message.edit_text(
            "Выберите новую единицу длительности:",
            reply_markup=kb.admin_select_duration_unit_keyboard(editing=True)
        )
        return

    await state.set_state(AdminStates.edit_plan_field)
    await callback.message.edit_text(
        f"Введите новое {field_map[field_to_edit]} для тарифа:",
        reply_markup=kb.back_to_previous_menu(f"admin_edit_plan_{plan_id}")
    )


@router.callback_query(F.data.startswith("admin_delete_plan_"))
async def admin_delete_plan_confirm(callback: CallbackQuery):
    plan_id = int(callback.data.split("_")[-1])
    await callback.message.edit_text(
        "Вы уверены, что хотите удалить этот тариф? Это действие необратимо и удалит тариф у всех, кто на него подписан!",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🗑️ Да, я понимаю и хочу удалить", callback_data=f"confirm_delete_plan_{plan_id}")],
            [InlineKeyboardButton(text="❌ Отмена", callback_data=f"admin_edit_plan_{plan_id}")]
        ])
    )


@router.callback_query(F.data.startswith("confirm_delete_plan_"))
async def admin_delete_plan_process(callback: CallbackQuery, state: FSMContext):
    plan_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        active_users_count = await session.scalar(
            select(func.count(UserSubscription.id)).where(UserSubscription.plan_id == plan_id)
        )

        if active_users_count > 0:
            await callback.answer(f"❌ Нельзя удалить: на этом тарифе {active_users_count} пользователей.",
                                  show_alert=True)
            await admin_edit_plan_menu(callback, state)
            return

        trial_links_count = await session.scalar(
            select(func.count(SubscriptionPlan.id)).where(SubscriptionPlan.upgrades_to_plan_id == plan_id)
        )

        if trial_links_count > 0:
            await callback.answer(f"❌ Нельзя удалить: {trial_links_count} пробных тарифов ссылаются на этот.",
                                  show_alert=True)
            await admin_edit_plan_menu(callback, state)
            return

        await session.execute(delete(SubscriptionPlan).where(SubscriptionPlan.id == plan_id))
        await session.commit()
    await callback.answer("Тариф успешно удален.", show_alert=True)
    await admin_plans_list(callback)


@router.callback_query(F.data == "admin_create_promo")
async def admin_create_promo_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_promo_code)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите текст промокода (например, `SALE2025`):",
        reply_markup=kb.back_to_previous_menu("admin_promocodes")
    )


@router.message(AdminStates.set_promo_code, F.text)
async def process_promo_code_text(message: Message, state: FSMContext, bot: Bot):
    await state.update_data(code=message.text.upper())
    await state.set_state(AdminStates.set_promo_discount)

    data = await state.get_data()
    message_id_to_edit = data['message_id_to_edit']

    await message.delete()
    await bot.edit_message_text(
        "Введите скидку в процентах (0-100). Если скидки нет, введите 0:",
        chat_id=message.chat.id,
        message_id=message_id_to_edit
    )


@router.message(AdminStates.set_promo_discount, F.text)
async def process_promo_discount(message: Message, state: FSMContext, bot: Bot):
    try:
        discount = int(message.text)
        if not 0 <= discount <= 100: raise ValueError
        await state.update_data(discount_percent=discount)
        await state.set_state(AdminStates.set_promo_days)

        data = await state.get_data()
        message_id_to_edit = data['message_id_to_edit']

        await message.delete()
        await bot.edit_message_text(
            "Введите количество бесплатных дней. Если бонуса нет, введите 0:",
            chat_id=message.chat.id,
            message_id=message_id_to_edit
        )
    except ValueError:
        await message.answer("❌ Ошибка: Введите целое число от 0 до 100.")
        await message.delete()


@router.message(AdminStates.set_promo_days, F.text)
async def process_promo_days(message: Message, state: FSMContext, bot: Bot):
    try:
        days = int(message.text)
        await state.update_data(free_days=days)
        await state.set_state(AdminStates.set_promo_uses)

        data = await state.get_data()
        message_id_to_edit = data['message_id_to_edit']

        await message.delete()
        await bot.edit_message_text(
            "Введите максимальное количество использований (например, 100):",
            chat_id=message.chat.id,
            message_id=message_id_to_edit
        )
    except ValueError:
        await message.answer("❌ Ошибка: Введите целое число.")
        await message.delete()


@router.message(AdminStates.set_promo_uses, F.text)
async def process_promo_uses(message: Message, state: FSMContext, bot: Bot):
    try:
        uses = int(message.text)
        data = await state.get_data()
        message_id_to_edit = data.get('message_id_to_edit')
        new_promo_id = None

        async with async_session_maker() as session:
            new_promo = PromoCode(
                code=data['code'],
                discount_percent=data['discount_percent'],
                free_days=data['free_days'],
                max_uses=uses
            )
            session.add(new_promo)
            await session.commit()
            new_promo_id = new_promo.id

        await state.clear()
        await message.delete()

        if message_id_to_edit and new_promo_id:
            try:
                await _show_admin_edit_promo_menu(
                    bot=bot,
                    chat_id=message.chat.id,
                    message_id=message_id_to_edit,
                    promo_id=new_promo_id
                )
            except TelegramBadRequest:
                pass

        temp_msg = await message.answer(f"✅ Промокод «{data['code']}» создан!")
        await asyncio.sleep(4)
        await temp_msg.delete()

    except ValueError:
        await message.answer("❌ Ошибка: Введите целое число.")
        await message.delete()


async def _show_admin_payment_keys_menu(bot: Bot, chat_id: int, message_id: int):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)

    def mask_key(key):
        return f"{key[:4]}...{key[-4:]}" if key and len(key) > 8 else "Не задан"

    text = (
        "<b>Текущие API ключи и ссылки:</b>\n\n"
        f"<b>ЮKassa Shop ID:</b> `{config.yookassa_shop_id or 'Не задан'}`\n"
        f"<b>ЮKassa Secret Key:</b> `{mask_key(config.yookassa_secret_key)}`\n\n"
        f"<b>Robokassa Merchant:</b> `{config.robokassa_merchant_login or 'Не задан'}`\n"
        f"<b>Robokassa Pass 1:</b> `{mask_key(config.robokassa_password_1)}`\n"
        f"<b>Robokassa Pass 2:</b> `{mask_key(config.robokassa_password_2)}`\n\n"
        f"<b>Telegram Pay Token:</b> `{mask_key(config.telegram_pay_token)}`\n\n"
        f"<b>URL Договора оферты:</b>\n`{config.offer_agreement_url or 'Не задан'}`\n"
        f"<b>URL Политики конфид.:</b>\n`{config.privacy_policy_url or 'Не задан'}`\n\n"
        "Нажмите на кнопку, чтобы изменить соответствующий ключ или ссылку."
    )
    await bot.edit_message_text(
        text=text,
        chat_id=chat_id,
        message_id=message_id,
        reply_markup=kb.admin_payment_keys_keyboard()
    )


@router.callback_query(F.data == "admin_payment_keys_menu")
async def admin_payment_keys_menu(callback: CallbackQuery):
    await _show_admin_payment_keys_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )


@router.callback_query(F.data.startswith("set_payment_key_"))
async def admin_set_single_payment_key_start(callback: CallbackQuery, state: FSMContext):
    key_to_edit = callback.data.split("_", 3)[-1]
    back_menu = "admin_payment_keys_menu"

    key_names = {
        "yookassa_shop_id": "ЮKassa Shop ID",
        "yookassa_secret_key": "ЮKassa Secret Key",
        "robokassa_merchant_login": "Robokassa Merchant Login",
        "robokassa_password_1": "Robokassa Password 1",
        "robokassa_password_2": "Robokassa Password 2",
        "telegram_pay_token": "Telegram Pay Token",
        "offer_agreement_url": "URL: Договор оферты",
        "privacy_policy_url": "URL: Политика конфид."
    }

    await state.set_state(AdminStates.set_single_payment_key)
    await state.update_data(
        key_to_edit=key_to_edit,
        message_id_to_edit=callback.message.message_id,
        back_menu=back_menu
    )

    await callback.message.edit_text(
        f"Пожалуйста, отправьте новое значение для <b>{key_names.get(key_to_edit, 'Неизвестный ключ')}</b>:",
        reply_markup=kb.back_to_previous_menu(back_menu)
    )


@router.message(AdminStates.set_single_payment_key, F.text)
async def admin_process_single_payment_key(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()

    if data.get('link_type'):
        return await process_test_link_input(message, state, bot)

    key_to_edit = data['key_to_edit']
    message_id_to_edit = data['message_id_to_edit']
    back_menu = data.get('back_menu', 'admin_panel')
    new_value = message.text.strip()

    async with async_session_maker() as session:
        await session.execute(
            update(SubscriptionConfig).where(SubscriptionConfig.id == 1).values({key_to_edit: new_value})
        )
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id_to_edit:
        try:
            if back_menu == "admin_payment_keys_menu":
                await _show_admin_payment_keys_menu(
                    bot=bot,
                    chat_id=message.chat.id,
                    message_id=message_id_to_edit
                )
            elif back_menu == "admin_ai_keys":
                callback_mock = type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'chat': message.chat,
                        'message_id': message_id_to_edit,
                        'edit_text': lambda *args, **kwargs: bot.edit_message_text(chat_id=message.chat.id,
                                                                                   message_id=message_id_to_edit, *args,
                                                                                   **kwargs)
                    })
                })()
                await admin_ai_keys_models(callback_mock)

        except TelegramBadRequest:
            pass

    temp_msg = await message.answer("✅ Ключ успешно обновлен!")
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.callback_query(F.data.startswith("admin_edit_promo_"))
async def admin_edit_promo_menu(callback: CallbackQuery):
    promo_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)

    if not promo:
        await callback.answer("Промокод не найден.", show_alert=True)
        return

    text = (
        f"<b>Редактирование промокода: `{promo.code}`</b>\n\n"
        f"<b>Скидка:</b> {promo.discount_percent}%\n"
        f"<b>Бесплатные дни:</b> {promo.free_days}\n"
        f"<b>Использовано:</b> {promo.times_used} / {promo.max_uses}\n"
        f"<b>Статус:</b> {'🟢 Активен' if promo.is_active else '⚪️ Неактивен'}"
    )
    await callback.message.edit_text(text, reply_markup=kb.admin_edit_promo_keyboard(promo.id, promo.is_active))


@router.callback_query(F.data.startswith("toggle_promo_activity_"))
async def admin_toggle_promo_activity(callback: CallbackQuery):
    promo_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if promo:
            promo.is_active = not promo.is_active
            await session.commit()
    await admin_edit_promo_menu(callback)


@router.callback_query(F.data.startswith("admin_delete_promo_"))
async def admin_delete_promo_confirm(callback: CallbackQuery):
    promo_id = int(callback.data.split("_")[-1])
    await callback.message.edit_text(
        "Вы уверены, что хотите удалить этот промокод? Это действие необратимо.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🗑️ Да, удалить", callback_data=f"confirm_delete_promo_{promo_id}")],
            [InlineKeyboardButton(text="❌ Отмена", callback_data=f"admin_edit_promo_{promo_id}")]
        ])
    )


@router.callback_query(F.data.startswith("confirm_delete_promo_"))
async def admin_delete_promo_process(callback: CallbackQuery):
    promo_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if not promo:
            await callback.answer("Промокод не найден.", show_alert=True)
            await admin_promocodes_list(callback)
            return

        if promo.times_used > 0:
            await callback.answer(f"❌ Нельзя удалить: промокод был использован {promo.times_used} раз.", show_alert=True)
            await admin_edit_promo_menu(callback)
            return

        await session.execute(delete(PromoCode).where(PromoCode.id == promo_id))
        await session.commit()
    await callback.answer("Промокод успешно удален.", show_alert=True)
    await admin_promocodes_list(callback)


@router.pre_checkout_query()
async def pre_checkout_query_handler(pre_checkout_query: PreCheckoutQuery, bot: Bot):
    await bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)


@router.message(F.successful_payment)
async def successful_payment_handler(message: Message, bot: Bot, state: FSMContext):
    payload_parts = message.successful_payment.invoice_payload.split('-')
    plan_id = int(payload_parts[2])
    user_id = int(payload_parts[4])

    await state.clear()
    now = datetime.utcnow()

    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            return

        if plan.is_trial:
            new_trial_record = TrialUsageHistory(
                user_id=user_id,
                plan_id=plan_id,
                used_at=now
            )
            session.add(new_trial_record)

        duration_value = plan.duration_value
        duration_unit = plan.duration_unit

        user_sub = await session.scalar(select(UserSubscription).where(UserSubscription.user_id == user_id))

        if user_sub and user_sub.end_date > now:
            start_date = user_sub.start_date
            base_end_date = user_sub.end_date
        else:
            start_date = now
            base_end_date = now

        if duration_unit == 'months':
            end_date = base_end_date + relativedelta(months=duration_value)
        else:
            end_date = base_end_date + timedelta(days=duration_value)

        if user_sub:
            user_sub.plan_id = plan_id
            user_sub.start_date = start_date
            user_sub.end_date = end_date
            user_sub.auto_renewal = True
            user_sub.payment_provider = 'TelegramPay'
            user_sub.payment_attempt_count = 0
            user_sub.last_payment_attempt = None
        else:
            new_sub = UserSubscription(
                user_id=user_id,
                plan_id=plan_id,
                start_date=start_date,
                end_date=end_date,
                auto_renewal=True,
                payment_provider='TelegramPay',
                payment_attempt_count=0,
                last_payment_attempt=None,
                discount_percent=0
            )
            session.add(new_sub)

        await session.commit()

    await bot.send_message(user_id, f"✅ Ваша подписка на тариф «{plan.name}» успешно оформлена!")

    config = await session.get(SubscriptionConfig, 1)
    if config and config.notifications_enabled:
        from config import OWNER_IDS
        for admin_id in await get_all_admin_ids():
            try:
                await bot.send_message(admin_id,
                                       f"🔔 Новый платеж (Telegram Pay)!\n\nПользователь: {user_id}\nТариф: {plan.name}")
            except Exception:
                pass


@router.callback_query(F.data.startswith("edit_promo_field_"))
async def admin_edit_promo_field_start(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    field_to_edit = "_".join(parts[3:-1])
    promo_id = int(parts[-1])

    field_map = {
        "discount_percent": "новую скидку в процентах (0-100)",
        "free_days": "новое количество бесплатных дней",
        "max_uses": "новое максимальное количество использований"
    }

    await state.set_state(AdminStates.edit_promo_field)
    await state.update_data(
        promo_id=promo_id,
        field=field_to_edit,
        message_id_to_edit=callback.message.message_id
    )

    await callback.message.edit_text(
        f"Введите {field_map[field_to_edit]}:",
        reply_markup=kb.back_to_previous_menu(f"admin_edit_promo_{promo_id}")
    )


@router.message(AdminStates.edit_promo_field, F.text)
async def process_promo_field_edit(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    promo_id = data['promo_id']
    field = data['field']
    message_id_to_edit = data.get('message_id_to_edit')

    try:
        new_value = int(message.text)
        if field == "discount_percent" and not 0 <= new_value <= 100:
            raise ValueError("Discount must be between 0 and 100")
    except ValueError:
        await message.delete()
        temp_msg = await message.answer("❌ Ошибка: Введено некорректное число. Попробуйте снова.")
        await asyncio.sleep(4)
        await temp_msg.delete()
        return

    async with async_session_maker() as session:
        await session.execute(
            update(PromoCode)
            .where(PromoCode.id == promo_id)
            .values({field: new_value})
        )
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id_to_edit:
        try:
            await _show_admin_edit_promo_menu(
                bot=bot,
                chat_id=message.chat.id,
                message_id=message_id_to_edit,
                promo_id=promo_id
            )
        except TelegramBadRequest:
            pass

    temp_msg = await message.answer("✅ Данные промокода успешно обновлены!")
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.message(F.text == "⚙️ Настройки")
async def user_settings_menu(message: Message, state: FSMContext):
    user_id = message.from_user.id
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
    if not user:
        return

    text = (
        "⚙️ <b>Настройки</b>\n\n"
        f"<b>Имя:</b> {html.escape(user.name or user.first_name or 'Не указано')}\n"
        f"<b>Пол:</b> {'👨 Мужской' if user.gender == 'male' else ('👩 Женский' if user.gender == 'female' else '❓ Не указан')}\n"
        f"<b>Возраст:</b> {user.age or 'Не указан'}\n"
        f"<b>Длина ответов:</b> {'📏 Обычный' if getattr(user, 'response_length', 'normal') != 'short' else '📏 Короткий'}\n"
    )
    await message.answer(text, reply_markup=kb.user_settings_keyboard(user))


@router.callback_query(F.data == "settings_change_name")
async def settings_change_name_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(UserStates.awaiting_new_name)
    await state.update_data(is_settings=True, settings_message_id=callback.message.message_id)
    await callback.message.edit_text("Пожалуйста, введите новое имя, как мне к вам обращаться?")
    await callback.answer()


@router.message(UserStates.awaiting_new_name, F.text)
async def process_new_name(message: Message, state: FSMContext, bot: Bot):
    user_id = message.from_user.id
    user_name = message.text.strip()

    if len(user_name) > 50 or not user_name:
        await message.answer("Пожалуйста, введите корректное имя.")
        return

    async with async_session_maker() as session:
        stmt = update(User).where(User.id == user_id).values(name=user_name)
        await session.execute(stmt)
        await session.commit()

    data = await state.get_data()
    await state.clear()
    await message.delete()

    if data.get('is_settings'):
        async with async_session_maker() as session:
            user = await session.get(User, user_id)
        text = (
            f"✅ Имя изменено на <b>{html.escape(user_name)}</b>\n\n"
            "⚙️ <b>Настройки</b>\n\n"
            f"<b>Имя:</b> {html.escape(user.name or user.first_name or 'Не указано')}\n"
            f"<b>Пол:</b> {'👨 Мужской' if user.gender == 'male' else ('👩 Женский' if user.gender == 'female' else '❓ Не указан')}\n"
            f"<b>Возраст:</b> {user.age or 'Не указан'}\n"
            f"<b>Длина ответов:</b> {'📏 Обычный' if getattr(user, 'response_length', 'normal') != 'short' else '📏 Короткий'}\n"
        )
        settings_msg_id = data.get('settings_message_id')
        if settings_msg_id:
            try:
                await bot.edit_message_text(text, chat_id=message.chat.id, message_id=settings_msg_id, reply_markup=kb.user_settings_keyboard(user))
            except Exception:
                await message.answer(text, reply_markup=kb.user_settings_keyboard(user))
        else:
            await message.answer(text, reply_markup=kb.user_settings_keyboard(user))
    else:
        await message.answer(f"Отлично! Теперь я буду называть вас {html.escape(user_name)}. Укажите ваш пол:", reply_markup=kb.gender_selection_keyboard())
        await state.update_data(is_name_change=True)
        await state.set_state(UserStates.awaiting_gender)


@router.callback_query(F.data == "settings_change_gender")
async def settings_change_gender(callback: CallbackQuery, state: FSMContext):
    await state.set_state(UserStates.awaiting_gender)
    await state.update_data(is_settings=True)
    await callback.message.edit_text("Выберите ваш пол:", reply_markup=kb.gender_selection_keyboard())
    await callback.answer()


@router.callback_query(F.data == "settings_change_age")
async def settings_change_age(callback: CallbackQuery, state: FSMContext):
    await state.set_state(UserStates.awaiting_age)
    await state.update_data(is_settings=True, settings_message_id=callback.message.message_id)
    await callback.message.edit_text("Введите ваш возраст числом (например, 25):")
    await callback.answer()


@router.callback_query(F.data == "settings_toggle_length")
async def settings_toggle_length(callback: CallbackQuery):
    user_id = callback.from_user.id
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            await callback.answer("Пользователь не найден")
            return
        new_length = 'short' if getattr(user, 'response_length', 'normal') != 'short' else 'normal'
        stmt = update(User).where(User.id == user_id).values(response_length=new_length)
        await session.execute(stmt)
        await session.commit()
        user = await session.get(User, user_id)

    length_text = "📏 Короткий" if new_length == 'short' else "📏 Обычный"
    text = (
        f"✅ Длина ответов изменена: {length_text}\n\n"
        "⚙️ <b>Настройки</b>\n\n"
        f"<b>Имя:</b> {html.escape(user.name or user.first_name or 'Не указано')}\n"
        f"<b>Пол:</b> {'👨 Мужской' if user.gender == 'male' else ('👩 Женский' if user.gender == 'female' else '❓ Не указан')}\n"
        f"<b>Возраст:</b> {user.age or 'Не указан'}\n"
        f"<b>Длина ответов:</b> {length_text}\n"
    )
    try:
        await callback.message.edit_text(text, reply_markup=kb.user_settings_keyboard(user))
    except Exception:
        pass
    await callback.answer()


@router.callback_query(F.data == "settings_close")
async def settings_close(callback: CallbackQuery):
    try:
        await callback.message.delete()
    except Exception:
        pass
    await callback.answer()

@router.callback_query(F.data == "admin_manage_admins")
async def admin_manage_admins_menu(callback: CallbackQuery):
    await _show_admin_manage_admins(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )


@router.callback_query(F.data == "admin_add_admin")
async def admin_add_admin_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.add_admin_id)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text("Введите Telegram ID пользователя, которого хотите назначить администратором:",
                                     reply_markup=kb.back_to_previous_menu("admin_manage_admins"))


@router.message(AdminStates.add_admin_id, F.text)
async def admin_add_admin_process(message: Message, state: FSMContext, bot: Bot):
    try:
        user_id = int(message.text)
    except ValueError:
        await message.answer("❌ ID должен быть числом. Попробуйте еще раз.")
        return

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            await message.answer("❌ Пользователь с таким ID не найден в базе данных бота.")
            return

        user.is_admin = True
        await session.commit()

    data = await state.get_data()
    message_id_to_edit = data.get('message_id_to_edit')

    await state.clear()
    await message.delete()

    temp_msg = await message.answer(
        f"✅ Пользователь {user.first_name} (<code>{user.id}</code>) успешно назначен администратором.")

    if message_id_to_edit:
        try:
            await _show_admin_manage_admins(
                bot=bot,
                chat_id=message.chat.id,
                message_id=message_id_to_edit
            )
        except TelegramBadRequest:
            pass

    await asyncio.sleep(4)
    await temp_msg.delete()


@router.callback_query(F.data.startswith("admin_panel_profile_"))
async def admin_view_admin_profile(callback: CallbackQuery):
    if callback.from_user.id not in OWNER_IDS:
        await callback.answer("Только владелец может управлять админами.", show_alert=True)
        return

    admin_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        admin = await session.get(User, admin_id)

    if not admin:
        await callback.answer("Админ не найден.", show_alert=True)
        return

    text = (
        f"<b>Профиль администратора:</b>\n\n"
        f"<b>Имя:</b> {html.escape(admin.first_name)}\n"
        f"<b>Username:</b> @{html.escape(admin.username or 'Не указан')}\n"
        f"<b>ID:</b> <code>{admin.id}</code>\n\n"
        f"<b>Доступ к истории:</b> {'✅' if admin.can_view_history else '❌'}"
    )

    await callback.message.edit_text(
        text,
        reply_markup=kb.admin_view_admin_profile_keyboard(admin.id, admin.can_view_history)
    )


@router.callback_query(F.data.startswith("admin_panel_toggle_history_"))
async def admin_toggle_admin_history_access(callback: CallbackQuery):
    if callback.from_user.id not in OWNER_IDS:
        await callback.answer("Только владелец может делать это.", show_alert=True)
        return

    admin_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        admin = await session.get(User, admin_id)
        if not admin:
            await callback.answer("Админ не найден.", show_alert=True)
            return

        admin.can_view_history = not admin.can_view_history
        await session.commit()
        await callback.answer(f"Доступ к истории {'ВЫДАН' if admin.can_view_history else 'ЗАБРАН'}.", show_alert=True)

    await admin_view_admin_profile(callback)


@router.callback_query(F.data.startswith("admin_panel_revoke_"))
async def admin_revoke_admin_status(callback: CallbackQuery):
    if callback.from_user.id not in OWNER_IDS:
        await callback.answer("Только владелец может делать это.", show_alert=True)
        return

    admin_id = int(callback.data.split("_")[-1])

    if admin_id == callback.from_user.id:
        await callback.answer("Вы не можете разжаловать сами себя.", show_alert=True)
        return

    async with async_session_maker() as session:
        admin = await session.get(User, admin_id)
        if not admin:
            await callback.answer("Админ не найден.", show_alert=True)
            return

        admin.is_admin = False
        admin.can_view_history = False
        await session.commit()

        await callback.answer(f"Права админа и доступ к истории для {admin.first_name} отозваны.", show_alert=True)

    await _show_admin_manage_admins(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )


@router.callback_query(F.data == "admin_mailing_menu")
async def admin_mailing_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()

    if callback.message.photo or callback.message.video:
        try:
            await callback.message.delete()
        except Exception as e:
            logging.error(f"Error deleting mailing preview: {e}")

        await callback.message.answer(
            "✉️ <b>Управление рассылками</b>",
            reply_markup=kb.mailing_menu_keyboard(),
            parse_mode="HTML"
        )
    else:
        try:
            await callback.message.edit_text(
                "✉️ <b>Управление рассылками</b>",
                reply_markup=kb.mailing_menu_keyboard(),
                parse_mode="HTML"
            )
        except Exception as e:
            logging.error(f"Error editing mailing menu: {e}")
            await callback.message.answer(
                "✉️ <b>Управление рассылками</b>",
                reply_markup=kb.mailing_menu_keyboard(),
                parse_mode="HTML"
            )

    await callback.answer()


@router.callback_query(F.data == "mailing_create")
async def admin_create_mailing_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.mailing_audience)
    await state.update_data(message_id=callback.message.message_id)
    await callback.message.edit_text("Шаг 1: Выберите аудиторию для рассылки:",
                                     reply_markup=kb.mailing_audience_keyboard())


def _birthday_mailing_step_text() -> str:
    return (
        "Шаг 2: Отправьте текст, фото или видео для ДР-рассылки.\n\n"
        "Если хотите сначала добавить только медиа, можно отправить фото/видео без подписи, "
        "а потом подставить готовый шаблон ДР.\n\n"
        f"Доступные подстановки: <code>{html.escape(BIRTHDAY_PLACEHOLDER_HINT)}</code>"
    )


async def _get_latest_birthday_mailing(session):
    return await session.scalar(
        select(Mailing)
        .where(Mailing.recurring_type == BIRTHDAY_MAILING_TYPE)
        .order_by(Mailing.created_at.desc(), Mailing.id.desc())
        .limit(1)
    )


async def _show_birthday_template_screen(message: Message, bot: Bot):
    async with async_session_maker() as session:
        mailing = await _get_latest_birthday_mailing(session)

    if not mailing:
        await message.edit_text(
            "🎂 <b>Шаблон ДР-рассылки</b>\n\n"
            "Сейчас шаблон не создан.\n\n"
            "Создай один шаблон, и планировщик будет использовать его автоматически в день рождения пользователя.",
            reply_markup=kb.birthday_template_keyboard(),
            parse_mode="HTML",
        )
        return

    callback_stub = SimpleNamespace(message=message, from_user=SimpleNamespace(id=message.chat.id), data=f"mailing_details_{mailing.id}")
    await admin_mailing_details(callback_stub, bot)


@router.callback_query(F.data.startswith("mailing_audience_"), AdminStates.mailing_audience)
async def admin_select_mailing_audience(callback: CallbackQuery, state: FSMContext):
    audience = callback.data.replace("mailing_audience_", "")
    await state.update_data(audience=audience)
    await state.set_state(AdminStates.mailing_content)

    step_text = "Шаг 2: Отправьте сообщение для рассылки (текст, фото с подписью или видео с подписью)."
    if audience == "birthday_today":
        step_text = _birthday_mailing_step_text()

    await callback.message.edit_text(
        step_text,
        reply_markup=kb.mailing_content_keyboard(audience),
        parse_mode="HTML",
    )


@router.callback_query(F.data == "mailing_create_birthday_template")
async def admin_create_birthday_template(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.mailing_content)
    await state.update_data(audience="birthday_today", message_id=callback.message.message_id)
    await callback.message.edit_text(
        _birthday_mailing_step_text(),
        reply_markup=kb.mailing_content_keyboard("birthday_today"),
        parse_mode="HTML",
    )


@router.callback_query(F.data == "mailing_birthday_template")
async def admin_birthday_template_menu(callback: CallbackQuery, state: FSMContext, bot: Bot):
    await state.clear()
    await _show_birthday_template_screen(callback.message, bot)
    await callback.answer()


@router.callback_query(F.data == "mailing_use_birthday_template", AdminStates.mailing_content)
@router.callback_query(F.data == "mailing_use_birthday_template", AdminStates.mailing_media_position)
async def admin_use_birthday_template(callback: CallbackQuery, state: FSMContext, bot: Bot):
    data = await state.get_data()
    if data.get("audience") != "birthday_today":
        await callback.answer()
        return

    if data.get("media_file_id") and not data.get("media_position"):
        await state.update_data(text=DEFAULT_BIRTHDAY_TEMPLATE)
        await callback.message.edit_text(
            "<b>Шаг 3:</b> Шаблон подставлен. Теперь выберите порядок отображения медиафайла и текста:",
            reply_markup=kb.mailing_media_position_keyboard("birthday_today"),
            parse_mode="HTML",
        )
        await state.set_state(AdminStates.mailing_media_position)
        await callback.answer("Шаблон подставлен")
        return

    await state.update_data(text=DEFAULT_BIRTHDAY_TEMPLATE, media_position='media_top')
    await show_mailing_preview(callback.message.chat.id, callback.message.message_id, state, bot)
    await callback.answer("Шаблон подставлен")


@router.message(AdminStates.mailing_content, (F.text | F.photo | F.video))
async def admin_process_mailing_content(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    audience = data.get("audience")
    current_text = data.get('text')
    current_media_id = data.get('media_file_id')
    current_media_type = data.get('media_file_type')
    current_position = data.get('media_position')
    target_msg_id = data.get('message_id')

    if message.photo:
        current_media_id = message.photo[-1].file_id
        current_media_type = 'photo'
        if message.caption:
            current_text = message.html_text
    elif message.video:
        current_media_id = message.video.file_id
        current_media_type = 'video'
        if message.caption:
            current_text = message.html_text
    elif message.text:
        current_text = message.html_text

    # For birthday templates, sending media without caption should not block the template flow.
    if audience == "birthday_today" and current_media_id and not current_text:
        current_text = DEFAULT_BIRTHDAY_TEMPLATE

    await message.delete()
    await state.update_data(
        text=current_text,
        media_file_id=current_media_id,
        media_file_type=current_media_type
    )

    if current_media_id and not current_position:
        await state.set_state(AdminStates.mailing_media_position)
        kb_markup = keyboards.mailing_media_position_keyboard(audience)
        instr_text = "<b>Шаг 3:</b> Выберите порядок отображения медиафайла и текста:"
        if audience == "birthday_today":
            instr_text = (
                "<b>Шаг 3:</b> Выберите порядок отображения медиафайла и текста.\n\n"
                f"Если нужно, можно ещё раз нажать <code>✨ Подставить шаблон ДР</code>."
            )
        try:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=target_msg_id,
                text=instr_text,
                reply_markup=kb_markup,
                parse_mode="HTML"
            )
        except Exception:
            new_msg = await bot.send_message(
                chat_id=message.chat.id,
                text=instr_text,
                reply_markup=kb_markup,
                parse_mode="HTML"
            )
            await state.update_data(message_id=new_msg.message_id)
    else:
        if not current_position:
            await state.update_data(media_position='media_top')
        await show_mailing_preview(message.chat.id, target_msg_id, state, bot)


@router.callback_query(AdminStates.mailing_media_position, F.data.startswith("mailing_pos_"))
async def admin_select_mailing_media_position(callback: CallbackQuery, state: FSMContext, bot: Bot):
    position = callback.data.replace("mailing_pos_", "")
    await state.update_data(media_position=position)
    await show_mailing_preview(callback.message.chat.id, callback.message.message_id, state, bot)
    await callback.answer()


async def show_mailing_preview(chat_id: int, message_id: int, state: FSMContext, bot: Bot):
    import re
    data = await state.get_data()
    text = data.get('text') or ""
    media_file_id = data.get('media_file_id')
    media_file_type = data.get('media_file_type')
    position = data.get('media_position', 'media_top')

    audience_name = get_mailing_audience_label(data['audience'])

    plain_text = re.sub(r'<[^>]+>', '', text)
    display_text = plain_text if len(plain_text) < 500 else plain_text[:500] + "..."
    pos_text = "🖼 Медиа сверху" if position == 'media_top' else "📝 Текст сверху"

    preview_caption = (f"<b>Предпросмотр рассылки</b>\n"
                       f"Аудитория: {audience_name}\n"
                       f"Порядок: {pos_text}\n"
                       f"-------------------\n"
                       f"{html.escape(display_text) if display_text else '[Текст отсутствует]'}\n"
                       f"-------------------")
    if data.get("audience") == "birthday_today":
        preview_caption += f"\nПеременные: <code>{html.escape(BIRTHDAY_PLACEHOLDER_HINT)}</code>"

    await state.set_state(AdminStates.mailing_confirmation)

    try:
        if media_file_id:
            try:
                await bot.delete_message(chat_id, message_id)
            except Exception:
                pass

            if media_file_type == 'photo':
                new_msg = await bot.send_photo(chat_id, media_file_id, caption=preview_caption, parse_mode="HTML",
                                               reply_markup=keyboards.mailing_confirmation_keyboard())
            else:
                new_msg = await bot.send_video(chat_id, media_file_id, caption=preview_caption, parse_mode="HTML",
                                               reply_markup=keyboards.mailing_confirmation_keyboard())
            await state.update_data(message_id=new_msg.message_id)
        else:
            await bot.edit_message_text(preview_caption, chat_id=chat_id, message_id=message_id, parse_mode="HTML",
                                        reply_markup=keyboards.mailing_confirmation_keyboard())
    except Exception:
        new_msg = await bot.send_message(chat_id, preview_caption, parse_mode="HTML",
                                         reply_markup=keyboards.mailing_confirmation_keyboard())
        await state.update_data(message_id=new_msg.message_id)


@router.callback_query(F.data == "mailing_edit_content", AdminStates.mailing_confirmation)
async def admin_edit_mailing_content(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.mailing_content)
    data = await state.get_data()
    audience = data.get("audience", "all")
    try:
        await callback.message.delete()
    except Exception:
        pass

    text = "Шаг 2: Отправьте сообщение для рассылки (текст, фото с подписью или видео с подписью)."
    if audience == "birthday_today":
        text = _birthday_mailing_step_text()
    sent_msg = await callback.message.answer(
        text,
        reply_markup=kb.mailing_content_keyboard(audience),
        parse_mode="HTML",
    )
    await state.update_data(message_id=sent_msg.message_id)
    await callback.answer()


@router.callback_query(F.data == "mailing_confirm_send", AdminStates.mailing_confirmation)
async def admin_confirm_mailing(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    audience = data.get('audience')
    async with async_session_maker() as session:
        if audience == "birthday_today":
            new_mailing = await _get_latest_birthday_mailing(session)
            if new_mailing is None:
                new_mailing = Mailing(
                    target_audience=audience,
                    creator_id=callback.from_user.id,
                    recurring_type=BIRTHDAY_MAILING_TYPE,
                    is_enabled=True,
                    status='active'
                )
                session.add(new_mailing)
            new_mailing.text = data.get('text')
            new_mailing.media_file_id = data.get('media_file_id')
            new_mailing.media_file_type = data.get('media_file_type')
            new_mailing.media_position = data.get('media_position', 'media_top')
            new_mailing.target_audience = audience
            new_mailing.creator_id = callback.from_user.id
            new_mailing.recurring_type = BIRTHDAY_MAILING_TYPE
            new_mailing.is_enabled = True
            new_mailing.status = 'active'
        else:
            new_mailing = Mailing(
                text=data.get('text'),
                media_file_id=data.get('media_file_id'),
                media_file_type=data.get('media_file_type'),
                media_position=data.get('media_position', 'media_top'),
                target_audience=audience,
                creator_id=callback.from_user.id,
                recurring_type=None,
                is_enabled=True,
                status='pending'
            )
            session.add(new_mailing)
        await session.commit()

    await state.clear()
    await callback.message.delete()
    if audience == "birthday_today":
        await callback.message.answer(
            "✅ Шаблон ДР-рассылки сохранен. Он обновляется в одном месте и будет отправляться пользователям в день рождения.",
            reply_markup=kb.birthday_template_keyboard(new_mailing.id, new_mailing.is_enabled)
        )
    else:
        await callback.message.answer("✅ Рассылка добавлена в очередь на отправку! Вы получите отчет по завершении.")
    await callback.answer()


@router.message(AdminStates.edit_plan_field, F.text)
async def process_plan_field_edit(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    plan_id = data['plan_id']
    field = data['field']
    message_id_to_edit = data.get('message_id_to_edit')
    new_value = message.text

    try:
        if field == 'price':
            new_value = float(new_value.replace(',', '.'))
        elif field in ['duration_value', 'trial_cooldown_days']:
            new_value = int(new_value)
    except ValueError:
        await message.delete()
        temp_msg = await message.answer("❌ Ошибка: Введено некорректное число. Попробуйте снова.")
        await asyncio.sleep(4)
        await temp_msg.delete()
        return

    async with async_session_maker() as session:
        await session.execute(
            update(SubscriptionPlan)
            .where(SubscriptionPlan.id == plan_id)
            .values({field: new_value})
        )
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id_to_edit:
        try:
            await _show_admin_edit_plan_menu(
                bot=bot,
                chat_id=message.chat.id,
                message_id=message_id_to_edit,
                plan_id=plan_id
            )
        except TelegramBadRequest:
            pass

    temp_msg = await message.answer("✅ Данные тарифа успешно обновлены!")
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.callback_query(AdminStates.edit_plan_field, F.data.startswith("set_duration_unit_"))
async def process_plan_field_edit_callback(callback: CallbackQuery, state: FSMContext, bot: Bot):
    data = await state.get_data()
    plan_id = data['plan_id']
    field = data['field']
    message_id_to_edit = data.get('message_id_to_edit')
    new_value = callback.data.split("_")[-1]

    if field != 'duration_unit':
         await callback.answer("Ошибка состояния. Попробуйте снова.", show_alert=True)
         await state.clear()
         return

    async with async_session_maker() as session:
        await session.execute(
            update(SubscriptionPlan)
            .where(SubscriptionPlan.id == plan_id)
            .values({field: new_value})
        )
        await session.commit()

    await state.clear()

    if message_id_to_edit:
        try:
            await _show_admin_edit_plan_menu(
                bot=bot,
                chat_id=callback.message.chat.id,
                message_id=message_id_to_edit,
                plan_id=plan_id
            )
        except TelegramBadRequest:
            pass

    await callback.answer("✅ Единица длительности обновлена!")


async def _show_admin_edit_promo_menu(bot: Bot, chat_id: int, message_id: int, promo_id: int):
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id, options=[selectinload(PromoCode.applicable_plans)])

    if not promo:
        await bot.edit_message_text("Промокод не найден.", chat_id=chat_id, message_id=message_id, reply_markup=None)
        return

    if promo.applies_to_all_plans:
        applies_to_text = "✅ Все тарифы"
    else:
        applies_to_text = f"Только {len(promo.applicable_plans)} тарифов"

    text = (
        f"<b>Редактирование промокода: `{promo.code}`</b>\n\n"
        f"<b>Скидка:</b> {promo.discount_percent}%\n"
        f"<b>Дни (для триала):</b> {promo.free_days}\n"
        f"<b>Использовано:</b> {promo.times_used} / {promo.max_uses}\n"
        f"<b>Применяется к:</b> {applies_to_text}\n"
        f"<b>Статус:</b> {'🟢 Активен' if promo.is_active else '⚪️ Неактивен'}"
    )
    await bot.edit_message_text(
        text=text,
        chat_id=chat_id,
        message_id=message_id,
        reply_markup=kb.admin_edit_promo_keyboard(promo.id, promo.is_active)
    )


async def _show_assign_promo_to_plan_menu(bot: Bot, chat_id: int, message_id: int, promo_id: int, page: int = 0):
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id, options=[selectinload(PromoCode.applicable_plans)])
        if not promo:
            await bot.edit_message_text("Промокод не найден.", chat_id=chat_id, message_id=message_id)
            return

        assigned_plan_ids = {f.id for f in promo.applicable_plans}

        total_plans = await session.scalar(select(func.count(SubscriptionPlan.id)))
        total_pages = math.ceil(total_plans / PAGE_SIZE) or 1
        page = max(0, min(page, total_pages - 1))

        all_plans_q = await session.execute(
            select(SubscriptionPlan).order_by(SubscriptionPlan.name).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
        )
        all_plans = all_plans_q.scalars().all()

    text = f"Привязка тарифов к промокоду «{promo.code}» (Стр. {page + 1}/{total_pages})\n"
    if not promo.applies_to_all_plans:
        text += "Нажмите на тариф, чтобы применить или убрать промокод."

    await bot.edit_message_text(
        text=text,
        chat_id=chat_id,
        message_id=message_id,
        reply_markup=kb.assign_promo_to_plan_keyboard(
            promo_id, all_plans, assigned_plan_ids, promo.applies_to_all_plans, page, total_pages
        )
    )


@router.callback_query(F.data.startswith("admin_assign_promo_"))
async def admin_assign_promo_to_plan(callback: CallbackQuery):
    parts = callback.data.split("_")
    promo_id = int(parts[3])
    page = int(parts[5])
    await _show_assign_promo_to_plan_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        promo_id=promo_id,
        page=page
    )


@router.callback_query(F.data.startswith("promo_plan_toggle_"))
async def admin_toggle_promo_plan_assignment(callback: CallbackQuery):
    parts = callback.data.split("_")
    action = parts[3]
    promo_id = int(parts[4])
    plan_id = int(parts[5])
    page = int(parts[6])

    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id, options=[selectinload(PromoCode.applicable_plans)])
        plan = await session.get(SubscriptionPlan, plan_id)

        if not promo or not plan:
            await callback.answer("Ошибка: Промокод или тариф не найден.", show_alert=True)
            return

        if action == "add" and plan not in promo.applicable_plans:
            promo.applicable_plans.append(plan)
        elif action == "remove" and plan in promo.applicable_plans:
            promo.applicable_plans.remove(plan)

        await session.commit()
        await callback.answer(f"Тариф '{plan.name}' {'добавлен' if action == 'add' else 'удален'}.")

    await _show_assign_promo_to_plan_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        promo_id=promo_id,
        page=page
    )


@router.callback_query(F.data.startswith("promo_toggle_all_plans_"))
async def admin_toggle_promo_all_plans(callback: CallbackQuery):
    parts = callback.data.split("_")
    promo_id = int(parts[4])
    page = int(parts[5])

    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if not promo:
            await callback.answer("Ошибка: Промокод не найден.", show_alert=True)
            return

        promo.applies_to_all_plans = not promo.applies_to_all_plans
        await session.commit()
        await callback.answer(f"Применение ко всем тарифам: {'ВКЛ' if promo.applies_to_all_plans else 'ВЫКЛ'}.")

    await _show_assign_promo_to_plan_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        promo_id=promo_id,
        page=page
    )


@router.callback_query(F.data.startswith("admin_edit_promo_"))
async def admin_edit_promo_menu(callback: CallbackQuery):
    promo_id = int(callback.data.split("_")[-1])
    await _show_admin_edit_promo_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        promo_id=promo_id
    )


async def delete_message_after_delay(message: Message, delay: int):
    await asyncio.sleep(delay)
    try:
        await message.delete()
    except TelegramBadRequest:
        pass


@router.callback_query(F.data == "admin_manage_buttons")
async def admin_manage_buttons(callback: CallbackQuery):
    await _show_admin_manage_buttons(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )


@router.callback_query(F.data.startswith("edit_button_visibility_"))
async def admin_toggle_button_visibility(callback: CallbackQuery):
    button_key = callback.data.replace("edit_button_visibility_", "")
    async with async_session_maker() as session:
        button = await session.get(Content, button_key)
        if button:
            button.is_visible = not button.is_visible
            await session.commit()
    await _show_admin_manage_buttons(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )


@router.callback_query(F.data.startswith("edit_button_title_"))
async def admin_edit_button_title_start(callback: CallbackQuery, state: FSMContext):
    button_key = callback.data.replace("edit_button_title_", "")
    await state.set_state(AdminStates.set_button_title)
    await state.update_data(button_key=button_key, message_id=callback.message.message_id)
    await callback.message.edit_text(
        "Введите новое название для кнопки:",
        reply_markup=kb.back_to_previous_menu("admin_manage_buttons")
    )


@router.message(AdminStates.set_button_title, F.text)
async def admin_process_button_title(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()

    if data.get('is_topics_btn'):
        async with async_session_maker() as session:
            stmt = update(SubscriptionConfig).where(SubscriptionConfig.id == 1).values(topics_btn_name=message.text)
            await session.execute(stmt)
            await session.commit()

        await state.clear()
        await message.delete()
        msg_id = data.get('message_id')
        if msg_id:
            try:
                async with async_session_maker() as session:
                    config = await session.get(SubscriptionConfig, 1)
                    total_topics_q = await session.execute(select(func.count(Topic.id)))
                    total_topics = total_topics_q.scalar_one()
                    total_pages = math.ceil(total_topics / PAGE_SIZE)
                    page = 0
                    topics_q = await session.execute(
                        select(Topic).order_by(Topic.name).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
                    )
                    topics = topics_q.scalars().all()

                await bot.edit_message_text(
                    f"💬 Управление темами диалогов (Стр. {page + 1}/{total_pages})",
                    chat_id=message.chat.id, message_id=msg_id,
                    reply_markup=kb.topics_admin_list_keyboard(topics, page, total_pages, config))
            except Exception:
                pass

        temp = await message.answer("✅ Название кнопки обновлено!")
        await asyncio.sleep(2)
        await temp.delete()
        return

    if data.get('is_secret_question'):
        await process_secret_question_text(message, state, bot)
        return

    button_key = data.get('button_key')

    if not button_key:
        await state.clear()
        await message.answer("❌ Ошибка: Не удалось определить тип кнопки. Попробуйте снова.")
        return

    message_id = data.get('message_id')

    async with async_session_maker() as session:
        stmt = update(Content).where(Content.key == button_key).values(button_title=message.text)
        await session.execute(stmt)
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id:
        try:
            await _show_admin_manage_buttons(bot, message.chat.id, message_id)
        except TelegramBadRequest:
            pass

    temp_msg = await message.answer("✅ Название кнопки обновлено!")
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.callback_query(F.data == "admin_add_button")
async def admin_add_button_start(callback: CallbackQuery, state: FSMContext):
    auto_key = f"btn_{uuid4().hex[:10]}"

    await state.update_data(button_key=auto_key)
    await state.set_state(AdminStates.add_button_title)
    await state.update_data(message_id=callback.message.message_id)

    text = (
        "<b>Создание новой кнопки.</b>\n\n"
        "Введите текст, который будет отображаться на кнопке в главном меню."
    )
    await callback.message.edit_text(text, reply_markup=kb.back_to_previous_menu("admin_manage_buttons"))


@router.message(AdminStates.add_button_title, F.text)
async def admin_add_button_title_process(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    button_key = data['button_key']
    message_id = data['message_id']
    button_title = message.text

    async with async_session_maker() as session:
        max_order = await session.scalar(select(func.max(Content.sort_order)))
        new_order = (max_order if max_order is not None else 0) + 1

        new_button = Content(
            key=button_key,
            button_title=button_title,
            is_visible=True,
            text_content=f"Это текст для новой кнопки '{button_title}'. Отредактируйте его в разделе '✏️ Контент'.",
            sort_order=new_order
        )
        session.add(new_button)
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id:
        try:
            await _show_admin_manage_buttons(bot, message.chat.id, message_id)
        except TelegramBadRequest:
            pass

    temp_msg = await message.answer(f"✅ Новая кнопка «{button_title}» успешно создана!")
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.callback_query(F.data.startswith("move_btn_"))
async def admin_move_button(callback: CallbackQuery):
    parts = callback.data.split("_")
    direction = parts[2]
    button_key = "_".join(parts[3:])

    async with async_session_maker() as session:
        stmt = select(Content).where(Content.button_title != None).order_by(Content.sort_order.asc())
        result = await session.execute(stmt)
        buttons = result.scalars().all()

        current_idx = -1
        for idx, btn in enumerate(buttons):
            if btn.key == button_key:
                current_idx = idx
                break

        if current_idx == -1:
            await callback.answer("Кнопка не найдена.")
            return

        target_idx = -1
        if direction == "up" and current_idx > 0:
            target_idx = current_idx - 1
        elif direction == "down" and current_idx < len(buttons) - 1:
            target_idx = current_idx + 1

        if target_idx != -1:
            btn_current = buttons[current_idx]
            btn_target = buttons[target_idx]

            c_ord = btn_current.sort_order if btn_current.sort_order is not None else 0
            t_ord = btn_target.sort_order if btn_target.sort_order is not None else 0

            if c_ord == t_ord:
                btn_current.sort_order = t_ord + 1
            else:
                btn_current.sort_order, btn_target.sort_order = t_ord, c_ord

            await session.commit()
            await callback.answer("Перемещено.")
        else:
            await callback.answer("Нельзя переместить дальше.")
            return

    try:
        await _show_admin_manage_buttons(
            bot=callback.bot,
            chat_id=callback.message.chat.id,
            message_id=callback.message.message_id
        )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("delete_button_"))
async def admin_delete_button_start(callback: CallbackQuery):
    button_key = callback.data.replace("delete_button_", "")
    async with async_session_maker() as session:
        button = await session.get(Content, button_key)

    if not button:
        await callback.answer("Кнопка уже удалена.", show_alert=True)
        return

    text = f"Вы уверены, что хотите безвозвратно удалить кнопку «{button.button_title}» и весь связанный с ней контент?"
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🗑️ Да, удалить", callback_data=f"confirm_delete_button_{button_key}")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="admin_manage_buttons")]
    ])
    await callback.message.edit_text(text, reply_markup=keyboard)


@router.callback_query(F.data.startswith("confirm_delete_button_"))
async def admin_delete_button_confirm(callback: CallbackQuery):
    button_key = callback.data.replace("confirm_delete_button_", "")
    async with async_session_maker() as session:
        await session.execute(delete(ContentMedia).where(ContentMedia.content_key == button_key))
        await session.execute(delete(Content).where(Content.key == button_key))
        await session.commit()

    await _show_admin_manage_buttons(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id
    )
    await callback.answer("Кнопка успешно удалена.", show_alert=True)


async def _show_edit_topic_menu(bot: Bot, chat_id: int, message_id: int, topic_id: int):
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id, options=[selectinload(Topic.knowledge_base_files)])
        if not topic:
            try:
                await bot.edit_message_text("Ошибка: Тема не найдена.", chat_id=chat_id, message_id=message_id)
            except TelegramBadRequest:
                pass
            return

        media_count = await session.scalar(
            select(func.count(MediaLibrary.id)).where(MediaLibrary.topic_id == topic_id)
        )

        # collections assigned to this topic
        coll_res = await session.execute(
            select(MediaCollection.name).join(
                topic_collection_association,
                topic_collection_association.c.collection_id == MediaCollection.id
            ).where(topic_collection_association.c.topic_id == topic_id)
        )
        assigned_coll_names = [r[0] for r in coll_res.all()]

    kb_files_count = len(topic.knowledge_base_files)
    colls_info = ", ".join(assigned_coll_names) if assigned_coll_names else "не привязаны"
    prompt_status = "✅ Задан" if topic.system_prompt else "❌ Не задан (используется общий)"
    active_status = "🟢 Активна" if topic.is_active else "⚪️ Неактивна"
    admin_only = getattr(topic, 'admin_only', False)
    admin_only_status = "🔒 Только для админов" if admin_only else "🔓 Видна всем"

    in_menu = "✅ В меню кнопок" if topic.show_in_main_menu else "❌ Не в меню"
    in_list = "✅ В списке тем" if topic.show_in_list else "❌ Не в списке"

    intro_status = "✅ Задано" if topic.start_message else "❌ Не задано"
    btn_status = "✅ Есть" if (topic.start_button_text and topic.start_button_payload) else "❌ Нет"

    text = (
        f"Редактирование темы: <b>{html.escape(topic.name)}</b>\n\n"
        f"<b>ID темы:</b> <code>{topic.id}</code>\n"
        f"<b>Ссылка:</b> <code>https://t.me/{(await bot.get_me()).username}?start=topic_{topic.id}</code>\n\n"
        f"<b>Статус:</b> {active_status}\n"
        f"<b>Видимость:</b> {admin_only_status}\n"
        f"<b>Отображение:</b> {in_menu} / {in_list}\n"
        f"<b>Системный промпт:</b> {prompt_status}\n"
        f"<b>Приветствие:</b> {intro_status}\n"
        f"<b>Кнопка действия:</b> {btn_status}\n"
        f"<b>Файлов БЗ (RAG):</b> {kb_files_count}\n"
        f"<b>Медиа-файлов (аудио/фото):</b> {media_count}\n"
        f"<b>Коллекции:</b> {colls_info}"
    )

    try:
        await bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=keyboards.edit_topic_keyboard(topic_id, topic.is_active, topic.show_in_main_menu,
                                                       topic.show_in_list, admin_only),
            parse_mode='HTML'
        )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("download_topic_prompt_"))
async def download_topic_prompt(callback: CallbackQuery):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)

    if not topic or not topic.system_prompt:
        await callback.answer("У этой темы нет собственного промпта.", show_alert=True)
        return

    prompt_text = topic.system_prompt
    filename = f"topic_{topic.id}_{topic.name}_prompt.txt"

    file_bytes = prompt_text.encode('utf-8')
    file_to_send = BufferedInputFile(
        file_bytes,
        filename=filename
    )
    await callback.message.answer_document(
        file_to_send,
        caption=f"📄 Системный промпт для темы «{topic.name}»."
    )
    await callback.answer()


@router.callback_query(F.data.startswith("reset_topic_prompt_"))
async def reset_topic_prompt(callback: CallbackQuery, state: FSMContext, bot: Bot):
    topic_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        stmt = update(Topic).where(Topic.id == topic_id).values(system_prompt=None)
        await session.execute(stmt)
        await session.commit()

    await state.clear()
    await callback.answer("✅ Промпт сброшен. Тема будет использовать общий промпт.", show_alert=True)
    await show_edit_topic_menu(callback, topic_id)


@router.callback_query(F.data.startswith("topic_random_phrases_"))
async def admin_topic_random_phrases_menu(callback: CallbackQuery, state: FSMContext):
    topic_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        count = await session.scalar(
            select(func.count(RandomMessage.id)).where(RandomMessage.topic_id == topic_id)
        )
        topic = await session.get(Topic, topic_id)

    text = (
        f"🎲 <b>Случайные фразы для темы: «{html.escape(topic.name)}»</b>\n\n"
        f"Загружено фраз: <b>{count}</b>\n\n"
        "Вы можете загрузить файл .txt (каждая фраза с новой строки). "
        "Бот будет выбирать одну случайную фразу из этого списка и добавлять её в контекст к каждому сообщению пользователя в этой теме."
    )

    builder = InlineKeyboardBuilder()
    builder.button(text="📥 Загрузить файл фраз (.txt)", callback_data=f"upload_topic_phrases_{topic_id}")
    builder.button(text="🗑️ Очистить все фразы", callback_data=f"clear_topic_phrases_{topic_id}")
    builder.button(text="⬅️ Назад к теме", callback_data=f"edit_topic_{topic_id}")
    builder.adjust(1)

    await callback.message.edit_text(text, reply_markup=builder.as_markup())


@router.callback_query(F.data.startswith("upload_topic_phrases_"))
async def admin_upload_topic_phrases_start(callback: CallbackQuery, state: FSMContext):
    topic_id = int(callback.data.split("_")[-1])

    await state.set_state(TopicPhrasesState.waiting_for_file)
    await state.update_data(topic_id=topic_id)

    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ Отмена", callback_data=f"topic_random_phrases_{topic_id}")

    await callback.message.edit_text(
        "Отправьте файл <b>.txt</b> со списком фраз (каждая с новой строки).\n\n"
        "Эти фразы будут привязаны ТОЛЬКО к текущей теме.",
        reply_markup=builder.as_markup()
    )


@router.message(TopicPhrasesState.waiting_for_file, F.document)
async def process_topic_phrases_file(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    topic_id = data.get('topic_id')
    document = message.document

    if not document.file_name.lower().endswith('.txt'):
        await message.answer("❌ Пожалуйста, отправьте файл .txt")
        return

    try:
        file_info = await bot.get_file(document.file_id)
        file_bytes_io = await bot.download_file(file_info.file_path)
        content = file_bytes_io.read().decode('utf-8')

        lines = [line.strip() for line in content.split('\n') if line.strip()]

        if not lines:
            await message.answer("❌ Файл пуст.")
            return

        async with async_session_maker() as session:

            for line in lines:
                session.add(RandomMessage(content=line, topic_id=topic_id))
            await session.commit()

        await message.answer(f"✅ Успешно добавлено {len(lines)} фраз в базу рандомайзера.")

        async with async_session_maker() as session:
            count = await session.scalar(
                select(func.count(RandomMessage.id)).where(RandomMessage.topic_id == topic_id)
            )
            topic = await session.get(Topic, topic_id)

        text = (
            f"🎲 <b>Случайные фразы для темы: «{html.escape(topic.name)}»</b>\n\n"
            f"Загружено фраз: <b>{count}</b>"
        )
        builder = InlineKeyboardBuilder()
        builder.button(text="📥 Загрузить файл фраз (.txt)", callback_data=f"upload_topic_phrases_{topic_id}")
        builder.button(text="🗑️ Очистить все фразы", callback_data=f"clear_topic_phrases_{topic_id}")
        builder.button(text="⬅️ Назад к теме", callback_data=f"edit_topic_{topic_id}")
        builder.adjust(1)

        await message.answer(text, reply_markup=builder.as_markup())
        await state.clear()

    except Exception as e:
        await message.answer(f"❌ Ошибка обработки файла: {e}")
        await state.clear()


@router.callback_query(F.data.startswith("clear_topic_phrases_"))
async def admin_clear_topic_phrases(callback: CallbackQuery):
    topic_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        await session.execute(delete(RandomMessage).where(RandomMessage.topic_id == topic_id))
        await session.commit()

    await callback.answer("✅ Все фразы для этой темы удалены.", show_alert=True)
    await admin_topic_random_phrases_menu(callback, None)


@router.callback_query(F.data.startswith("mailing_history_page_"))
async def admin_mailing_history(callback: CallbackQuery):
    page = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        history_filter = or_(Mailing.recurring_type.is_(None), Mailing.recurring_type != BIRTHDAY_MAILING_TYPE)
        total_mailings = await session.scalar(select(func.count(Mailing.id)).where(history_filter))

        if total_mailings == 0:
            text_to_send = "📜 История рассылок пуста."
            markup_to_send = kb.mailing_history_keyboard([], 0, 0)
        else:
            total_pages = math.ceil(total_mailings / PAGE_SIZE)
            page = max(0, min(page, total_pages - 1))

            mailings_result = await session.execute(
                select(Mailing)
                .where(history_filter)
                .order_by(Mailing.created_at.desc())
                .offset(page * PAGE_SIZE)
                .limit(PAGE_SIZE)
            )
            mailings = mailings_result.scalars().all()

            text_to_send = f"📜 История рассылок (стр. {page + 1}/{total_pages})"
            markup_to_send = kb.mailing_history_keyboard(mailings, page, total_pages)

    try:
        await callback.message.edit_text(
            text_to_send,
            reply_markup=markup_to_send
        )
    except TelegramBadRequest:
        try:
            await callback.message.delete()
        except TelegramBadRequest:
            pass
        await callback.bot.send_message(
            callback.from_user.id,
            text_to_send,
            reply_markup=markup_to_send
        )

    await callback.answer()


@router.callback_query(F.data.startswith("mailing_details_"))
async def admin_mailing_details(callback: CallbackQuery, bot: Bot):
    mailing_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)

    if not mailing:
        await callback.answer("Рассылка не найдена.", show_alert=True)
        return

    audience_name = get_mailing_audience_label(mailing.target_audience)
    status = get_mailing_status_label(mailing)
    start_time_str = to_msk(mailing.start_time).strftime('%d.%m.%Y %H:%M') if mailing.start_time else "Еще не запускалась"
    end_time_str = to_msk(mailing.end_time).strftime('%d.%m.%Y %H:%M') if mailing.end_time else "N/A"
    mailing_type = "Автоматическая ДР-рассылка" if is_birthday_mailing(mailing) else "Обычная рассылка"

    if is_birthday_mailing(mailing):
        text = (
            "<b>🎂 Шаблон ДР-рассылки</b>\n\n"
            f"<b>Статус:</b> {status}\n"
            f"<b>Создан:</b> {to_msk(mailing.created_at).strftime('%d.%m.%Y %H:%M')}\n"
            f"<b>Отправлено успешно:</b> {mailing.success_count}\n"
            f"<b>Ошибок:</b> {mailing.failure_count}\n\n"
            f"<b><u>Текст:</u></b>\n{mailing.text or 'Нет текста'}"
        )
    else:
        text = (
            f"<b>Детали рассылки #{mailing.id}</b>\n\n"
            f"<b>Тип:</b> {mailing_type}\n"
            f"<b>Аудитория:</b> {audience_name}\n"
            f"<b>Статус:</b> {status}\n"
            f"<b>Начало:</b> {start_time_str}\n"
            f"<b>Конец:</b> {end_time_str}\n"
            f"<b>Успешно:</b> {mailing.success_count}\n"
            f"<b>Ошибки:</b> {mailing.failure_count}\n\n"
            f"<b><u>Текст:</u></b>\n{mailing.text or 'Нет текста'}"
        )
    if is_birthday_mailing(mailing):
        text += f"\n\n<b>Переменные:</b> <code>{html.escape(BIRTHDAY_PLACEHOLDER_HINT)}</code>"

    back_keyboard = kb.mailing_details_keyboard(mailing)

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass

    try:
        if mailing.media_file_id:
            media_id = mailing.media_file_id
            if len(text) <= 1024:
                if mailing.media_file_type == 'photo':
                    await bot.send_photo(callback.from_user.id, media_id, caption=text, reply_markup=back_keyboard, parse_mode='HTML')
                else:
                    await bot.send_video(callback.from_user.id, media_id, caption=text, reply_markup=back_keyboard, parse_mode='HTML')
            else:
                if mailing.media_file_type == 'photo':
                    await bot.send_photo(callback.from_user.id, media_id)
                else:
                    await bot.send_video(callback.from_user.id, media_id)
                await bot.send_message(callback.from_user.id, text, reply_markup=back_keyboard, parse_mode='HTML')
        else:
            await bot.send_message(callback.from_user.id, text, reply_markup=back_keyboard, parse_mode='HTML')
    except Exception as e:
        logging.error(f"Error showing mailing details: {e}")
        await bot.send_message(callback.from_user.id, text, reply_markup=back_keyboard, parse_mode='HTML')

    answer_method = getattr(callback, "answer", None)
    if callable(answer_method):
        await answer_method()


@router.callback_query(lambda c: c.data and c.data.startswith("mailing_delete_birthday_") and not c.data.startswith("mailing_delete_birthday_confirm_"))
async def admin_delete_birthday_template_prompt(callback: CallbackQuery):
    mailing_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)
        if not mailing or not is_birthday_mailing(mailing):
            await callback.answer("Шаблон не найден.", show_alert=True)
            return

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass

    await callback.bot.send_message(
        callback.from_user.id,
        "🗑 Удалить шаблон ДР-рассылки?\n\nЭто удалит текущий шаблон и старые дубли ДР-шаблонов, если они есть.",
        reply_markup=kb.birthday_template_delete_keyboard(mailing_id),
    )
    answer_method = getattr(callback, "answer", None)
    if callable(answer_method):
        await answer_method()


@router.callback_query(F.data.startswith("mailing_delete_birthday_confirm_"))
async def admin_delete_birthday_template_confirm(callback: CallbackQuery, state: FSMContext, bot: Bot):
    async with async_session_maker() as session:
        birthday_ids = (
            await session.execute(
                select(Mailing.id).where(Mailing.recurring_type == BIRTHDAY_MAILING_TYPE)
            )
        ).scalars().all()

        if birthday_ids:
            await session.execute(delete(MailingDeliveryLog).where(MailingDeliveryLog.mailing_id.in_(birthday_ids)))
            await session.execute(delete(Mailing).where(Mailing.id.in_(birthday_ids)))
            await session.commit()

    await state.clear()
    await callback.answer("Шаблон ДР удален.", show_alert=True)
    await _show_birthday_template_screen(callback.message, bot)


@router.callback_query(F.data.startswith("mailing_toggle_enabled_"))
async def admin_toggle_mailing_enabled(callback: CallbackQuery, bot: Bot):
    mailing_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)
        if not mailing or not is_birthday_mailing(mailing):
            await callback.answer("Шаблон не найден.", show_alert=True)
            return

        mailing.is_enabled = not mailing.is_enabled
        await session.commit()

    await admin_mailing_details(callback, bot)


@router.callback_query(F.data.startswith("mailing_send_test_"))
async def admin_send_test_birthday_mailing(callback: CallbackQuery, bot: Bot):
    mailing_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        mailing = await session.get(Mailing, mailing_id)
        if not mailing or not is_birthday_mailing(mailing):
            await callback.answer("Шаблон не найден.", show_alert=True)
            return

        db_user = await session.get(User, callback.from_user.id)

    preview_user = db_user or SimpleNamespace(
        name=callback.from_user.first_name,
        first_name=callback.from_user.first_name,
        username=callback.from_user.username,
        birth_day=None,
        birth_month=None,
        birth_year=None,
    )

    try:
        rendered_text = render_mailing_text(mailing.text, preview_user)
        await send_mailing_content(bot, callback.from_user.id, mailing, rendered_text=rendered_text)
        await callback.answer("Тест отправлен вам в личку", show_alert=True)
    except Exception as exc:
        log.error(
            "Birthday mailing self-test failed mailing_id=%s admin_id=%s error=%s",
            mailing_id,
            callback.from_user.id,
            exc,
        )
        await callback.answer("Не удалось отправить тест", show_alert=True)


async def _show_admin_manage_buttons(bot: Bot, chat_id: int, message_id: int):
    async with async_session_maker() as session:
        stmt = select(Content).where(
            Content.button_title != None,
            Content.key.not_in(['test_intro', 'secret_test_outro', 'disclaimer', 'test_results', 'test_button'])
        )
        result = await session.execute(stmt)
        buttons = result.scalars().all()

        config = await session.get(SubscriptionConfig, 1)
        change_name_status = config.change_name_button_enabled if config else True

    bot_info = await bot.get_me()

    text = (
        "🎛️ <b>Управление кнопками главного меню</b>\n\n"
        "Здесь можно включить/выключить отображение кнопок, изменить их названия и порядок.\n\n"
        "🔗 <b>Ссылки на разделы (нажмите на ссылку или ID, чтобы скопировать):</b>\n\n"
    )

    if buttons:
        for btn in buttons:
            status = "🟢" if btn.is_visible else "⚪️"
            link = f"https://t.me/{bot_info.username}?start={btn.key}"
            text += (
                f"{status} <b>{btn.button_title}</b>\n"
                f"ID: <code>{btn.key}</code>\n"
                f"Ссылка: <code>{link}</code>\n\n"
            )
    else:
        text += "<i>Пользовательских кнопок пока нет.</i>"

    try:
        await bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=kb.manage_buttons_keyboard(buttons, change_name_status),
            parse_mode="HTML"
        )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("pay_robokassa_"))
async def create_robokassa_invoice(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    plan_id = int(parts[2])

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        plan = await session.get(SubscriptionPlan, plan_id)
        user = await session.get(
            User, callback.from_user.id,
            options=[
                selectinload(User.subscription),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )

    if not plan:
        await callback.answer("Тариф не найден.", show_alert=True)
        return

    user_sub = user.subscription if user else None
    user_promos = user.promo_codes if user else []
    discount_percent = user_sub.discount_percent if user_sub else 0

    plan_specific_promo = next((
        p for p in user_promos
        if not p.applies_to_all_plans and any(ap.id == plan_id for ap in p.applicable_plans)
    ), None)
    if plan_specific_promo:
        discount_percent = plan_specific_promo.discount_percent
    elif discount_percent == 0:
        all_plans_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
        if all_plans_promo:
            discount_percent = all_plans_promo.discount_percent

    price = plan.price
    if discount_percent > 0 and not plan.is_trial:
        price *= (1 - discount_percent / 100)
    price_decimal = decimal.Decimal(f"{price:.2f}")

    if not config.robokassa_merchant_login or not config.robokassa_password_1:
        await callback.message.edit_text(
            "❌ Платёжная система Robokassa временно недоступна. Администратор не настроил ключи API.")
        return

    if price < 1.0:
        await callback.answer(
            "❌ Сумма к оплате меньше 1 руб. — Robokassa не принимает такие платежи. Обратитесь к администратору.",
            show_alert=True)
        return

    promo_code_str = None
    if user_sub and user_sub.discount_percent > 0:
        promo_code_str = f"discount_{user_sub.discount_percent}"

    async with async_session_maker() as session:
        new_payment = RobokassaPayment(
            user_id=callback.from_user.id,
            plan_id=plan_id,
            promo_code=promo_code_str,
            amount=price,
            expires_at=datetime.utcnow() + ROBOKASSA_INVOICE_LIFETIME
        )
        session.add(new_payment)
        await session.commit()
        await session.refresh(new_payment)
        inv_id = new_payment.id

    description = ''.join(c for c in f"Оплата подписки на тариф «{plan.name}»" if ord(c) <= 0xFFFF)

    plan_allows_renewal = getattr(plan, 'allow_auto_renewal', True) if plan else True

    local_payment_url = build_local_robokassa_redirect_url(
        payment_id=inv_id,
        user_id=callback.from_user.id,
        password=config.robokassa_password_1
    )
    expires_at_msk = new_payment.expires_at.astimezone(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M МСК')
    user_ref = f"{callback.from_user.first_name or ''}"
    if callback.from_user.username:
        user_ref += f" (@{callback.from_user.username})"
    user_ref += f" [id=<code>{callback.from_user.id}</code>]"
    plog.info(
        f"СЧЕТ_СОЗДАН | Robokassa | {user_ref} | {plan.name} | {price:.2f} руб | "
        f"InvId={inv_id} | до {expires_at_msk}"
    )

    privacy_url = config.privacy_policy_url or "#"
    offer_url = config.offer_agreement_url or "#"

    if plan_allows_renewal:
        consent_line = f"Нажимая «Оплатить», я даю согласие на <a href='{privacy_url}'>обработку персональных данных</a> и принимаю <a href='{offer_url}'>договор оферты</a>."
    else:
        consent_line = f"Нажимая «Оплатить», я даю согласие на <a href='{privacy_url}'>обработку персональных данных</a> и принимаю <a href='{offer_url}'>договор оферты</a>."

    payment_type_line = (
        "Регулярная оплата, можно отключить в любой момент"
        if plan_allows_renewal
        else "Разовая оплата"
    )

    text = (
        "Ваша ссылка на оплату готова.\n\n"
        f"{consent_line}\n\n"
        f"<b>Сумма:</b> {price:.2f} руб.\n"
        f"{payment_type_line}\n"
        f"<b>Счёт действует до:</b> {expires_at_msk}\n\n"
        "Если срок действия истечёт, по кнопке ниже автоматически откроется новый счёт."
    )

    await callback.message.edit_text(
        text,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💳 Оплатить через Robokassa", url=local_payment_url)],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data=f"sub_pay_{plan_id}_{price}")]
        ]),
        disable_web_page_preview=True
    )
    await callback.answer()


@router.message(F.voice, StateFilter(None))
async def handle_voice_message(message: Message, state: FSMContext, bot: Bot):
    user_id = message.from_user.id

    async with async_session_maker() as session:
        ai_config = await session.get(AIConfig, 1)
        if not ai_config:
            ai_config = AIConfig()

        max_duration_sec = ai_config.max_voice_duration_sec
        transcription_provider = ai_config.transcription_provider

    if transcription_provider == 'None':
        await message.answer(
            "Извините, но распознавание голосовых сообщений в данный момент отключено администратором.")
        return

    if message.voice.duration > max_duration_sec:
        max_duration_minutes = max_duration_sec / 60
        if max_duration_minutes.is_integer():
            minutes_str = f"{int(max_duration_minutes)}"
        else:
            minutes_str = f"{max_duration_minutes:.1f}"

        await message.answer(
            f"К сожалению, слишком длинное голосовое сообщение ({message.voice.duration} сек.).\n"
            f"Попробуйте ещё раз, максимум до {minutes_str} минут(ы)."
        )
        return

    thinking_msg = await message.answer("🤖 Распознаю ваше голосовое сообщение...")

    try:
        file_info = await bot.get_file(message.voice.file_id)
        file_bytes_io = await bot.download_file(file_info.file_path)
        file_bytes = file_bytes_io.read()

        filename = f"{message.voice.file_id}.ogg"

        prompt_text = await transcribe_voice_message(file_bytes, filename)

        if prompt_text.startswith("❌ Ошибка:"):
            await thinking_msg.edit_text(prompt_text)
            return

        try:
            escaped_prompt = html.escape(prompt_text)
            chunks = split_message(escaped_prompt, 4000)
            italicized_chunks = [f"<i>{chunk}</i>" for chunk in chunks]

            first_chunk_to_send = italicized_chunks[0]
            if len(first_chunk_to_send) > 4096:
                first_chunk_to_send = f"<i>{chunks[0][:4000]}</i>"

            await thinking_msg.edit_text(first_chunk_to_send)

            if len(italicized_chunks) > 1:
                for chunk in italicized_chunks[1:]:
                    await _safe_send_html(
                        lambda text, pm: message.answer(text, parse_mode=pm),
                        chunk,
                    )
                    await asyncio.sleep(0.3)

        except TelegramBadRequest as e:
            await thinking_msg.delete()
            logging.warning(f"Failed to edit transcription text: {e}. Sending as new messages.")
            escaped_prompt = html.escape(prompt_text)
            chunks = split_message(escaped_prompt, 4000)
            italicized_chunks = [f"<i>{chunk}</i>" for chunk in chunks]

            for chunk in italicized_chunks:
                await _safe_send_html(
                    lambda text, pm: message.answer(text, parse_mode=pm),
                    chunk,
                )
                await asyncio.sleep(0.3)

    except InsufficientBalanceError as e:
        provider, model = _resolve_ai_provider_model(ai_config, "transcription")
        await _report_ai_failure(
            bot,
            title="Критическая ошибка API транскрибации",
            user=message.from_user,
            provider=provider,
            model=model,
            stage="voice_transcription_balance",
            details=str(e),
            extra={"duration_sec": message.voice.duration},
            exception=e,
        )
        await thinking_msg.edit_text(
            "К сожалению, сервис транскрибации временно недоступен из-за технической проблемы. Мы уже работаем над ее решением.")
        return
    except AIServiceError as e:
        provider, model = _resolve_ai_provider_model(ai_config, "transcription")
        await _report_ai_failure(
            bot,
            title="Сбой транскрибации",
            user=message.from_user,
            provider=provider,
            model=model,
            stage="voice_transcription",
            details=str(e),
            extra={"duration_sec": message.voice.duration},
            exception=e,
        )
        await thinking_msg.edit_text(
            "Упс... У нас что-то сломалось. Мы уже сообщили нашим создателям. Попробуйте вернуться и повторить через несколько минут."
        )
        return
    except Exception as e:
        provider, model = _resolve_ai_provider_model(ai_config, "transcription")
        await _report_ai_failure(
            bot,
            title="Непредвиденная ошибка обработки голосового",
            user=message.from_user,
            provider=provider,
            model=model,
            stage="voice_handler_unexpected",
            details=str(e),
            extra={"duration_sec": message.voice.duration},
            exception=e,
        )
        await thinking_msg.edit_text(
            f"Произошла непредвиденная ошибка при обработке аудио.\n<code>{html.escape(str(e))}</code>")
        return

    async with async_session_maker() as session:
        user = await session.get(User, user_id, options=[selectinload(User.subscription)])

        if not user:
            user = User(id=user_id, username=message.from_user.username, first_name=message.from_user.full_name,
                        is_admin=await is_admin(user_id))
            session.add(user)
            await session.flush()
        else:
            user.username = message.from_user.username
            user.first_name = message.from_user.full_name

        await _sync_user_birthdate_from_telegram(bot, user)
        await session.commit()
        await session.refresh(user, ['subscription'])

        sub_config = await session.get(SubscriptionConfig, 1)
        subscriptions_active = sub_config.subscriptions_enabled if sub_config else True

        is_user_admin = await is_admin(user_id)
        if not is_user_admin and subscriptions_active:
            if not user.subscription or user.subscription.end_date < datetime.utcnow():
                await thinking_msg.delete()
                await message.answer(
                    "Чтобы продолжить диалог, активируйте подписку / бонусные дни.",
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(text="Начать пользоваться ботом",
                                              callback_data="show_subscription_info_from_chat")]
                    ])
                )
                return

        if not user.name:
            await state.set_state(UserStates.awaiting_name)
            await state.update_data(initial_prompt=prompt_text)
            await message.answer("Прежде чем мы начнем, подскажите, как я могу к вам обращаться?")
            return

        if not user.accepted_disclaimer:
            disclaimer_content = await get_content_from_db("disclaimer")

            if disclaimer_content.get('is_visible', True):
                await state.set_state(UserStates.awaiting_disclaimer_acceptance)
                await state.update_data(initial_prompt=prompt_text)
                text_to_send = disclaimer_content.get('text') or "Текст дисклеймера не задан."
                await message.answer(text_to_send, reply_markup=kb.confirm_disclaimer_keyboard())
                return
            else:
                stmt = update(User).where(User.id == user_id).values(accepted_disclaimer=True)
                await session.execute(stmt)
                await session.commit()

    await process_user_prompt(message, user_id, prompt_text, bot)


@router.callback_query(F.data == "set_audio_limit")
async def set_audio_limit_start(callback: CallbackQuery, state: FSMContext):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        current_limit = config.max_voice_duration_sec if config else 180

    await state.set_state(AdminStates.set_voice_limit)
    await state.update_data(message_id_to_edit=callback.message.message_id)

    current_limit_min = current_limit / 60
    if current_limit_min.is_integer():
        minutes_str = f"{int(current_limit_min)}"
    else:
        minutes_str = f"{current_limit_min:.1f}"

    await callback.message.edit_text(
        f"⏱️ <b>Лимит длительности аудио</b>\n\n"
        f"Текущий лимит: <b>{current_limit} сек.</b> ({minutes_str} мин.)\n\n"
        f"Введите новое значение в секундах (например, `180` для 3 минут):",
        reply_markup=kb.back_to_previous_menu("admin_ai_keys")
    )


@router.message(AdminStates.set_voice_limit, F.text)
async def process_audio_limit(message: Message, state: FSMContext, bot: Bot):
    try:
        new_limit_sec = int(message.text)
        if new_limit_sec <= 0:
            raise ValueError("Limit must be positive")

    except ValueError:
        await message.delete()
        temp_msg = await message.answer("❌ Ошибка: Введите корректное целое число (например, `180`).")
        await delete_message_after_delay(temp_msg, 4)
        return

    data = await state.get_data()
    message_id_to_edit = data.get('message_id_to_edit')

    transcription_provider = 'OpenAI'
    config = None
    async with async_session_maker() as session:
        stmt = update(AIConfig).where(AIConfig.id == 1).values(max_voice_duration_sec=new_limit_sec)
        await session.execute(stmt)
        await session.commit()

        config = await session.get(AIConfig, 1)
        if config:
            transcription_provider = config.transcription_provider

    await state.clear()
    await message.delete()

    if message_id_to_edit:
        try:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=message_id_to_edit,
                text="🔑 Настройка ключей и моделей API",
                reply_markup=kb.ai_keys_models_keyboard(
                    transcription_provider,
                    config.context_limit_first if config else 2,
                    config.context_limit_recent if config else 10,
                    config.vision_provider if config else 'Gemini',
                    config.vision_model if config else 'gemini-3-flash-preview',
                    getattr(config, 'temperature', 0.7) if config else 0.7,
                    get_memory_mode(config) if config else MEMORY_MODE_RESET
                )
            )
        except TelegramBadRequest:
            pass

    temp_msg = await message.answer(f"✅ Лимит аудио обновлен: {new_limit_sec} сек.")
    await delete_message_after_delay(temp_msg, 3)


async def check_history_permission(user_id: int) -> bool:
    if user_id in OWNER_IDS:
        return True
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        return user and user.is_admin and user.can_view_history


def _client_back_to_list_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⬅️ К списку клиентов", callback_data="admin_clients_page_0")]]
    )


def _calculate_effective_discount(user_sub: UserSubscription | None, user_promos: list[PromoCode], plan_id: int | None) -> int:
    discount_percent = user_sub.discount_percent if user_sub else 0
    if not plan_id:
        return discount_percent

    plan_specific_promo = next((
        p for p in user_promos
        if not p.applies_to_all_plans and any(ap.id == plan_id for ap in p.applicable_plans)
    ), None)
    if plan_specific_promo and plan_specific_promo.discount_percent > discount_percent:
        return plan_specific_promo.discount_percent

    all_plans_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
    if all_plans_promo and all_plans_promo.discount_percent > discount_percent:
        return all_plans_promo.discount_percent

    return discount_percent


async def _build_client_payment_info_text(user_id: int) -> str:
    now = datetime.utcnow()
    msk = timezone(timedelta(hours=3))

    async with async_session_maker() as session:
        user = await session.get(
            User,
            user_id,
            options=[
                selectinload(User.subscription).selectinload(UserSubscription.plan),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)
            ]
        )
        if not user:
            return "Клиент не найден."

        user_sub = user.subscription
        user_promos = list(user.promo_codes or [])

        robo_paid = await session.scalar(
            select(func.coalesce(func.sum(RobokassaPayment.amount), 0.0)).where(
                RobokassaPayment.user_id == user_id,
                RobokassaPayment.status == 'completed'
            )
        ) or 0.0
        yk_paid = await session.scalar(
            select(func.coalesce(func.sum(YookassaPayment.amount), 0.0)).where(
                YookassaPayment.user_id == user_id,
                YookassaPayment.status.in_(['completed', 'succeeded'])
            )
        ) or 0.0

    active_access = bool(user_sub and user_sub.end_date and user_sub.end_date > now)
    active_paid = bool(active_access and user_sub.plan_id is not None)
    active_bonus = bool(active_access and user_sub.plan_id is None)
    active_referral_bonus = bool(active_bonus and user_sub.payment_provider in ['Trial Referral', 'Trial Referral Bonus'])
    active_promo_bonus = bool(active_bonus and user_sub.payment_provider == 'Trial Promo')
    active_welcome_bonus = bool(active_bonus and user_sub.payment_provider == 'Trial Welcome')
    active_discount = bool((user_sub.discount_percent if user_sub else 0) > 0)

    text_lines = [
        "<b>💳 Платежная информация клиента</b>",
        "",
        f"<b>ID:</b> <code>{user_id}</code>",
        f"<b>Статус доступа:</b> {'✅ Активен' if (active_access or active_discount) else '❌ Неактивен'}",
    ]

    reasons = []
    if active_paid:
        reasons.append("оплаченный тариф")
    if active_referral_bonus:
        reasons.append("реферальный бонус")
    if active_promo_bonus:
        reasons.append("промо-бонус")
    if active_welcome_bonus:
        reasons.append("приветственный бонус")
    if active_discount:
        reasons.append(f"скидка {user_sub.discount_percent}%")
    if reasons:
        text_lines.append(f"<b>Основание:</b> {', '.join(reasons)}")

    if user_sub and user_sub.end_date:
        text_lines.append(f"<b>Доступ до:</b> {user_sub.end_date.astimezone(msk).strftime('%d.%m.%Y %H:%M')} МСК")

    text_lines.append("")
    text_lines.append("<b>Текущий тариф:</b>")
    if active_paid and user_sub and user_sub.plan:
        plan = user_sub.plan
        effective_discount = _calculate_effective_discount(user_sub, user_promos, plan.id)
        final_price = plan.price * (1 - effective_discount / 100)
        duration_unit = "дн." if plan.duration_unit == 'days' else "мес."
        text_lines.append(f"• {html.escape(plan.name)} ({plan.duration_value} {duration_unit})")
        text_lines.append(f"• Автопродление: {'✅ Включено' if user_sub.auto_renewal else '❌ Выключено'}")
        text_lines.append(f"• Стоимость со скидкой: {final_price:.2f} руб.")
    else:
        text_lines.append("• Активного платного тарифа нет")

    text_lines.append("")
    text_lines.append("<b>Промокоды:</b>")
    if user_promos:
        for promo in user_promos:
            scope_text = "все тарифы" if promo.applies_to_all_plans else f"{len(promo.applicable_plans)} тариф(ов)"
            text_lines.append(
                f"• {html.escape(promo.code)} | скидка {promo.discount_percent}% | дней {promo.free_days} | область: {scope_text}"
            )
    else:
        text_lines.append("• Не активировались")

    if active_promo_bonus and user_sub and user_sub.end_date:
        promo_candidates = [p.code for p in user_promos if p.free_days > 0]
        promo_name = promo_candidates[0] if len(promo_candidates) == 1 else "не удалось определить точно"
        text_lines.append(
            f"• Активный промо-бонус: {html.escape(promo_name)} до {user_sub.end_date.astimezone(msk).strftime('%d.%m.%Y %H:%M')} МСК"
        )
    elif active_discount:
        text_lines.append("• Активная скидка сохранена в аккаунте")
        if active_bonus and user_sub and user_sub.end_date:
            text_lines.append(f"• Скидка действует до конца бонусного периода: {user_sub.end_date.astimezone(msk).strftime('%d.%m.%Y %H:%M')} МСК")
        else:
            text_lines.append("• Точный срок скидочного промокода в текущей схеме БД отдельно не хранится")
    else:
        text_lines.append("• Активного промокода сейчас нет")

    text_lines.append("")
    text_lines.append("<b>Полученные оплаты:</b>")
    text_lines.append(f"• YooKassa: {yk_paid:.2f} руб.")
    text_lines.append(f"• Robokassa: {robo_paid:.2f} руб.")
    text_lines.append(f"• Итого: {yk_paid + robo_paid:.2f} руб.")

    return "\n".join(text_lines)


@router.callback_query(F.data.startswith("toggle_plan_is_trial_"))
async def admin_toggle_plan_is_trial(callback: CallbackQuery):
    plan_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.is_trial = not plan.is_trial
            if not plan.is_trial:
                plan.upgrades_to_plan_id = None
            await session.commit()

    await _show_admin_edit_plan_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        plan_id=plan_id
    )


@router.callback_query(F.data.startswith("set_plan_upgrade_target_"))
async def admin_set_plan_upgrade_target_start(callback: CallbackQuery, state: FSMContext):
    plan_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        plans = (await session.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.is_trial == False, SubscriptionPlan.id != plan_id)
        )).scalars().all()

    if not plans:
        await callback.answer("Нет доступных тарифов для перехода. Сначала создайте обычный (не-триальный) тариф.",
                              show_alert=True)
        return

    await state.set_state(AdminStates.set_plan_upgrade_target)
    await state.update_data(plan_id=plan_id, message_id_to_edit=callback.message.message_id)

    await callback.message.edit_text(
        "Выберите тариф, на который перейдет клиент после окончания пробного периода:",
        reply_markup=kb.admin_select_upgrade_plan_keyboard(plans, plan_id)
    )


@router.callback_query(F.data.startswith("set_upgrade_plan_"), AdminStates.set_plan_upgrade_target)
async def admin_set_plan_upgrade_target_process(callback: CallbackQuery, state: FSMContext):
    target_plan_id = int(callback.data.split("_")[-1])
    data = await state.get_data()
    plan_id = data['plan_id']
    message_id_to_edit = data['message_id_to_edit']

    async with async_session_maker() as session:
        await session.execute(
            update(SubscriptionPlan)
            .where(SubscriptionPlan.id == plan_id)
            .values(upgrades_to_plan_id=target_plan_id)
        )
        await session.commit()

    await state.clear()
    await _show_admin_edit_plan_menu(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=message_id_to_edit,
        plan_id=plan_id
    )
    await callback.answer("✅ Тариф для перехода назначен!")


async def _show_admin_manage_admins(bot: Bot, chat_id: int, message_id: int):
    async with async_session_maker() as session:
        admins_result = await session.execute(select(User).where(User.is_admin == True))
        admins = admins_result.scalars().all()

    text = "👮‍♂️ <b>Управление администраторами</b>\n\n"
    text += "👑 <b>Владельцы (из конфига):</b>\n"
    for owner_id in OWNER_IDS:
        try:
            chat = await bot.get_chat(owner_id)
            text += f" • {chat.full_name} (@{chat.username or 'N/A'}) - <code>{owner_id}</code>\n"
        except TelegramBadRequest:
            text += f" • Не удалось получить инфо - <code>{owner_id}</code>\n"

    text += "\n"
    text += "👮‍♂️ <b>Администраторы (из БД):</b>\n"
    db_admins = [a for a in admins if a.id not in OWNER_IDS]
    if not db_admins:
        text += "<i>Список пуст</i>"

    try:
        await bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=kb.admin_management_keyboard(db_admins)
        )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("reset_user_promos_"))
async def reset_user_promos(callback: CallbackQuery, state: FSMContext):
    user_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        user = await session.get(User, user_id,
                                 options=[selectinload(User.promo_codes), selectinload(User.subscription)])
        if not user:
            await callback.answer("Клиент не найден", show_alert=True)
            return

        user.promo_codes.clear()

        if user.subscription:
            user.subscription.discount_percent = 0

        await session.commit()

    await callback.answer(f"✅ Все промокоды и скидки для клиента {user_id} сброшены.", show_alert=True)
    await view_client_profile(callback, state)


@router.callback_query(
    lambda c: c.data
    and c.data.startswith("admin_reset_sub_")
    and not c.data.startswith("admin_reset_sub_confirm_")
)
async def prompt_reset_client_subscription(callback: CallbackQuery):
    user_id = int(callback.data.split("_")[-1])
    await callback.message.edit_text(
        f"Сбросить подписку клиента <code>{user_id}</code>?\n\n"
        "Будет удалена запись подписки (тариф, автопродление, привязка к платежу).\n"
        "Платёжные записи и остальные данные аккаунта сохранятся.",
        reply_markup=kb.confirm_client_action_keyboard(
            f"admin_reset_sub_confirm_{user_id}",
            f"view_client_{user_id}"
        ),
        parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("admin_reset_sub_confirm_"))
async def reset_client_subscription_confirm(callback: CallbackQuery, state: FSMContext):
    user_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        result = await session.execute(
            delete(UserSubscription).where(UserSubscription.user_id == user_id)
        )
        await session.commit()

    if result.rowcount:
        await callback.answer(f"✅ Подписка клиента {user_id} сброшена.", show_alert=True)
    else:
        await callback.answer(f"У клиента {user_id} не было подписки.", show_alert=True)
    await view_client_profile(callback, state)


@router.callback_query(F.data.startswith("client_payment_info_"))
async def view_client_payment_info(callback: CallbackQuery):
    user_id = int(callback.data.split("_")[-1])
    text = await _build_client_payment_info_text(user_id)
    await callback.message.edit_text(
        text,
        reply_markup=kb.back_to_client_profile(user_id),
        parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(
    lambda c: c.data
    and c.data.startswith("admin_delete_client_history_")
    and not c.data.startswith("admin_delete_client_history_confirm_")
)
async def prompt_delete_client_history(callback: CallbackQuery):
    user_id = int(callback.data.split("_")[-1])
    if not await check_history_permission(callback.from_user.id):
        await callback.answer("У вас нет прав на удаление истории.", show_alert=True)
        return

    await callback.message.edit_text(
        "Удалить всю историю диалогов этого клиента?\n\n"
        "Будут удалены все сообщения из БД. Профиль и платежные данные останутся.",
        reply_markup=kb.confirm_client_action_keyboard(
            f"admin_delete_client_history_confirm_{user_id}",
            f"view_client_{user_id}"
        )
    )
    await callback.answer()


@router.callback_query(F.data.startswith("admin_delete_client_history_confirm_"))
async def delete_client_history_confirm(callback: CallbackQuery, state: FSMContext):
    user_id = int(callback.data.split("_")[-1])
    if not await check_history_permission(callback.from_user.id):
        await callback.answer("У вас нет прав на удаление истории.", show_alert=True)
        return

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            await callback.answer("Клиент не найден.", show_alert=True)
            return

        await session.execute(delete(DBMessage).where(DBMessage.user_id == user_id))
        await session.execute(delete(UserTopicState).where(UserTopicState.user_id == user_id))
        user.current_dialogue_id += 1
        await session.commit()

    await callback.answer("✅ История клиента удалена.", show_alert=True)
    await view_client_profile(callback, state)


@router.callback_query(
    lambda c: c.data
    and c.data.startswith("admin_reset_client_")
    and not c.data.startswith("admin_reset_client_confirm_")
)
async def prompt_reset_client_account(callback: CallbackQuery):
    user_id = int(callback.data.split("_")[-1])
    if callback.from_user.id not in OWNER_IDS:
        await callback.answer("Только владелец может делать полный сброс аккаунта.", show_alert=True)
        return

    await callback.message.edit_text(
        "Полностью сбросить аккаунт этого клиента?\n\n"
        "Будут удалены профиль, история, подписка, бонусы, промокоды, платежные записи и реферальные связи.\n"
        "При следующем /start пользователь будет считаться новым.",
        reply_markup=kb.confirm_client_action_keyboard(
            f"admin_reset_client_confirm_{user_id}",
            f"view_client_{user_id}"
        )
    )
    await callback.answer()


@router.callback_query(F.data.startswith("admin_reset_client_confirm_"))
async def reset_client_account_confirm(callback: CallbackQuery):
    user_id = int(callback.data.split("_")[-1])
    if callback.from_user.id not in OWNER_IDS:
        await callback.answer("Только владелец может делать полный сброс аккаунта.", show_alert=True)
        return

    async with async_session_maker() as session:
        user = await session.get(User, user_id, options=[selectinload(User.promo_codes)])
        if not user:
            await callback.answer("Клиент не найден.", show_alert=True)
            return

        user.promo_codes.clear()
        await session.flush()

        await session.execute(delete(DBMessage).where(DBMessage.user_id == user_id))
        await session.execute(delete(UserSubscription).where(UserSubscription.user_id == user_id))
        await session.execute(delete(TrialUsageHistory).where(TrialUsageHistory.user_id == user_id))
        await session.execute(delete(TestSession).where(TestSession.user_id == user_id))
        await session.execute(delete(UserTopicState).where(UserTopicState.user_id == user_id))
        await session.execute(delete(ReferralPaymentLog).where(
            or_(
                ReferralPaymentLog.referrer_id == user_id,
                ReferralPaymentLog.referred_user_id == user_id
            )
        ))
        await session.execute(delete(RobokassaPayment).where(RobokassaPayment.user_id == user_id))
        await session.execute(delete(YookassaPayment).where(YookassaPayment.user_id == user_id))
        await session.execute(delete(IndexingQueue).where(IndexingQueue.uploader_id == user_id))
        await session.execute(update(User).where(User.referred_by == user_id).values(referred_by=None))
        await session.delete(user)
        await session.commit()

    await callback.message.edit_text(
        f"✅ Аккаунт пользователя <code>{user_id}</code> полностью сброшен.\n"
        "При следующем /start он будет создан заново.",
        reply_markup=_client_back_to_list_keyboard(),
        parse_mode="HTML"
    )
    await callback.answer()


def generate_progress_bar(current, total, length=10):
    percent = current / total
    filled_length = int(length * percent)
    bar = "█" * filled_length + "░" * (length - filled_length)
    return f"{bar} {int(percent * 100)}%"


async def start_psych_test(message: Message, state: FSMContext, user_id: int):
    async with async_session_maker() as session:
        questions_result = await session.execute(
            select(TestQuestion).order_by(TestQuestion.sort_order.asc())
        )
        questions = questions_result.scalars().all()

        if not questions:
            await message.answer("Ошибка: Вопросы теста не загружены. Обратитесь к администратору.")
            return

        await session.execute(delete(TestSession).where(TestSession.user_id == user_id))
        new_session = TestSession(user_id=user_id, current_question_index=0, answers="[]")
        session.add(new_session)
        await session.commit()

        question = questions[0]
        progress = generate_progress_bar(0, len(questions))

        legend = (
            "1 — Совершенно не согласен(на)\n"
            "2 — Скорее не согласен(на)\n"
            "3 — Нейтрален(на)\n"
            "4 — Скорее согласен(на)\n"
            "5 — Полностью согласен(на)"
        )

        text = (
            f"<b>Вопрос 1/{len(questions)}</b>\n{progress}\n\n"
            f"<b>{question.text}</b>\n\n"
            f"<i>{legend}</i>"
        )

    await state.set_state(UserStates.in_test)
    await message.answer(text, reply_markup=kb.test_answer_keyboard())


@router.callback_query(F.data.startswith("test_ans_"))
async def process_test_answer(callback: CallbackQuery, state: FSMContext, bot: Bot):
    user_id = callback.from_user.id
    answer_value = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id)
        if not test_session or test_session.is_finished:
            await callback.message.answer("Эта сессия уже завершена или не существует.")
            await callback.answer()
            return

        current_answers = test_session.answers
        if current_answers:
            test_session.answers = f"{current_answers},{answer_value}"
        else:
            test_session.answers = str(answer_value)

        test_session.current_question_index += 1

        total_questions = await session.scalar(select(func.count()).select_from(TestQuestion))

        await session.commit()

        if test_session.current_question_index < total_questions:
            await callback.message.delete()
            await send_next_question(callback.message, user_id, state, bot)
        else:
            await callback.message.delete()
            loading_msg = await callback.message.answer(
                "🤖 Спасибо! Анализирую твои ответы и подбираю похожий случай из практики...")

            all_questions = await session.execute(select(TestQuestion).order_by(TestQuestion.sort_order.asc()))
            questions_list = all_questions.scalars().all()
            user_answers_list = [int(a) for a in test_session.answers.split(",") if a]

            categories_stats = {}
            total_score = 0
            total_max = 0

            for q, ans in zip(questions_list, user_answers_list):
                cat = q.category if q.category else "Общее"
                if cat not in categories_stats:
                    categories_stats[cat] = {'score': 0, 'max': 0}

                final_score = ans
                if q.is_reverse:
                    final_score = 6 - ans

                categories_stats[cat]['score'] += final_score
                categories_stats[cat]['max'] += 5
                total_score += final_score
                total_max += 5

            report_lines = ["📊 <b>ТВОЯ КАРТА САМООЦЕНКИ</b>\n"]
            sorted_cats = sorted(categories_stats.keys())
            weak_points = []

            for cat in sorted_cats:
                data = categories_stats[cat]
                score = data['score']
                max_score = data['max']
                percentage = score / max_score if max_score > 0 else 0

                filled = int(percentage * 10)
                bar = "█" * filled + "░" * (10 - filled)

                status = ""
                if percentage >= 0.75:
                    status = "   ✅ Сильная сторона"
                elif percentage <= 0.45:
                    status = "   ⚠️ Зона внимания"
                    weak_points.append(CATEGORY_NAMES.get(cat, cat))

                cat_name = CATEGORY_NAMES.get(cat, cat)
                report_lines.append(f"<b>{cat_name}</b>")
                report_lines.append(f"{bar} {score}/{max_score} баллов{status}")

            total_percent = int((total_score / total_max) * 100) if total_max > 0 else 0
            report_lines.append("\n" + "─" * 15)
            report_lines.append(f"<b>ОБЩИЙ ИТОГ: {total_score}/{total_max} баллов ({total_percent}%)</b>")

            diagram_text = "\n".join(report_lines)

            async with async_session_maker() as update_session:
                stmt = update(TestSession).where(TestSession.user_id == user_id).values(answers=diagram_text,
                                                                                        is_finished=True)
                await update_session.execute(stmt)
                await update_session.commit()

            user = await session.get(User, user_id)
            user_name = user.name or "Незнакомец"
            user_age = user.age or "Не указан"
            user_gender_raw = user.gender or "unknown"
            user_gender_str = "Женский" if user_gender_raw == 'female' else "Мужской" if user_gender_raw == 'male' else "Не определен"

            search_query = f"Пол: {user_gender_str}, Возраст: {user_age}. Проблемы: {', '.join(weak_points)}"
            found_case_text = await search_relevant_case(search_query)

            context_case_instruction = ""
            if found_case_text:
                context_case_instruction = (
                    f"\n\n[РЕАЛЬНЫЙ КЕЙС ИЗ БАЗЫ ЗНАНИЙ]\n"
                    f"{found_case_text}\n"
                    f"ИНСТРУКЦИЯ: Используй этот кейс как основу для истории. Адаптируй его под пользователя."
                )
            else:
                context_case_instruction = "\n\n(Подходящий кейс в базе не найден. Придумай собирательный образ на основе проблемных зон пользователя)."

            user_prompt_data = (
                f"ВЫПОЛНИ ЗАДАЧУ 1: СЦЕНАРИСТ (ИСТОРИЯ-ЗЕРКАЛО).\n\n"
                f"ДАННЫЕ ПОЛЬЗОВАТЕЛЯ:\n"
                f"Имя: {user_name}\n"
                f"Возраст: {user_age}\n"
                f"Пол: {user_gender_str} (Важно: Персонаж истории должен быть этого же пола!)\n\n"
                f"РЕЗУЛЬТАТЫ ТЕСТА:\n{diagram_text}\n"
                f"{context_case_instruction}\n\n"
                f"Напиши историю, которая вызовет чувство 'Это же про меня!'. Не пиши цифры и баллы, только историю."
            )

            test_config = await session.get(TestConfig, 1)
            system_instruction = test_config.test_system_prompt

            case_study_text = await get_ai_response_direct(user_id, system_instruction, user_prompt_data)
            html_story = markdown_to_html(case_study_text)

            final_message_text = f"{html_story}\n\n<i>Присмотрись. Возможно, ты мог узнать себя в этой истории</i>"

            async with async_session_maker() as msg_session:
                msg_session.add(DBMessage(
                    user_id=user_id,
                    role='assistant',
                    content=case_study_text,
                    dialogue_id=user.current_dialogue_id,
                    topic_id=user.current_topic_id
                ))
                await msg_session.commit()

            await loading_msg.edit_text(
                final_message_text,
                reply_markup=kb.case_study_confirmation_keyboard()
            )

    await callback.answer()


@router.callback_query(UserStates.awaiting_gender, F.data.startswith("gender_"))
async def process_test_gender(callback: CallbackQuery, state: FSMContext, bot: Bot):
    gender_code = callback.data.split("_")[-1]
    user_id = callback.from_user.id

    async with async_session_maker() as session:
        stmt = update(User).where(User.id == user_id).values(gender=gender_code)
        await session.execute(stmt)
        await session.commit()
        user = await session.get(User, user_id)

    data = await state.get_data()
    await callback.answer()

    gender_label = "👨 Мужской" if gender_code == "male" else "👩 Женский"

    if data.get('is_settings'):
        await state.clear()
        async with async_session_maker() as session:
            user = await session.get(User, user_id)
        text = (
            f"✅ Пол изменён: {gender_label}\n\n"
            "⚙️ <b>Настройки</b>\n\n"
            f"<b>Имя:</b> {html.escape(user.name or user.first_name or 'Не указано')}\n"
            f"<b>Пол:</b> {'👨 Мужской' if user.gender == 'male' else ('👩 Женский' if user.gender == 'female' else '❓ Не указан')}\n"
            f"<b>Возраст:</b> {user.age or 'Не указан'}\n"
            f"<b>Длина ответов:</b> {'📏 Обычный' if getattr(user, 'response_length', 'normal') != 'short' else '📏 Короткий'}\n"
        )
        try:
            await callback.message.edit_text(text, reply_markup=kb.user_settings_keyboard(user))
        except Exception:
            pass
        return

    if data.get('is_name_change'):
        await state.clear()
        try:
            await callback.message.edit_text(f"Пол: {gender_label} ✅")
        except Exception:
            pass
        return

    if data.get('is_onboarding'):
        prompt_text = data.get('initial_prompt')
        await state.clear()
        try:
            await callback.message.edit_text(f"Пол: {gender_label} ✅")
        except Exception:
            pass

        if not user.accepted_disclaimer:
            disclaimer_content = await get_content_from_db("disclaimer")
            if disclaimer_content.get('is_visible', True):
                await state.set_state(UserStates.awaiting_disclaimer_acceptance)
                await state.update_data(initial_prompt=prompt_text)
                text_to_send = disclaimer_content.get('text') or "Текст дисклеймера не задан."
                await callback.message.answer(text_to_send, reply_markup=kb.confirm_disclaimer_keyboard())
                return
            else:
                async with async_session_maker() as session:
                    stmt = update(User).where(User.id == user_id).values(accepted_disclaimer=True)
                    await session.execute(stmt)
                    await session.commit()

        if prompt_text:
            await process_user_prompt(callback.message, user_id, prompt_text, bot)
        return

    await state.set_state(UserStates.awaiting_age)
    await callback.message.answer("А перед началом скажи: сколько тебе лет?")


async def send_next_question(message: Message, user_id: int, state: FSMContext, bot: Bot):
    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id)

        if not test_session:
            await message.answer("Ошибка: сессия теста не найдена. Попробуйте начать заново: /test")
            return

        user = await session.get(User, user_id)
        user_gender = user.gender

        stmt = select(TestQuestion).order_by(TestQuestion.sort_order.asc())
        result = await session.execute(stmt)
        questions = result.scalars().all()

    total_count = len(questions)
    current_index = test_session.current_question_index

    if current_index >= total_count:
        answers_list = []
        if test_session.answers:
            answers_list = [int(a) for a in test_session.answers.split(",") if a]
        await finish_test_generation(message, user_id, answers_list, questions)
        return

    question = questions[current_index]

    if user_gender == 'male':
        suffix = "ен"
        neutral_word = "Нейтрален"
    else:
        suffix = "на"
        neutral_word = "Нейтральна"

    legend_text = (
        f"1 — Совершенно не соглас{suffix}\n"
        f"2 — Скорее не соглас{suffix}\n"
        f"3 — {neutral_word}\n"
        f"4 — Скорее соглас{suffix}\n"
        f"5 — Полностью соглас{suffix}"
    )

    q_num = current_index + 1
    percent = int((current_index / total_count) * 100)

    filled_length = int(percent / 10)
    bar = "█" * filled_length + "░" * (10 - filled_length)

    progress_text = f"{bar} {percent}% ({q_num}/{total_count})"

    text = (
        f"<b>Вопрос {q_num} из {total_count}</b>\n"
        f"{progress_text}\n\n"
        f"<b>{question.text}</b>\n\n"
        f"{legend_text}"
    )

    try:
        await message.edit_text(text, reply_markup=kb.test_answer_keyboard())
    except TelegramBadRequest:
        await message.answer(text, reply_markup=kb.test_answer_keyboard())


@router.message(UserStates.awaiting_age)
async def process_test_age(message: Message, state: FSMContext, bot: Bot):
    if not message.text or not message.text.strip().isdigit():
        await message.answer("Пожалуйста, введите возраст числом (например, 25).")
        return

    age = int(message.text.strip())
    if age < 1 or age > 120:
        await message.answer("Пожалуйста, введите реальный возраст.")
        return

    user_id = message.from_user.id

    data = await state.get_data()

    if data.get('is_settings'):
        async with async_session_maker() as session:
            stmt = update(User).where(User.id == user_id).values(age=str(age))
            await session.execute(stmt)
            await session.commit()
            user = await session.get(User, user_id)

        await state.clear()
        await message.delete()

        text = (
            f"✅ Возраст установлен: {age}\n\n"
            "⚙️ <b>Настройки</b>\n\n"
            f"<b>Имя:</b> {html.escape(user.name or user.first_name or 'Не указано')}\n"
            f"<b>Пол:</b> {'👨 Мужской' if user.gender == 'male' else ('👩 Женский' if user.gender == 'female' else '❓ Не указан')}\n"
            f"<b>Возраст:</b> {user.age or 'Не указан'}\n"
            f"<b>Длина ответов:</b> {'📏 Обычный' if getattr(user, 'response_length', 'normal') != 'short' else '📏 Короткий'}\n"
        )
        settings_msg_id = data.get('settings_message_id')
        if settings_msg_id:
            try:
                await bot.edit_message_text(text, chat_id=message.chat.id, message_id=settings_msg_id, reply_markup=kb.user_settings_keyboard(user))
            except Exception:
                await message.answer(text, reply_markup=kb.user_settings_keyboard(user))
        else:
            await message.answer(text, reply_markup=kb.user_settings_keyboard(user))
        return

    async with async_session_maker() as session:
        stmt = update(User).where(User.id == user_id).values(age=str(age))
        await session.execute(stmt)

        existing_session = await session.get(TestSession, user_id)

        if existing_session:
            existing_session.created_at = datetime.utcnow()
            existing_session.answers = ""
            existing_session.is_finished = False
            existing_session.current_question_index = 0
        else:
            new_session = TestSession(
                user_id=user_id,
                created_at=datetime.utcnow(),
                answers="",
                is_finished=False,
                current_question_index=0
            )
            session.add(new_session)

        await session.commit()

    await state.update_data(answers=[])
    await state.set_state(UserStates.in_test)
    await send_next_question(message, user_id, state, bot)


async def finish_test_generation(message: Message, user_id: int, answers: list, questions: list):
    scores = {key: 0 for key in CATEGORY_NAMES.keys()}
    max_scores = {key: 0 for key in CATEGORY_NAMES.keys()}
    total_score = 0
    total_max = 0

    for i, ans in enumerate(answers):
        if i < len(questions):
            q = questions[i]
            cat = q.category
            final_score = ans
            if q.is_reverse:
                final_score = 6 - ans
            if cat in scores:
                scores[cat] += final_score
                max_scores[cat] += 5
            total_score += final_score
            total_max += 5

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        user_name = user.name or "Незнакомец"
        user_age = user.age or "Не указан"
        user_gender_raw = user.gender or "unknown"
        user_gender_str = "Женский" if user_gender_raw == 'female' else "Мужской" if user_gender_raw == 'male' else "Не определен"

        dialogue_id = user.current_dialogue_id
        topic_id = user.current_topic_id
        test_config = await session.get(TestConfig, 1)
        system_instruction = test_config.test_system_prompt

    diagram_text = "📊 <b>ТВОЯ КАРТА САМООЦЕНКИ</b>\n\n"
    weak_points = []
    for cat, score in scores.items():
        max_s = max_scores[cat]
        if max_s == 0: continue
        percent = score / max_s
        blocks_count = 10
        filled = int(percent * blocks_count)
        empty = blocks_count - filled
        bar = "█" * filled + "░" * empty
        status = ""
        if percent < 0.4:
            status = " ⚠️ Зона внимания"
            weak_points.append(CATEGORY_NAMES.get(cat, cat))
        elif percent > 0.8:
            status = " ✅ Сильная сторона"
        cat_name = CATEGORY_NAMES.get(cat, cat)
        diagram_text += f"<b>{cat_name}</b>\n{bar} {score}/{max_s}{status}\n"

    total_percent = int(total_score / total_max * 100) if total_max > 0 else 0
    diagram_text += f"\n<b>ОБЩИЙ ИТОГ:</b> {total_score}/{total_max} баллов ({total_percent}%)"

    loading_msg = await message.edit_text("🤖 Спасибо! Анализирую твои ответы и подбираю похожий случай из практики...")

    async with async_session_maker() as session:
        stmt = update(TestSession).where(TestSession.user_id == user_id).values(answers=diagram_text, is_finished=True)
        await session.execute(stmt)
        await session.commit()

    search_query = f"Пол: {user_gender_str}, Возраст: {user_age}. Проблемы: {', '.join(weak_points)}"
    found_case_text = await search_relevant_case(search_query)

    context_case_instruction = ""
    if found_case_text:
        context_case_instruction = (
            f"\n\n[РЕАЛЬНЫЙ КЕЙС ИЗ БАЗЫ ЗНАНИЙ (ДЛЯ ПРИМЕРА СТРУКТУРЫ)]\n"
            f"{found_case_text}\n"
            f"ИНСТРУКЦИЯ: Используй сюжет этого кейса как шаблон ситуации, но ПОЛНОСТЬЮ ЗАМЕНИ героя на пользователя.\n"
            f"ВАЖНО: Если в кейсе указано другое имя или возраст — ЗАБУДЬ ИХ. Пиши про {user_name}."
        )
    else:
        context_case_instruction = "\n\n(Подходящий кейс в базе не найден. Придумай собирательный образ на основе проблемных зон пользователя)."

    user_prompt_data = (
        f"ВЫПОЛНИ ЗАДАЧУ 1: СЦЕНАРИСТ (ИСТОРИЯ-ЗЕРКАЛО).\n\n"
        f"ДАННЫЕ ТЕКУЩЕГО ПОЛЬЗОВАТЕЛЯ (ГЛАВНЫЙ ГЕРОЙ):\n"
        f"Имя: {user_name}\n"
        f"Возраст: {user_age}\n"
        f"Пол: {user_gender_str}\n\n"
        f"РЕЗУЛЬТАТЫ ТЕСТА:\n{diagram_text}\n"
        f"{context_case_instruction}\n\n"
        f"Напиши историю от третьего лица про {user_name} ({user_age} лет), которая вызовет чувство 'Это же про меня!'.\n"
        f"СТРОГОЕ ПРАВИЛО: Имя героя — ТОЛЬКО {user_name}. Возраст — {user_age}. Не используй имена из примеров/кейсов.\n"
        f"Не пиши цифры и баллы, только историю жизни."
    )

    case_study_text = await get_ai_response_direct(user_id, system_instruction, user_prompt_data)

    html_story = markdown_to_html(case_study_text)

    final_message_text = f"{html_story}\n\n<i>Присмотрись. Возможно, ты мог узнать себя в этой истории</i>"

    async with async_session_maker() as session:
        session.add(DBMessage(
            user_id=user_id,
            role='assistant',
            content=case_study_text,
            dialogue_id=dialogue_id,
            topic_id=topic_id
        ))
        await session.commit()

    await loading_msg.edit_text(
        final_message_text,
        reply_markup=kb.case_study_confirmation_keyboard()
    )


@router.callback_query(F.data == "test_confirm_case")
async def show_test_results(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id

    await callback.message.edit_reply_markup(reply_markup=None)

    async with async_session_maker() as session:
        test_session = await session.get(TestSession, user_id)
        diagram_text = test_session.answers

        test_config = await session.get(TestConfig, 1)
        system_instruction = test_config.test_system_prompt
        marathon_url = test_config.marathon_url

        user = await session.get(User, user_id)
        dialogue_id = user.current_dialogue_id
        topic_id = user.current_topic_id
        user_name = user.name if user.name else "Друг"

        content_obj = await session.get(Content, "test_results", options=[selectinload(Content.media)])

    if content_obj:
        text_content = content_obj.text_content
        media_files = content_obj.media
        content_order = content_obj.content_order

        html_text = markdown_to_html(text_content) if text_content else ""

        async def send_text():
            if html_text:
                await callback.message.answer(html_text)

        async def send_media():
            if media_files:
                if len(media_files) == 1 and len(html_text) < 1000 and content_order == 'media_top' and html_text:
                    item = media_files[0]
                    if item.file_type == 'photo':
                        await callback.message.answer_photo(item.file_id, caption=html_text)
                    elif item.file_type == 'video':
                        await callback.message.answer_video(item.file_id, caption=html_text)
                    return True

                media_group = []
                for item in media_files:
                    media_to_add = InputMediaPhoto(
                        media=item.file_id) if item.file_type == 'photo' else InputMediaVideo(media=item.file_id)
                    media_group.append(media_to_add)

                if media_group:
                    await callback.message.answer_media_group(media_group)
            return False

        sent_combined = False
        if media_files and len(media_files) == 1 and len(html_text) < 1000 and content_order == 'media_top':
            sent_combined = await send_media()

        if not sent_combined:
            if content_order == 'media_top':
                await send_media()
                await asyncio.sleep(0.3)
                await send_text()
            else:
                await send_text()
                await asyncio.sleep(0.3)
                await send_media()

    await callback.message.answer(diagram_text)

    loading_msg = await callback.message.answer("⏳ Генерирую подробную расшифровку и план действий...")

    total_score = 0
    try:
        match = re.search(r"ОБЩИЙ ИТОГ: (\d+)/", diagram_text)
        if match:
            total_score = int(match.group(1))
    except Exception:
        total_score = 0

    interpretation_text_base = ""
    if 155 <= total_score:
        interpretation_text_base = (
            "У Вас высокая самооценка, Вы уверены в своих силах и способностях, и довольны собой. "
            "Но при этом у Вас есть потребность в доминировании. Вы часто критикуете других, но критику в свой адрес не воспринимаете. "
            "Многие удивляются Вашей уверенности в себе. Не исключено, что иногда Вы выгораете. "
            "Рекомендуем обратиться за помощью к специалисту."
        )
    elif 101 <= total_score <= 154:
        interpretation_text_base = (
            "Скорее всего, Вы довольны собой и своей жизнью, реалистично смотрите на окружающих. "
            "Очень высока вероятность, что у Вас нормальная, адекватная самооценка. "
            "Но нет предела совершенству. Посмотрите, по какому направлению у Вас наиболее низкий балл. "
            "Возможно, именно там - Ваша зона роста."
        )
    elif 50 <= total_score <= 100:
        interpretation_text_base = (
            "У вас заниженная самооценка. Вероятно, Вы воспринимаете себя излишне критично. "
            "Задачи, которые Вы себе ставите, зачастую превышают Ваши возможности. "
            "Вам необходимо обратить внимание на тайм-менеджмент, снизить критику и ожидания. "
            "Научиться относиться к ошибкам более спокойно, а к себе - с большей заботой."
        )
    elif total_score < 50:
        interpretation_text_base = (
            "Вероятно, Вы не очень довольны собой, периодически Вас мучают сомнения. "
            "Вы зависимы от мнения других людей. Из-за этого Вам сложно найти себя в жизни. "
            "Вам обязательно необходимо обратиться к специалисту за помощью."
        )

    user_prompt_data = (
        f"ВЫПОЛНИ ЗАДАЧУ 2: МАРКЕТОЛОГ (РАСШИФРОВКА).\n\n"
        f"Имя клиента: {user_name}\n"
        f"Результаты пользователя:\n{diagram_text}\n\n"
        f"БАЗОВАЯ ИНТЕРПРЕТАЦИЯ (ИСПОЛЬЗУЙ ЕЁ КАК ОСНОВУ ДЛЯ АНАЛИЗА):\n"
        f"'{interpretation_text_base}'\n\n"
        "Твои действия:\n"
        "1. Дай расшифровку своими словами, обращаясь к клиенту по имени.\n"
        "2. Объясни риски текущего состояния.\n"
        "3. Презентуй марафон.\n"
        "4. В конце сделай подводку к кнопке, например: 'Жми кнопку ниже, чтобы пройти секретный тест'. НЕ пиши текст кнопки в кавычках и не описывай, как она выглядит, просто призови нажать."
    )

    interpretation_response = await get_ai_response_direct(user_id, system_instruction, user_prompt_data)

    interpretation_response = interpretation_response.replace("{user_name}", user_name)

    html_response = markdown_to_html(interpretation_response)

    async with async_session_maker() as session:
        session.add(DBMessage(
            user_id=user_id,
            role='assistant',
            content=interpretation_response,
            dialogue_id=dialogue_id,
            topic_id=topic_id
        ))
        await session.commit()

    await loading_msg.delete()

    chunks = split_message(html_response, 4000)
    for chunk in chunks:
        await _safe_send_html(
            lambda text, pm: callback.message.answer(text, parse_mode=pm),
            chunk,
        )
        await asyncio.sleep(0.3)

    secret_test_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔐 Пройти секретный тест", callback_data="start_secret_test")],
        [InlineKeyboardButton(text="Сразу на марафон 🚀", url=marathon_url)]
    ])

    await callback.message.answer(
        "Готовы копнуть глубже и получить личный разбор от меня?",
        reply_markup=secret_test_kb
    )


@router.callback_query(F.data == "start_secret_test")
async def start_secret_test_handler(callback: CallbackQuery, state: FSMContext):
    async with async_session_maker() as session:
        questions = (
            await session.execute(select(SecretTestQuestion).order_by(SecretTestQuestion.sort_order))).scalars().all()

    if not questions:
        questions_text = "Вопросы еще не добавлены администратором."
    else:
        questions_text = "<b>🔐 Секретный блок вопросов</b>\n\nОтветь на них максимально честно.\n\n"
        for i, q in enumerate(questions):
            questions_text += f"{i + 1}️⃣ <i>{q.text}</i>\n\n"
        questions_text += "👇 <b>Напиши ответы одним сообщением ниже.</b>"

    await callback.message.edit_text(questions_text)
    await state.set_state(UserStates.secret_test_answering)


@router.message(UserStates.secret_test_answering, F.text)
async def process_secret_answers(message: Message, state: FSMContext):
    async with async_session_maker() as session:
        stmt = update(TestSession).where(TestSession.user_id == message.from_user.id).values(secret_answers=message.text)
        await session.execute(stmt)
        await session.commit()

        config = await session.get(TestConfig, 1)
        marathon_url = config.marathon_url

        content_obj = await session.get(Content, "secret_test_outro", options=[selectinload(Content.media)])

        final_text = "Спасибо за ответы!"
        media_files = []
        content_order = 'media_top'

        if content_obj:
            final_text = content_obj.text_content or final_text
            content_order = content_obj.content_order
            media_files = [
                {'type': m.file_type, 'file_id': m.file_id} for m in content_obj.media
            ]

    final_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔥 Программа марафона", url=marathon_url)],
        [InlineKeyboardButton(text="🗣 Продолжить общение", callback_data="continue_dialogue_after_test")]
    ])

    html_text = markdown_to_html(final_text)

    async def send_text_part():
        if html_text:
            await message.answer(html_text, reply_markup=final_kb)

    async def send_media_part():
        if media_files:
            if len(media_files) == 1 and len(html_text) < 1000 and content_order == 'media_top':
                item = media_files[0]
                if item['type'] == 'photo':
                    await message.answer_photo(item['file_id'], caption=html_text, reply_markup=final_kb)
                elif item['type'] == 'video':
                    await message.answer_video(item['file_id'], caption=html_text, reply_markup=final_kb)
                return True

            media_group = []
            for item in media_files:
                media_to_add = InputMediaPhoto(media=item['file_id']) if item['type'] == 'photo' else InputMediaVideo(
                    media=item['file_id'])
                media_group.append(media_to_add)

            if media_group:
                await message.answer_media_group(media_group)
        return False

    sent_combined = False

    if media_files and len(media_files) == 1 and len(html_text) < 1000 and content_order == 'media_top':
        sent_combined = await send_media_part()

    if not sent_combined:
        if content_order == 'media_top':
            await send_media_part()
            await asyncio.sleep(0.3)
            await send_text_part()
        else:
            await send_text_part()
            await asyncio.sleep(0.3)
            await send_media_part()

    await state.clear()


@router.callback_query(F.data == "continue_dialogue_after_test")
async def continue_dialogue_handler(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.answer()
    await callback.message.answer("Я здесь! Мы можем обсудить твои результаты или поговорить на любую другую тему. Слушаю тебя.")


@router.callback_query(F.data == "admin_upload_questions")
async def admin_upload_questions_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.upload_test_questions_file)

    text = (
        "<b>Загрузка вопросов для теста</b>\n\n"
        "Загрузите файл <b>.xlsx</b> (Excel).\n\n"
        "<b>Формат Excel (3 колонки):</b>\n"
        "Col A: Текст вопроса\n"
        "Col B: Код категории (body, face, age, health, abilities, relations, success)\n"
        "Col C: Обратный счет (1 - да, 0 - нет)\n"
    )

    await state.update_data(instruction_msg_id=callback.message.message_id)
    await callback.message.edit_text(text, reply_markup=kb.back_to_previous_menu("admin_test_menu"))


@router.message(AdminStates.upload_test_questions_file, F.document)
async def admin_process_questions_file(message: Message, state: FSMContext, bot: Bot):
    try:
        await message.delete()
    except:
        pass

    data = await state.get_data()
    instruction_msg_id = data.get("instruction_msg_id")
    if instruction_msg_id:
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=instruction_msg_id)
        except:
            pass

    async def send_new_menu(text, reply_markup=None, **kwargs):
        await message.answer(text, reply_markup=reply_markup, **kwargs)

    mock_message = type('MockMessage', (object,), {
        'chat': message.chat,
        'message_id': message.message_id,
        'edit_text': send_new_menu,
        'answer': message.answer
    })()

    call_mock = type('MockCallback', (object,), {
        'message': mock_message,
        'bot': bot,
        'from_user': message.from_user,
        'answer': lambda *a, **k: None,
        'data': 'admin_test_menu'
    })()

    document = message.document
    file_ext = document.file_name.split('.')[-1].lower()

    if file_ext not in ['xlsx', 'txt']:
        await message.answer("❌ Поддерживаются только файлы .xlsx и .txt")
        await admin_test_menu(call_mock)
        await state.clear()
        return

    try:
        file_info = await bot.get_file(document.file_id)
        file_bytes_io = await bot.download_file(file_info.file_path)

        questions_data = await parse_questions_file(file_bytes_io, document.file_name)

        if not questions_data:
            await message.answer("❌ Не удалось найти вопросы в файле. Проверьте формат.")
            await admin_test_menu(call_mock)
            await state.clear()
            return

        async with async_session_maker() as session:
            await session.execute(delete(TestQuestion))

            for idx, q_data in enumerate(questions_data):
                new_q = TestQuestion(
                    text=q_data['text'],
                    category=q_data['category'],
                    is_reverse=q_data['is_reverse'],
                    sort_order=idx
                )
                session.add(new_q)

            await session.commit()

        await message.answer(f"✅ Успешно загружено {len(questions_data)} вопросов!")
        await admin_test_menu(call_mock)
        await state.clear()

    except Exception as e:
        logging.error(f"Error uploading questions: {e}")
        await message.answer(f"❌ Произошла ошибка: {e}")
        await admin_test_menu(call_mock)
        await state.clear()


async def get_ai_response_direct(user_id: int, system_prompt: str, user_prompt: str) -> str:
    async with async_session_maker() as session:
        ai_config = await session.get(AIConfig, 1)
        provider = ai_config.provider

        provider_key = provider.strip().lower() if provider else ""

        api_key = getattr(ai_config, f"{provider_key}_api_key", None)
        if provider_key in ['anthropic', 'claude'] and not api_key:
            api_key = ai_config.claude_api_key

        model = getattr(ai_config, f"{provider_key}_model", None)
        if provider_key in ['anthropic', 'claude'] and not model:
            model = ai_config.claude_model

        fake_history = [DBMessage(role='user', content=user_prompt)]

        if provider_key == 'gemini':
            return await _call_gemini_api(api_key, model, fake_history, "", system_prompt)
        elif provider_key == 'openai':
            return await _call_openai_api(api_key, model, fake_history, "", system_prompt)
        elif provider_key in ['anthropic', 'claude']:
            return await _call_claude_api(api_key, model, fake_history, "", system_prompt)
        elif provider_key == 'deepseek':
            return await _call_deepseek_api(api_key, model, fake_history, "", system_prompt)

        return f"Ошибка: Неизвестный провайдер ИИ ({provider})."


@router.message(Command("test"))
@router.message(TestButtonFilter())
async def cmd_start_test(message: Message, state: FSMContext):
    user_id = message.from_user.id
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        if config and not config.is_enabled:
            if not await is_admin(user_id):
                await message.answer("⚠️ Тестирование в данный момент отключено.")
                return

        questions_exist = await session.scalar(select(func.count(TestQuestion.id)))
        if not questions_exist:
            await message.answer("⚠️ Тест временно недоступен: вопросы еще не загружены.")
            return

        content = await get_content_from_db("test_intro")

    await state.set_state(UserStates.awaiting_gender)

    text = content.get('text', "Привет! Для начала выбери свой пол:")
    media = content.get('media', [])

    keyboard = kb.gender_selection_keyboard()

    if media:
        if len(media) == 1 and len(text) < 1024:
            item = media[0]
            if item['type'] == 'photo':
                await message.answer_photo(item['file_id'], caption=text, reply_markup=keyboard)
            elif item['type'] == 'video':
                await message.answer_video(item['file_id'], caption=text, reply_markup=keyboard)
        else:
            for item in media:
                if item['type'] == 'photo':
                    await message.answer_photo(item['file_id'])
                elif item['type'] == 'video':
                    await message.answer_video(item['file_id'])
            await message.answer(text, reply_markup=keyboard)
    else:
        await message.answer(text, reply_markup=keyboard)


@router.callback_query(F.data == "admin_test_menu")
async def admin_test_menu(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        if not config:
            config = TestConfig(id=1)
            session.add(config)
            await session.commit()

    await callback.message.edit_text(
        "🧩 <b>Управление разделом 'Тест'</b>\n\n"
        "Здесь вы можете настроить всё, что касается прохождения теста пользователем.",
        reply_markup=kb.admin_test_menu_keyboard(config.is_enabled)
    )


@router.callback_query(F.data == "admin_test_toggle_status")
async def admin_test_toggle_status(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        config.is_enabled = not config.is_enabled
        new_status = config.is_enabled

        btn_content = await session.get(Content, "test_button")
        if btn_content:
            btn_content.is_visible = new_status

        await session.commit()

    await admin_test_menu(callback)


@router.callback_query(F.data == "admin_test_links")
async def admin_test_links_menu(callback: CallbackQuery):
    await _show_admin_test_links_menu(callback.bot, callback.message.chat.id, callback.message.message_id)


@router.callback_query(F.data.startswith("set_test_link_"))
async def start_set_test_link(callback: CallbackQuery, state: FSMContext):
    link_type = callback.data.replace("set_test_link_", "")
    await state.set_state(AdminStates.set_single_payment_key)
    await state.update_data(link_type=link_type, message_id_to_edit=callback.message.message_id)

    prompt = "Введите Username администратора (без @):" if link_type == "admin" else "Введите ссылку на марафон:"
    await callback.message.edit_text(prompt, reply_markup=kb.back_to_previous_menu("admin_test_links"))


@router.message(AdminStates.set_single_payment_key, F.text)
async def process_test_link_input(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    link_type = data.get('link_type')

    if not link_type:
        await admin_process_single_payment_key(message, state, bot)
        return

    new_value = message.text.strip()
    if link_type == "admin":
        new_value = new_value.replace("@", "")

    async with async_session_maker() as session:
        if link_type == "admin":
            await session.execute(update(TestConfig).where(TestConfig.id == 1).values(admin_username=new_value))
        else:
            await session.execute(update(TestConfig).where(TestConfig.id == 1).values(marathon_url=new_value))
        await session.commit()

    await state.clear()
    await message.delete()

    msg_id = data.get('message_id_to_edit')
    if msg_id:
        try:
            await _show_admin_test_links_menu(bot, message.chat.id, msg_id)
        except:
            pass

    temp = await message.answer("✅ Значение обновлено!")
    await asyncio.sleep(3)
    await temp.delete()


async def _show_admin_test_links_menu(bot, chat_id, msg_id):
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)

    text = (
        "🔗 <b>Настройка ссылок</b>\n\n"
        f"👤 <b>Admin Username:</b> @{config.admin_username}\n"
        f"🚀 <b>Марафон URL:</b> {config.marathon_url}\n\n"
        "Нажмите кнопку, чтобы изменить значение."
    )
    await bot.edit_message_text(text, chat_id=chat_id, message_id=msg_id, reply_markup=kb.admin_test_links_keyboard(),
                                disable_web_page_preview=True)


@router.callback_query(F.data == "admin_secret_questions")
async def admin_secret_questions_menu(callback: CallbackQuery):
    async with async_session_maker() as session:
        qs = (await session.execute(select(SecretTestQuestion).order_by(SecretTestQuestion.sort_order))).scalars().all()

    await callback.message.edit_text(
        "🔐 <b>Секретные вопросы</b>\n\nДобавляйте вопросы, которые бот задаст пользователю после прохождения основного теста.",
        reply_markup=kb.admin_secret_questions_keyboard(qs)
    )


@router.callback_query(F.data == "add_secret_question")
async def add_secret_question_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_button_title)
    await state.update_data(is_secret_question=True, message_id=callback.message.message_id)
    await callback.message.edit_text("Введите текст нового секретного вопроса:", reply_markup=kb.back_to_previous_menu("admin_secret_questions"))


@router.message(AdminStates.set_button_title, F.text)
async def process_secret_question_text(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    if not data.get('is_secret_question'):
        await admin_process_button_title(message, state, bot)
        return

    text = message.text
    async with async_session_maker() as session:
        count = await session.scalar(select(func.count(SecretTestQuestion.id)))
        new_q = SecretTestQuestion(text=text, sort_order=count + 1)
        session.add(new_q)
        await session.commit()

    await state.clear()
    await message.delete()

    msg_id = data.get('message_id')
    if msg_id:
        try:
            async with async_session_maker() as session:
                qs = (await session.execute(
                    select(SecretTestQuestion).order_by(SecretTestQuestion.sort_order))).scalars().all()
            await bot.edit_message_text(
                "🔐 <b>Секретные вопросы</b>\n\nДобавляйте вопросы, которые бот задаст пользователю после прохождения основного теста.",
                chat_id=message.chat.id, message_id=msg_id, reply_markup=kb.admin_secret_questions_keyboard(qs)
            )
        except:
            pass

    temp = await message.answer("✅ Вопрос добавлен")
    await asyncio.sleep(2)
    await temp.delete()


@router.callback_query(F.data.startswith("delete_secret_q_"))
async def delete_secret_question(callback: CallbackQuery):
    q_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        await session.execute(delete(SecretTestQuestion).where(SecretTestQuestion.id == q_id))
        await session.commit()

    await admin_secret_questions_menu(callback)


@router.callback_query(F.data.startswith("admin_case_studies_page_"))
async def admin_case_studies_list(callback: CallbackQuery):
    page = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        total = await session.scalar(select(func.count(CaseStudy.id)))
        total_pages = math.ceil(total / PAGE_SIZE) if total > 0 else 1
        page = max(0, min(page, total_pages - 1))

        cases = (await session.execute(
            select(CaseStudy).order_by(CaseStudy.id.desc()).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
        )).scalars().all()

    await callback.message.edit_text(
        f"📖 <b>База Историй/Кейсов (Стр. {page + 1}/{total_pages})</b>\n\n"
        "Бот будет искать здесь похожую историю по результатам теста пользователя и использовать её в ответе.",
        reply_markup=kb.admin_case_studies_keyboard(cases, page, total_pages)
    )


@router.callback_query(F.data == "add_case_study")
async def add_case_study_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.upload_kb_file)
    await state.update_data(is_case_study=True)
    await callback.message.edit_text(
        "Отправьте текст истории (кейса) одним сообщением или файлом .txt.\n\n"
        "Опишите проблему, с которой пришел клиент, и результат, чтобы бот мог подобрать это под пользователя.",
        reply_markup=kb.back_to_previous_menu("admin_case_studies_page_0")
    )


@router.message(AdminStates.upload_kb_file, (F.text | F.document))
async def process_case_study_upload(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    if not data.get('is_case_study'):
        await process_kb_file(message, state, bot)
        return

    text_content = ""
    if message.text:
        text_content = message.text
    elif message.document:
        file_info = await bot.get_file(message.document.file_id)
        file_bytes = await bot.download_file(file_info.file_path)
        text_content = file_bytes.read().decode('utf-8')

    if not text_content:
        await message.answer("Пустой контент.")
        return

    async with async_session_maker() as session:
        new_case = CaseStudy(text=text_content)
        session.add(new_case)
        await session.commit()
        await session.refresh(new_case)
        case_id = new_case.id

    await update_case_study_index(case_id, text_content)

    await state.clear()
    await message.delete()

    temp = await message.answer(f"✅ Кейс #{case_id} добавлен и проиндексирован!")

    async with async_session_maker() as session:
        cases = (
            await session.execute(select(CaseStudy).order_by(CaseStudy.id.desc()).limit(PAGE_SIZE))).scalars().all()

    await message.answer(
        "📖 <b>База Историй/Кейсов</b>",
        reply_markup=kb.admin_case_studies_keyboard(cases, 0, 1)
    )

    await asyncio.sleep(3)
    await temp.delete()


@router.callback_query(F.data.startswith("delete_case_"))
async def delete_case_study(callback: CallbackQuery):
    case_id = int(callback.data.split("_")[-1])
    async with async_session_maker() as session:
        await session.execute(delete(CaseStudy).where(CaseStudy.id == case_id))
        await session.commit()

    delete_case_study_vectors(case_id)
    await callback.answer("Кейс удален.")
    await admin_case_studies_list(callback)


@router.callback_query(F.data == "admin_edit_test_prompt")
async def admin_edit_test_prompt_menu(callback: CallbackQuery, state: FSMContext):
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        current_prompt = config.test_system_prompt or "Не задан."

    await state.set_state(AdminStates.set_test_system_prompt)
    await state.update_data(message_id=callback.message.message_id)

    display_prompt = current_prompt
    if len(display_prompt) > 3000:
        display_prompt = display_prompt[:3000] + "\n\n[...] (Текст обрезан для превью)"

    text = (
        f"<b>Текущий системный промпт для теста:</b>\n\n"
        f"<code>{html.escape(display_prompt)}</code>\n\n"
        f"Отправьте новый текст промпта или загрузите <b>.txt файл</b> с промптом."
    )

    try:
        await callback.message.edit_text(text, reply_markup=kb.test_prompt_keyboard())
    except TelegramBadRequest:
        await callback.message.delete()
        await callback.message.answer(text, reply_markup=kb.test_prompt_keyboard())


@router.callback_query(F.data == "download_test_prompt")
async def download_test_prompt(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(TestConfig, 1)
        prompt_text = config.test_system_prompt or "Промпт не задан."

    file_bytes = prompt_text.encode('utf-8')
    file_to_send = BufferedInputFile(
        file_bytes,
        filename="test_system_prompt.txt"
    )

    await callback.message.answer_document(
        file_to_send,
        caption="📄 Текущий системный промпт для режима 'Тест'."
    )
    await callback.answer()


@router.message(AdminStates.set_test_system_prompt, (F.text | F.document))
async def admin_process_test_prompt(message: Message, state: FSMContext, bot: Bot):
    new_prompt = None
    notification_text = ""

    if message.text:
        new_prompt = message.text.strip()
        notification_text = "✅ Промпт для теста обновлен текстом."
    elif message.document:
        if not message.document.file_name.lower().endswith('.txt'):
            temp = await message.answer("❌ Ошибка: Пожалуйста, загрузите файл в формате .txt")
            await asyncio.sleep(4)
            await temp.delete()
            return

        try:
            file_info = await bot.get_file(message.document.file_id)
            file_bytes = await bot.download_file(file_info.file_path)
            new_prompt = file_bytes.read().decode('utf-8')
            notification_text = f"✅ Промпт обновлен из файла `{message.document.file_name}`."
        except Exception as e:
            temp = await message.answer(f"❌ Ошибка чтения файла: {e}")
            await asyncio.sleep(4)
            await temp.delete()
            return

    data = await state.get_data()
    message_id = data.get('message_id')

    async with async_session_maker() as session:
        stmt = update(TestConfig).where(TestConfig.id == 1).values(test_system_prompt=new_prompt)
        await session.execute(stmt)
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id:
        try:
            callback_mock = type('obj', (object,), {
                'message': type('obj', (object,), {
                    'chat': message.chat,
                    'message_id': message_id,
                    'edit_text': lambda *args, **kwargs: bot.edit_message_text(chat_id=message.chat.id, message_id=message_id, *args, **kwargs),
                    'delete': lambda *args, **kwargs: bot.delete_message(chat_id=message.chat.id, message_id=message_id),
                    'answer': lambda *args, **kwargs: bot.send_message(chat_id=message.chat.id, *args, **kwargs)
                })
            })()
            await admin_edit_test_prompt_menu(callback_mock, state)
        except Exception:
            pass

    temp = await message.answer(notification_text)
    await asyncio.sleep(3)
    await temp.delete()


@router.callback_query(AdminStates.edit_content, F.data.startswith("toggle_content_visibility_"))
async def toggle_content_visibility_handler(callback: CallbackQuery, state: FSMContext):
    content_key = callback.data.replace("toggle_content_visibility_", "")

    async with async_session_maker() as session:
        content_obj = await session.get(Content, content_key)
        if content_obj:
            content_obj.is_visible = not content_obj.is_visible
            is_visible = content_obj.is_visible
            await session.commit()
        else:
            is_visible = False

    text, keyboard = await get_content_display(state)
    if text:
        try:
            await callback.message.edit_text(text, reply_markup=keyboard)
        except TelegramBadRequest:
            pass

    status_text = "показан" if is_visible else "скрыт"
    await callback.answer(f"Раздел теперь {status_text}.")


@router.callback_query(F.data == "admin_toggle_change_name_btn")
async def admin_toggle_change_name_btn(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.change_name_button_enabled = not config.change_name_button_enabled
        is_enabled = config.change_name_button_enabled
        await session.commit()

    await callback.answer(f"Кнопка 'Настройки' {'включена' if is_enabled else 'выключена'}")
    await callback.message.edit_reply_markup(
        reply_markup=kb.admin_payment_settings_keyboard(config)
    )


@router.callback_query(F.data == "admin_toggle_topics_on_top")
async def admin_toggle_topics_on_top(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.topics_btn_on_top = not config.topics_btn_on_top
        await session.commit()

    await callback.answer("Расположение кнопки тем изменено.")
    await show_topics_admin_list(callback, 0)


@router.callback_query(F.data == "admin_rename_topics_btn")
async def admin_set_topics_btn_name(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_button_title)
    await state.update_data(is_topics_btn=True, message_id=callback.message.message_id)
    await callback.message.edit_text(
        "Введите новое название для кнопки 'Темы диалогов':",
        reply_markup=kb.back_to_previous_menu("admin_topics_page_0")
    )


@router.message(AdminStates.set_button_title, F.text)
async def process_button_title_general(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()

    if data.get('is_topics_btn'):
        async with async_session_maker() as session:
            stmt = update(SubscriptionConfig).where(SubscriptionConfig.id == 1).values(topics_btn_name=message.text)
            await session.execute(stmt)
            await session.commit()

        await state.clear()
        await message.delete()
        msg_id = data.get('message_id')
        if msg_id:
            try:
                async with async_session_maker() as session:
                    config = await session.get(SubscriptionConfig, 1)
                    total_topics_q = await session.execute(select(func.count(Topic.id)))
                    total_topics = total_topics_q.scalar_one()
                    total_pages = math.ceil(total_topics / PAGE_SIZE)
                    page = 0
                    topics_q = await session.execute(
                        select(Topic).order_by(Topic.name).offset(page * PAGE_SIZE).limit(PAGE_SIZE)
                    )
                    topics = topics_q.scalars().all()

                await bot.edit_message_text(
                    f"💬 Управление темами диалогов (Стр. {page + 1}/{total_pages})",
                    chat_id=message.chat.id, message_id=msg_id,
                    reply_markup=kb.topics_admin_list_keyboard(topics, page, total_pages, config))
            except Exception:
                pass

        temp = await message.answer("✅ Название кнопки обновлено!")
        await asyncio.sleep(2)
        await temp.delete()
        return

    if data.get('is_secret_question'):
        await process_secret_question_text(message, state, bot)
        return

    button_key = data['button_key']
    message_id = data['message_id']

    async with async_session_maker() as session:
        stmt = update(Content).where(Content.key == button_key).values(button_title=message.text)
        await session.execute(stmt)
        await session.commit()

    await state.clear()
    await message.delete()

    if message_id:
        try:
            await _show_admin_manage_buttons(bot, message.chat.id, message_id)
        except TelegramBadRequest:
            pass

    temp_msg = await message.answer("✅ Название кнопки обновлено!")
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.callback_query(F.data.startswith("toggle_topic_display_"))
async def admin_toggle_topic_display(callback: CallbackQuery):
    parts = callback.data.split("_")
    mode = parts[3]
    topic_id = int(parts[-1])

    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            if mode == "menu":
                topic.show_in_main_menu = not topic.show_in_main_menu
            elif mode == "list":
                topic.show_in_list = not topic.show_in_list
            await session.commit()

    await _show_edit_topic_menu(callback.bot, callback.message.chat.id, callback.message.message_id, topic_id)
    await callback.answer("Настройки отображения обновлены.")


@router.callback_query(F.data.startswith("toggle_kb_general_"))
async def admin_toggle_kb_general(callback: CallbackQuery):
    parts = callback.data.split("_")
    kb_id = int(parts[-1])
    page = int(parts[-2])

    async with async_session_maker() as session:
        kb_file = await session.get(KnowledgeBase, kb_id)
        if kb_file:
            kb_file.use_in_general_mode = not kb_file.use_in_general_mode
            status_text = "включен" if kb_file.use_in_general_mode else "выключен"
            await session.commit()

    await callback.answer(f"Файл {status_text} для общего режима.")
    await _display_kb_page(callback.message, page)


@router.message(TopicsButtonFilter())
async def handle_topics_button_click(message: Message):
    await select_topic_menu(message)


class TopicDirectButtonFilter(Filter):
    async def __call__(self, message: Message) -> bool | dict:
        if not message.text: return False
        async with async_session_maker() as session:
            is_admin_user = message.from_user.id in OWNER_IDS
            if not is_admin_user:
                user = await session.get(User, message.from_user.id)
                is_admin_user = bool(user and user.is_admin)
            conditions = [Topic.is_active == True, Topic.show_in_main_menu == True, Topic.name == message.text]
            if not is_admin_user:
                conditions.append(Topic.admin_only == False)
            topic = await session.scalar(select(Topic).where(*conditions))
            if topic:
                return {'topic_id': topic.id, 'topic_name': topic.name}
            return False


@router.message(TopicDirectButtonFilter())
async def handle_direct_topic_button(message: Message, topic_id: int, topic_name: str, bot: Bot):
    async with async_session_maker() as session:
        restored = False
        user = await session.get(User, message.from_user.id)
        if user:
            user.current_topic_id = topic_id
            ai_config = await session.get(AIConfig, 1)
            memory_mode = get_memory_mode(ai_config) if ai_config else MEMORY_MODE_RESET
            restored = await _apply_topic_switch(session, user, topic_id, memory_mode)
            await session.commit()

        topic = await session.get(Topic, topic_id)

    if topic and topic.start_message:
        text_to_send = topic.start_message
    else:
        text_to_send = _topic_switch_message(topic_name, restored, memory_mode)

    reply_markup = None
    if topic and topic.start_button_text and topic.start_button_payload:
        reply_markup = kb.action_button_keyboard(topic.start_button_text, "topic_action")

    await message.answer(
        text_to_send,
        parse_mode="HTML" if topic and topic.start_message else "Markdown",
        reply_markup=reply_markup
    )


@router.callback_query(F.data == "action_btn_click")
async def handle_action_button_click(callback: CallbackQuery, bot: Bot):
    user_id = callback.from_user.id
    payload_text = None

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if not user:
            await callback.answer("Ошибка пользователя.")
            return

        if user.current_topic_id:
            topic = await session.get(Topic, user.current_topic_id)
            if topic and topic.start_button_payload:
                payload_text = topic.start_button_payload

        if not payload_text:
            content = await session.get(Content, "start_message")
            if content and content.action_btn_payload:
                payload_text = content.action_btn_payload

    if payload_text:
        await callback.message.edit_reply_markup(reply_markup=None)

        mock_message = type('obj', (object,), {
            'from_user': callback.from_user,
            'text': payload_text,
            'chat': callback.message.chat,
            'answer': callback.message.answer,
            'delete': callback.message.delete
        })()

        await callback.message.answer(html.escape(payload_text))

        await process_user_prompt(mock_message, user_id, payload_text, bot)
        await callback.answer()
    else:
        await callback.answer("Действие не назначено.", show_alert=True)


@router.message(
    StateFilter(AdminStates.set_topic_intro_msg, AdminStates.set_topic_btn_text, AdminStates.set_topic_btn_payload),
    F.text)
async def admin_save_topic_extra_fields(message: Message, state: FSMContext, bot: Bot):
    current_state = await state.get_state()
    data = await state.get_data()
    topic_id = data.get('topic_id')
    message_id_to_edit = data.get('message_id_to_edit')

    value = message.text.strip()
    if value == "-": value = None

    column_map = {
        AdminStates.set_topic_intro_msg: "start_message",
        AdminStates.set_topic_btn_text: "start_button_text",
        AdminStates.set_topic_btn_payload: "start_button_payload"
    }

    target_column = column_map.get(current_state)

    async with async_session_maker() as session:
        await session.execute(update(Topic).where(Topic.id == topic_id).values({target_column: value}))
        await session.commit()

    await message.delete()
    await state.clear()

    temp = await message.answer("✅ Значение обновлено!")

    if message_id_to_edit:
        await _show_edit_topic_menu(bot, message.chat.id, message_id_to_edit, topic_id)

    await asyncio.sleep(2)
    await temp.delete()


@router.callback_query(F.data.startswith("clear_topic_btn_"))
async def clear_topic_btn_handler(callback: CallbackQuery, bot: Bot):
    topic_id = int(callback.data.split("_")[-1])

    async with async_session_maker() as session:
        await session.execute(
            update(Topic)
            .where(Topic.id == topic_id)
            .values(start_button_text=None, start_button_payload=None)
        )
        await session.commit()

    await _show_edit_topic_menu(bot, callback.message.chat.id, callback.message.message_id, topic_id)
    await callback.answer("✅ Кнопка действия для темы удалена.")


@router.message(StateFilter(AdminStates.set_content_btn_text, AdminStates.set_content_btn_payload), F.text)
async def admin_save_content_btn_fields(message: Message, state: FSMContext, bot: Bot):
    current_state = await state.get_state()
    data = await state.get_data()
    content_key = data.get('content_key')
    message_id_to_edit = data.get('message_id_to_edit')

    value = message.text.strip()
    if value == "-": value = None

    target_column = "action_btn_text" if current_state == AdminStates.set_content_btn_text else "action_btn_payload"

    async with async_session_maker() as session:
        await session.execute(update(Content).where(Content.key == content_key).values({target_column: value}))
        await session.commit()

    await message.delete()

    await state.set_state(AdminStates.edit_content)
    current_content = await get_content_from_db(content_key)
    async with async_session_maker() as session:
        obj = await session.get(Content, content_key)
        order = getattr(obj, 'content_order', 'media_top')

    await state.update_data(
        content_key=content_key,
        text_content=current_content.get('text'),
        media_files=current_content.get('media', []),
        content_order=order,
        message_id_to_edit=message_id_to_edit
    )

    text, keyboard = await get_content_display(state)
    if message_id_to_edit:
        try:
            await bot.edit_message_text(text, chat_id=message.chat.id, message_id=message_id_to_edit,
                                        reply_markup=keyboard)
        except TelegramBadRequest:
            pass

    temp = await message.answer(f"✅ Кнопка обновлена: {value}")
    await asyncio.sleep(2)
    await temp.delete()


@router.callback_query(F.data.startswith("clear_content_btn_"))
async def clear_content_btn_handler(callback: CallbackQuery, state: FSMContext, bot: Bot):
    content_key = callback.data.replace("clear_content_btn_", "")

    async with async_session_maker() as session:
        await session.execute(
            update(Content)
            .where(Content.key == content_key)
            .values(action_btn_text=None, action_btn_payload=None)
        )
        await session.commit()

    await state.set_state(AdminStates.edit_content)
    current_content = await get_content_from_db(content_key)
    async with async_session_maker() as session:
        obj = await session.get(Content, content_key)
        order = getattr(obj, 'content_order', 'media_top')

    await state.update_data(
        content_key=content_key,
        text_content=current_content.get('text'),
        media_files=current_content.get('media', []),
        content_order=order,
        message_id_to_edit=callback.message.message_id
    )

    text, keyboard = await get_content_display(state)
    try:
        await callback.message.edit_text(text, reply_markup=keyboard)
    except TelegramBadRequest:
        pass

    await callback.answer("✅ Кнопка действия удалена.")


@router.callback_query(F.data == "admin_toggle_change_name_btn_from_menu")
async def admin_toggle_change_name_btn_from_menu(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.change_name_button_enabled = not config.change_name_button_enabled
        await session.commit()

    await _show_admin_manage_buttons(callback.bot, callback.message.chat.id, callback.message.message_id)
    await callback.answer("Статус кнопки 'Настройки' обновлен")


@router.callback_query(F.data == "admin_set_welcome_bonus")
async def admin_set_welcome_bonus_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_welcome_bonus_days)
    await state.update_data(message_id_to_edit=callback.message.message_id)
    await callback.message.edit_text(
        "Введите количество дней для приветственного бонуса новым пользователям.\n"
        "Введите <b>0</b>, чтобы отключить бонус.",
        reply_markup=kb.back_to_previous_menu("admin_payment_settings")
    )

@router.message(AdminStates.set_welcome_bonus_days, F.text)
async def admin_save_welcome_bonus(message: Message, state: FSMContext, bot: Bot):
    try:
        days = int(message.text)
        if days < 0: raise ValueError
    except ValueError:
        await message.answer("❌ Введите целое положительное число или 0.")
        return

    data = await state.get_data()
    message_id_to_edit = data.get('message_id_to_edit')

    async with async_session_maker() as session:
        await session.execute(
            update(SubscriptionConfig).where(SubscriptionConfig.id == 1).values(welcome_bonus_days=days)
        )
        await session.commit()
        config = await session.get(SubscriptionConfig, 1)

    await state.clear()
    await message.delete()

    if message_id_to_edit:
        try:
            await bot.edit_message_text(
                "⚙️ Настройки платежных систем и уведомлений",
                chat_id=message.chat.id,
                message_id=message_id_to_edit,
                reply_markup=kb.admin_payment_settings_keyboard(config)
            )
        except TelegramBadRequest:
            pass

    temp_msg = await message.answer(f"✅ Приветственный бонус обновлен: {days} дн.")
    await asyncio.sleep(3)
    await temp_msg.delete()


@router.callback_query(F.data == "admin_payment_stats")
async def admin_payment_stats(callback: CallbackQuery):
    async with async_session_maker() as session:
        now = datetime.utcnow()

        total_users_stmt = select(func.count(User.id))
        total_users = (await session.execute(total_users_stmt)).scalar() or 0

        active_paid_stmt = select(UserSubscription).where(
            UserSubscription.plan_id.is_not(None),
            UserSubscription.end_date > now
        ).options(selectinload(UserSubscription.plan))
        active_paid_result = await session.execute(active_paid_stmt)
        active_paid_subs = active_paid_result.scalars().all()

        active_paid_count = len(active_paid_subs)
        current_mrr = sum(sub.plan.price for sub in active_paid_subs if sub.plan)

        active_trial_stmt = select(func.count(UserSubscription.id)).where(
            UserSubscription.plan_id.is_(None),
            UserSubscription.end_date > now
        )
        active_trials_count = (await session.execute(active_trial_stmt)).scalar() or 0

        expired_stmt = select(func.count(UserSubscription.id)).where(
            UserSubscription.end_date <= now
        )
        expired_count = (await session.execute(expired_stmt)).scalar() or 0

        robo_revenue_stmt = select(func.sum(RobokassaPayment.amount))
        total_robo_revenue = (await session.execute(robo_revenue_stmt)).scalar() or 0.0

        plan_stats_stmt = select(
            SubscriptionPlan.name,
            func.count(UserSubscription.id)
        ).join(UserSubscription).where(
            UserSubscription.end_date > now
        ).group_by(SubscriptionPlan.name)

        plan_stats_result = await session.execute(plan_stats_stmt)
        plan_breakdown = plan_stats_result.all()

    text = (
        "<b>📊 Расширенная статистика</b>\n\n"
        f"👥 <b>Пользователи:</b>\n"
        f"• Всего в базе: {total_users}\n"
        f"• Активные платные: {active_paid_count}\n"
        f"• На пробном периоде: {active_trials_count}\n"
        f"• Истекшие подписки: {expired_count}\n\n"

        f"💰 <b>Финансы:</b>\n"
        f"• Текущий MRR (активные подписки): {current_mrr:,.2f} руб.\n"
        f"• Доход Robokassa (History): {total_robo_revenue:,.2f} руб.\n\n"

        f"📉 <b>Популярность тарифов (Активные):</b>\n"
    )

    if plan_breakdown:
        for name, count in plan_breakdown:
            text += f"• {name}: {count} шт.\n"
    else:
        text += "• Нет активных тарифов\n"

    await callback.message.edit_text(
        text,
        reply_markup=kb.back_to_previous_menu("admin_payment_settings")
    )


@router.callback_query(F.data == "reset_topic_keep")
async def process_reset_topic_keep(callback: CallbackQuery, state: FSMContext, bot: Bot):
    await state.clear()

    text_to_send = "✅ Память очищена."
    reply_markup = None
    topic_mode_html = "Markdown"

    async with async_session_maker() as session:
        user = await session.get(User, callback.from_user.id, options=[selectinload(User.current_topic)])

        if user:
            ai_config = await session.get(AIConfig, 1)
            memory_mode = get_memory_mode(ai_config) if ai_config else MEMORY_MODE_RESET
            await _new_dialogue_update_state(session, user, user.current_topic_id or 0, memory_mode)

            topic = user.current_topic
            if topic:
                text_to_send = f"✅ Диалог в теме «{topic.name}» перезапущен. Память очищена."
                if topic.start_message:
                    text_to_send = topic.start_message
                    topic_mode_html = "HTML"

                if topic.start_button_text and topic.start_button_payload:
                    reply_markup = kb.action_button_keyboard(topic.start_button_text, "topic_action")

            await session.commit()

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass

    await bot.send_message(
        callback.from_user.id,
        text_to_send,
        parse_mode=topic_mode_html,
        reply_markup=reply_markup
    )


@router.callback_query(F.data == "reset_topic_to_main")
async def process_reset_topic_to_main(callback: CallbackQuery, state: FSMContext, bot: Bot):
    await state.clear()

    async with async_session_maker() as session:
        user = await session.get(User, callback.from_user.id)
        if user:
            user.current_topic_id = None
            ai_config = await session.get(AIConfig, 1)
            memory_mode = get_memory_mode(ai_config) if ai_config else MEMORY_MODE_RESET
            await _apply_topic_switch(session, user, 0, memory_mode)

            await session.commit()

        stmt = select(Content).where(Content.key == "start_message").options(selectinload(Content.media))
        content_obj = await session.scalar(stmt)

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        pass

    main_kb = await kb.main_client_keyboard()
    chat_id = callback.from_user.id

    if not content_obj:
        text = "✅ Вы вернулись в основной режим. Память очищена."
        if is_global_memory_mode(memory_mode):
            text = "✅ Вы вернулись в основной режим. Контекст диалога сохранен."
        await bot.send_message(chat_id, text, reply_markup=main_kb)
        return

    text = content_obj.text_content
    media = content_obj.media
    order = content_obj.content_order

    inline_kb = None
    if content_obj.action_btn_text and content_obj.action_btn_payload:
        inline_kb = kb.action_button_keyboard(content_obj.action_btn_text, "start_action")

    markup_to_send = inline_kb if inline_kb else main_kb

    async def send_text_func():
        if text:
            await bot.send_message(chat_id, text, reply_markup=markup_to_send, parse_mode="HTML")
            return True
        return False

    async def send_media_func():
        if media:
            if len(media) == 1 and len(text or "") < 1000 and order == 'media_top' and text:
                item = media[0]
                if item.file_type == 'photo':
                    await bot.send_photo(chat_id, item.file_id, caption=text, reply_markup=markup_to_send, parse_mode="HTML")
                elif item.file_type == 'video':
                    await bot.send_video(chat_id, item.file_id, caption=text, reply_markup=markup_to_send, parse_mode="HTML")
                return True

            if len(media) == 1:
                item = media[0]
                if item.file_type == 'photo':
                    await bot.send_photo(chat_id, item.file_id, reply_markup=markup_to_send)
                elif item.file_type == 'video':
                    await bot.send_video(chat_id, item.file_id, reply_markup=markup_to_send)
                return True

            media_group = []
            for item in media:
                media_to_add = InputMediaPhoto(media=item.file_id) if item.file_type == 'photo' else InputMediaVideo(
                    media=item.file_id)
                media_group.append(media_to_add)
            if media_group:
                await bot.send_media_group(chat_id, media_group)
                return False
        return False

    sent_combined = False
    media_sent_with_kb = False
    text_sent_with_kb = False

    if media and len(media) == 1 and len(text or "") < 1000 and order == 'media_top':
        sent_combined = await send_media_func()

    if not sent_combined:
        if order == 'media_top':
            media_sent_with_kb = await send_media_func()
            await asyncio.sleep(0.2)
            text_sent_with_kb = await send_text_func()
        else:
            text_sent_with_kb = await send_text_func()
            await asyncio.sleep(0.2)
            media_sent_with_kb = await send_media_func()

    keyboard_was_sent = sent_combined or media_sent_with_kb or text_sent_with_kb

    if inline_kb:
        await bot.send_message(chat_id, "Главное меню:", reply_markup=main_kb)
    elif not keyboard_was_sent and not inline_kb:
        text = "✅ Вы вернулись в основной режим. Память очищена."
        if is_global_memory_mode(memory_mode):
            text = "✅ Вы вернулись в основной режим. Контекст диалога сохранен."
        await bot.send_message(chat_id, text, reply_markup=main_kb)


@router.message(F.document, Command("upload_random"))
async def cmd_upload_random_messages(message: Message, bot: Bot):
    if message.from_user.id not in OWNER_IDS:
        return

    document = message.document
    if not document.file_name.endswith('.txt'):
        await message.answer("❌ Пожалуйста, отправьте файл .txt")
        return

    file_in_io = io.BytesIO()
    await bot.download(document, destination=file_in_io)
    file_content = file_in_io.read().decode('utf-8')

    messages_list = [line.strip() for line in file_content.split('\n') if line.strip()]

    if not messages_list:
        await message.answer("❌ Файл пуст или формат неверен.")
        return

    async with async_session_maker() as session:

        count = 0
        for text in messages_list:
            session.add(RandomMessage(content=text))
            count += 1

        await session.commit()

    await message.answer(f"✅ Успешно загружено {count} посланий в базу случайных сообщений.")


def format_ai_response_to_html(text: str) -> str:
    if not text:
        return ""

    text = html.escape(text)

    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    text = re.sub(r'###\s*(.*)', r'\n<b>\1</b>', text)
    text = re.sub(r'#/\s*(.*)', r'\n<b>\1</b>', text)

    return text


@router.message(F.photo, StateFilter(None))
async def handle_photo_message(message: Message, state: FSMContext, bot: Bot):
    user_id = message.from_user.id
    processing_msg = None
    typing_task = None

    async def keep_typing():
        try:
            while True:
                await bot.send_chat_action(chat_id=message.chat.id, action="typing")
                await asyncio.sleep(4.5)
        except asyncio.CancelledError:
            pass

    try:
        async with async_session_maker() as session:
            user = await session.get(User, user_id, options=[
                selectinload(User.subscription),
                selectinload(User.current_topic)
            ])

            if not user:
                user = User(
                    id=user_id,
                    username=message.from_user.username,
                    first_name=message.from_user.full_name,
                    is_admin=await is_admin(user_id)
                )
                session.add(user)
                await session.flush()

            sub_config = await session.get(SubscriptionConfig, 1)
            subscriptions_active = sub_config.subscriptions_enabled if sub_config else True
            is_user_admin = await is_admin(user_id)

            if not is_user_admin and subscriptions_active:
                if not user.subscription or user.subscription.end_date < datetime.utcnow():
                    await message.answer(
                        "Чтобы отправлять фото и получать разборы, активируйте подписку.",
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="💳 Оформить подписку",
                                                  callback_data="show_subscription_info_from_chat")]
                        ])
                    )
                    return

            typing_task = asyncio.create_task(keep_typing())
            processing_msg = await message.answer("👀 Тщательно изучаю изображение...")

            ai_config = await session.get(AIConfig, 1)

            current_topic_id = user.current_topic_id

            system_prompt_text = ai_integration._load_configured_system_prompt(
                ai_config,
                user.current_topic.system_prompt if user.current_topic else None
            )
            shared_prompt_block = (getattr(ai_config, 'shared_prompt_block', "") or "").strip()
            base_prompt_parts = [part for part in [system_prompt_text.strip(), shared_prompt_block] if part]
            base_prompt_text = "\n\n".join(base_prompt_parts)

            if not base_prompt_text:
                base_prompt_text = "Ты — профессиональный эксперт. Проанализируй это изображение максимально подробно."

            system_prompt = (
                f"ДАННЫЕ КЛИЕНТА:\n"
                f"{_describe_subscription_status(user.subscription)}\n\n"
                f"{base_prompt_text}\n\n"
                "ИНСТРУКЦИЯ ПО АНАЛИЗУ ФОТО:\n"
                "1. Если пользователь просит ИЗМЕНИТЬ это фото или 'сделать так же', добавь в конце: EDIT_IMG: <prompt on english>.\n"
                "2. Если нужно создать НОВОЕ фото с нуля, добавь в конце: GEN_IMG: <prompt on english>.\n"
                "3. ВАЖНО: Диалог уже начат. НЕ здоровайся, не представляйся и не используй вежливые вступления. Сразу переходи к сути разбора изображения."
            )

            memory_mode = get_memory_mode(ai_config)
            stmt = select(DBMessage).where(
                DBMessage.user_id == user.id,
                DBMessage.dialogue_id == user.current_dialogue_id,
            )
            if not is_global_memory_mode(memory_mode):
                stmt = stmt.where(DBMessage.topic_id == current_topic_id)
            stmt = stmt.options(selectinload(DBMessage.topic)).order_by(DBMessage.timestamp.asc())
            result = await session.execute(stmt)
            history = result.scalars().all()
            history, global_memory_context = ai_integration._build_memory_aware_history(
                history,
                current_topic_id,
                user.current_topic.name if user.current_topic else None,
                memory_mode,
            )
            if global_memory_context:
                system_prompt = f"{system_prompt}\n\n{global_memory_context}"

            photo = message.photo[-1]
            file_info = await bot.get_file(photo.file_id)
            file_content = await bot.download_file(file_info.file_path)
            image_bytes = file_content.read()

            analysis_result = await ai_integration.analyze_image_content(image_bytes, system_prompt, history=history)

            if typing_task:
                typing_task.cancel()

            edit_prompt, clean_text = _extract_ai_directive_payload(analysis_result, "EDIT_IMG")
            gen_prompt, clean_text = _extract_ai_directive_payload(clean_text, "GEN_IMG")

            formatted_html = markdown_to_html(clean_text)

            if processing_msg:
                try:
                    await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
                except Exception:
                    pass

            if formatted_html:
                parts = split_html_text(formatted_html)
                for part in parts:
                    await _safe_send_html(
                        lambda text, pm: message.answer(text, parse_mode=pm),
                        part,
                    )

            if edit_prompt:
                m_gen_status = await message.answer("🎨 Редактирую ваше фото...")
                edited_data = await ai_integration.edit_image(edit_prompt, image_bytes)
                if edited_data:
                    upload_task = _start_chat_action_loop(bot, message.chat.id, "upload_photo")
                    try:
                        await message.answer_photo(photo=BufferedInputFile(edited_data, filename="edited.png"), caption="✨ Результат редактирования:")
                    finally:
                        upload_task.cancel()
                else:
                    await message.answer("😔 К сожалению, не удалось отредактировать изображение. Возможно, сервис дал сбой или запрос был отклонен фильтрами безопасности.")
                try:
                    await bot.delete_message(chat_id=message.chat.id, message_id=m_gen_status.message_id)
                except Exception:
                    pass

            elif gen_prompt:
                m_gen_status = await message.answer("🖼 Генерирую новое изображение...")
                new_img = await ai_integration.generate_image(gen_prompt)
                if new_img:
                    upload_task = _start_chat_action_loop(bot, message.chat.id, "upload_photo")
                    try:
                        await message.answer_photo(photo=BufferedInputFile(new_img, filename="generated.png"), caption="✨ Новая генерация:")
                    finally:
                        upload_task.cancel()
                try:
                    await bot.delete_message(chat_id=message.chat.id, message_id=m_gen_status.message_id)
                except Exception:
                    pass

            session.add(DBMessage(user_id=user_id, role='user', content="[Фото для анализа]", dialogue_id=user.current_dialogue_id, topic_id=current_topic_id))
            session.add(DBMessage(user_id=user_id, role='assistant', content=clean_text, dialogue_id=user.current_dialogue_id, topic_id=current_topic_id))
            await session.commit()

    except AIServiceError as e:
        if typing_task:
            typing_task.cancel()
        vision_provider, vision_model = _resolve_ai_provider_model(ai_config if 'ai_config' in locals() else None, "vision")
        if processing_msg:
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
            except Exception:
                pass
        await message.answer(
            "Упс... У нас что-то сломалось. Мы уже сообщили нашим создателям. Попробуйте вернуться и повторить через несколько минут."
        )
        await _report_ai_failure(
            bot,
            title="Сбой анализа изображения",
            user=message.from_user,
            provider=vision_provider,
            model=vision_model,
            stage="photo_handler",
            details=str(e),
            exception=e,
        )
    except Exception as e:
        if typing_task:
            typing_task.cancel()
        provider, model = _resolve_ai_provider_model(ai_config if 'ai_config' in locals() else None, "vision")
        if processing_msg:
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
            except Exception:
                pass
        await _report_ai_failure(
            bot,
            title="Непредвиденная ошибка обработки фото",
            user=message.from_user,
            provider=provider,
            model=model,
            stage="photo_handler_unexpected",
            details=str(e),
            exception=e,
        )
        await message.answer("Произошла ошибка при обработке фото.")


@router.callback_query(F.data.startswith("admin_topic_media_"))
async def admin_topic_media_list(callback: CallbackQuery):
    MEDIA_PAGE_SIZE = 10
    parts = callback.data.split("_")
    topic_id = int(parts[3])
    page = int(parts[4]) if len(parts) > 4 else 0

    async with async_session_maker() as session:
        total_count = await session.scalar(
            select(func.count()).select_from(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
        )
        stmt = (
            select(MediaLibrary)
            .where(MediaLibrary.topic_id == topic_id)
            .order_by(MediaLibrary.id)
            .offset(page * MEDIA_PAGE_SIZE)
            .limit(MEDIA_PAGE_SIZE)
        )
        result = await session.execute(stmt)
        media_list = result.scalars().all()
        topic = await session.get(Topic, topic_id)

        # Проверяем какие категории есть и у каких есть _back
        cat_stmt = select(MediaLibrary.category, MediaLibrary.file_name).where(
            MediaLibrary.topic_id == topic_id,
            MediaLibrary.category != None,
            MediaLibrary.category != ''
        )
        cat_res = await session.execute(cat_stmt)
        cat_rows = cat_res.all()
        categories = set()
        back_categories = set()
        for cat, fname in cat_rows:
            categories.add(cat)
            if fname == '_back':
                back_categories.add(cat)

    total_pages = max(1, (total_count + MEDIA_PAGE_SIZE - 1) // MEDIA_PAGE_SIZE)

    cats_info = ""
    if categories:
        cat_lines = []
        for cat in sorted(categories):
            has_back = "🃏" if cat in back_categories else "⚠️ нет рубашки"
            cat_lines.append(f"  <code>{cat}</code> — {has_back}")
        cats_info = "\n<b>Категории:</b>\n" + "\n".join(cat_lines) + "\n"

    text = (
        f"📁 Медиа-библиотека темы: <b>{topic.name}</b>\n"
        f"Файлов: {total_count}"
        f"{cats_info}\n"
        f"<b>Теги для AI:</b>\n"
        f"<code>[RANDOM_IMG: категория]</code> — случайная карта\n"
        f"<code>[RANDOM_IMG: категория | N]</code> — N случайных карт сразу\n"
        f"<code>[CHOICE_IMG: категория | N]</code> — выбор из N (лицом)\n"
        f"<code>[CHOICE_IMG: категория | N | R]</code> — расклад из R карт, выбор из N\n"
        f"<code>[CHOICE_IMG_HIDDEN: категория | N]</code> — выбор из N (рубашкой)\n"
        f"<code>[CHOICE_IMG_HIDDEN: категория | N | R]</code> — расклад из R карт вслепую\n"
        f"<code>[SHOW_IMG: имя_файла]</code> — конкретная карта\n"
        f"<code>[SEND_AUDIO: имя_файла]</code> — аудиофайл\n\n"
        f"🃏 Для скрытого выбора добавьте файл с именем <code>_back</code> в нужную категорию."
    )
    kb = keyboards.topic_media_manage_keyboard(topic_id, media_list, page, total_pages)

    if callback.message.text:
        await callback.message.edit_text(text, reply_markup=kb, parse_mode='HTML')
    else:
        await callback.message.answer(text, reply_markup=kb, parse_mode='HTML')
        await callback.message.delete()


@router.callback_query(F.data.startswith("admin_media_view_"))
async def admin_media_view(callback: CallbackQuery):
    media_id = int(callback.data.split("_")[3])
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if not media:
            await callback.answer("Файл не найден.", show_alert=True)
            return
        # collections this file belongs to
        coll_res = await session.execute(
            select(MediaCollection.name).join(
                media_collection_items,
                media_collection_items.c.collection_id == MediaCollection.id
            ).where(media_collection_items.c.media_id == media_id)
        )
        coll_names = [r[0] for r in coll_res.all()]

    role_hint = ""
    if media.file_name == '_back':
        role_hint = f"\n🃏 <b>Рубашка</b> для категории <code>{media.category}</code>"

    colls_text = ", ".join(coll_names) if coll_names else "нет"
    text = (
        f"<b>📄 Данные файла:</b>\n"
        f"ID: <code>{media.id}</code>\n"
        f"Имя для AI: <code>{media.file_name}</code>\n"
        f"Тип: {media.media_type}\n"
        f"Категория: {media.category or 'Не задана'}\n"
        f"Коллекции: {colls_text}\n"
        f"Описание: {media.description or 'Нет'}"
        f"{role_hint}"
    )

    kb = keyboards.media_edit_keyboard(media.id, media.topic_id)
    try:
        if media.media_type == 'audio':
            await callback.message.answer_audio(
                audio=media.file_id, caption=text, reply_markup=kb, parse_mode='HTML'
            )
        else:
            await callback.message.answer_photo(
                photo=media.file_id, caption=text, reply_markup=kb, parse_mode='HTML'
            )
    except Exception:
        await callback.message.answer_document(
            document=media.file_id, caption=text, reply_markup=kb, parse_mode='HTML'
        )

    await callback.message.delete()
    await callback.answer()


@router.callback_query(F.data.startswith("admin_media_editname_"))
async def admin_media_edit_name_start(callback: CallbackQuery, state: FSMContext):
    media_id = int(callback.data.split("_")[3])
    await state.update_data(edit_media_id=media_id)
    await state.set_state(AdminMediaState.editing_name)
    await callback.message.answer(
        "Введи новое <b>техническое имя</b> для файла (на английском, без пробелов):",
        parse_mode='HTML'
    )
    await callback.answer()


@router.message(AdminMediaState.editing_name)
async def admin_media_edit_name_finish(message: Message, state: FSMContext):
    data = await state.get_data()
    media_id = data['edit_media_id']
    new_name = message.text.strip().lower().replace(" ", "_")

    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.file_name = new_name
            await session.commit()
            topic_id = media.topic_id
    await state.clear()
    await message.answer(f"✅ Имя изменено на <code>{new_name}</code>.", parse_mode='HTML')


@router.callback_query(F.data.startswith("admin_media_editcat_"))
async def admin_media_edit_category_start(callback: CallbackQuery, state: FSMContext):
    media_id = int(callback.data.split("_")[3])
    await state.update_data(edit_media_id=media_id)
    await state.set_state(AdminMediaState.editing_category)
    await callback.message.answer(
        "Введи новую <b>категорию</b> (например: <code>tarot</code>, <code>mak</code>, <code>oracle</code>):",
        parse_mode='HTML'
    )
    await callback.answer()


@router.message(AdminMediaState.editing_category)
async def admin_media_edit_category_finish(message: Message, state: FSMContext):
    data = await state.get_data()
    media_id = data['edit_media_id']
    new_category = message.text.strip().lower().replace(" ", "_")

    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.category = new_category
            await session.commit()
    await state.clear()
    await message.answer(f"✅ Категория изменена на <code>{new_category}</code>.", parse_mode='HTML')


@router.callback_query(F.data.startswith("admin_media_editdesc_"))
async def admin_media_edit_desc_start(callback: CallbackQuery, state: FSMContext):
    media_id = int(callback.data.split("_")[3])
    await state.update_data(edit_media_id=media_id)
    await state.set_state(AdminMediaState.editing_description)
    await callback.message.answer(
        "Введи новое <b>описание</b> для файла:",
        parse_mode='HTML'
    )
    await callback.answer()


@router.message(AdminMediaState.editing_description)
async def admin_media_edit_desc_finish(message: Message, state: FSMContext):
    data = await state.get_data()
    media_id = data['edit_media_id']

    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.description = message.text.strip()
            await session.commit()
    await state.clear()
    await message.answer("✅ Описание обновлено.")


@router.callback_query(F.data.startswith("admin_media_editfile_"))
async def admin_media_edit_file_start(callback: CallbackQuery, state: FSMContext):
    media_id = int(callback.data.split("_")[3])
    await state.update_data(edit_media_id=media_id)
    await state.set_state(AdminMediaState.editing_file)
    await callback.message.answer("Отправь новый файл (фото или аудио) для замены:")
    await callback.answer()


@router.message(AdminMediaState.editing_file, F.photo | F.audio | F.voice | F.document)
async def admin_media_edit_file_finish(message: Message, state: FSMContext):
    data = await state.get_data()
    media_id = data['edit_media_id']

    if message.photo:
        file_id = message.photo[-1].file_id
        m_type = "photo"
    elif message.audio:
        file_id = message.audio.file_id
        m_type = "audio"
    elif message.voice:
        file_id = message.voice.file_id
        m_type = "audio"
    elif message.document:
        file_id = message.document.file_id
        m_type = "photo" if message.document.mime_type and message.document.mime_type.startswith('image/') else "audio"
    else:
        await message.answer("Отправь фото или аудиофайл.")
        return

    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.file_id = file_id
            media.media_type = m_type
            await session.commit()
    await state.clear()
    await message.answer("✅ Файл заменён.")


@router.callback_query(F.data.startswith("admin_media_add_"))
async def admin_media_add_start(callback: CallbackQuery, state: FSMContext):
    topic_id = int(callback.data.split("_")[3])
    await state.update_data(target_topic_id=topic_id)
    await state.set_state(AdminMediaState.waiting_for_file)
    await callback.message.edit_text("Отправь мне файл (фото или аудио), который хочешь добавить в эту тему.")


@router.message(AdminMediaState.waiting_for_file, F.photo | F.audio | F.voice | F.document)
async def admin_media_file_receive(message: Message, state: FSMContext):
    m_type = ""
    file_id = ""

    if message.photo:
        m_type = "photo"
        file_id = message.photo[-1].file_id
    elif message.audio:
        m_type = "audio"
        file_id = message.audio.file_id
    elif message.voice:
        m_type = "audio"
        file_id = message.voice.file_id
    elif message.document:
        m_type = "photo" if message.document.mime_type.startswith('image/') else "audio"
        file_id = message.document.file_id

    await state.update_data(m_file_id=file_id, m_type=m_type)
    await state.set_state(AdminMediaState.waiting_for_name)

    await message.answer(
        f"✅ <b>Файл получен как {m_type}!</b>\n\n"
        f"Теперь придумай короткое <b>техническое имя</b> для этого файла на английском (например: <code>morning_meditation</code>, <code>card_death</code>).\n\n"
        f"⚠️ <b>ВАЖНО:</b> Это имя вы будете вставлять в системный промпт в формате <code>[SEND_AUDIO: имя]</code>, чтобы AI отправил этот файл пользователю.",
        parse_mode='HTML'
    )


@router.message(AdminMediaState.waiting_for_name)
async def admin_media_name_receive(message: Message, state: FSMContext):
    tech_name = message.text.strip().lower().replace(" ", "_")
    await state.update_data(m_name=tech_name)
    data = await state.get_data()
    m_type = data.get('m_type', '')

    if m_type == 'photo':
        await state.set_state(AdminMediaState.waiting_for_category)
        await message.answer(
            f"👌 Имя <code>{tech_name}</code> принято.\n\n"
            f"Теперь введи <b>категорию</b> для этого изображения (например: <code>tarot</code>, <code>mak</code>, <code>oracle</code>).\n\n"
            f"Категория используется для группировки — AI будет случайно выбирать карту из всех файлов одной категории.\n"
            f"Если карты из одной колоды — используй одинаковую категорию для всех.",
            parse_mode='HTML'
        )
    else:
        await state.set_state(AdminMediaState.waiting_for_description)
        await message.answer(
            f"👌 Имя <code>{tech_name}</code> принято.\n\n"
            f"Теперь введи описание файла.\n\n"
            f"Для аудио — опиши, в какой момент AI должен предложить эту практику пользователю.",
            parse_mode='HTML'
        )


@router.message(AdminMediaState.waiting_for_category)
async def admin_media_category_receive(message: Message, state: FSMContext):
    category = message.text.strip().lower().replace(" ", "_")
    await state.update_data(m_category=category)
    await state.set_state(AdminMediaState.waiting_for_description)
    await message.answer(
        f"👌 Категория <code>{category}</code> принята.\n\n"
        f"Теперь введи описание карты — это будет её трактовка, которую AI учтёт при интерпретации.",
        parse_mode='HTML'
    )


@router.message(AdminMediaState.waiting_for_description)
async def admin_media_final(message: Message, state: FSMContext):
    data = await state.get_data()
    topic_id = data['target_topic_id']
    m_type = data['m_type']
    m_name = data['m_name']

    category = data.get('m_category', '')

    async with async_session_maker() as session:
        new_media = MediaLibrary(
            topic_id=topic_id,
            media_type=m_type,
            file_id=data['m_file_id'],
            file_name=m_name,
            category=category,
            description=message.text.strip()
        )
        session.add(new_media)

        # Автопривязка колоды к текущему топику
        if category and topic_id:
            existing_deck = await session.execute(
                select(TopicMediaDeck).where(
                    TopicMediaDeck.topic_id == topic_id,
                    TopicMediaDeck.deck_name == category
                )
            )
            if not existing_deck.scalar_one_or_none():
                session.add(TopicMediaDeck(topic_id=topic_id, deck_name=category))

        await session.commit()

        MEDIA_PAGE_SIZE = 10
        total_count = await session.scalar(
            select(func.count()).select_from(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
        )
        total_pages = max(1, (total_count + MEDIA_PAGE_SIZE - 1) // MEDIA_PAGE_SIZE)
        last_page = total_pages - 1
        stmt = (
            select(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
            .order_by(MediaLibrary.id)
            .offset(last_page * MEDIA_PAGE_SIZE).limit(MEDIA_PAGE_SIZE)
        )
        result = await session.execute(stmt)
        media_list = result.scalars().all()
        topic = await session.get(Topic, topic_id)

    await state.clear()

    if m_name == '_back':
        usage_hint = (
            f"🃏 Это рубашка для категории <code>{category}</code>.\n"
            f"Теперь AI может использовать скрытый выбор:\n"
            f"<code>[CHOICE_IMG_HIDDEN: {category} | 3]</code>"
        )
    elif m_type == 'audio':
        usage_hint = f"<code>[SEND_AUDIO: {m_name}]</code>"
    else:
        usage_hint = (
            f"<code>[RANDOM_IMG: {category}]</code> — одна случайная карта\n"
            f"<code>[RANDOM_IMG: {category} | 5]</code> — 5 случайных карт сразу\n"
            f"<code>[CHOICE_IMG: {category} | 3]</code> — выбор из 3 карт (лицом)\n"
            f"<code>[CHOICE_IMG: {category} | 3 | 5]</code> — расклад из 5 карт, выбор из 3\n"
            f"<code>[CHOICE_IMG_HIDDEN: {category} | 3]</code> — выбор вслепую (рубашкой)\n"
            f"<code>[CHOICE_IMG_HIDDEN: {category} | 3 | 5]</code> — расклад из 5 вслепую\n"
            f"<code>[SHOW_IMG: {m_name}]</code> — показать именно эту карту"
        )

    await message.answer(
        f"✅ <b>Файл успешно добавлен!</b>\n\n"
        f"AI автоматически знает об этом файле и может использовать его через теги:\n"
        f"{usage_hint}\n\n"
        f"Файл привязан к теме: <b>{topic.name}</b>",
        parse_mode='HTML'
    )

    await message.answer(
        f"📁 Медиа-библиотека темы: <b>{topic.name}</b>\nФайлов: {total_count}",
        reply_markup=keyboards.topic_media_manage_keyboard(topic_id, media_list, last_page, total_pages),
        parse_mode='HTML'
    )


@router.callback_query(F.data.startswith("admin_media_delete_"))
async def admin_media_delete(callback: CallbackQuery):
    media_id = int(callback.data.split("_")[3])
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if not media:
            await callback.answer("Файл не найден.")
            return

        MEDIA_PAGE_SIZE = 10
        topic_id = media.topic_id
        await session.delete(media)
        await session.commit()

        total_count = await session.scalar(
            select(func.count()).select_from(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
        )
        total_pages = max(1, (total_count + MEDIA_PAGE_SIZE - 1) // MEDIA_PAGE_SIZE)
        stmt = (
            select(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
            .order_by(MediaLibrary.id).limit(MEDIA_PAGE_SIZE)
        )
        res = await session.execute(stmt)
        media_list = res.scalars().all()
        topic = await session.get(Topic, topic_id)

    await callback.answer("Файл удален")

    text = (
        f"📁 Медиа-библиотека темы: <b>{topic.name}</b>\n"
        f"Файлов: {total_count}"
    )
    kb = keyboards.topic_media_manage_keyboard(topic_id, media_list, 0, total_pages)

    await callback.message.answer(text, reply_markup=kb, parse_mode='HTML')
    await callback.message.delete()


# ── Реферальная кнопка меню — регистрируется ДО AI-хэндлера ──
@router.message(ReferralButtonFilter())
async def show_referral_info(message: Message, bot: Bot):
    """Показывает экран реферальной программы при нажатии кнопки в меню."""
    text = await _get_referral_screen_text(message.from_user.id, bot)
    await message.answer(text, parse_mode="HTML", disable_web_page_preview=True)


@router.callback_query(F.data == "referral_sub_info")
async def show_referral_from_sub(callback: CallbackQuery, bot: Bot):
    """Показывает экран реферальной программы из меню подписки."""
    text = await _get_referral_screen_text(callback.from_user.id, bot)
    await callback.message.answer(text, parse_mode="HTML", disable_web_page_preview=True)
    await callback.answer()


@router.message(F.text, StateFilter(None))
async def handle_ai_chat(message: Message, state: FSMContext, bot: Bot):
    user_id = message.from_user.id

    async with async_session_maker() as session:
        user = await session.get(User, user_id, options=[selectinload(User.subscription)])

        if not user:
            user = User(id=user_id, username=message.from_user.username, first_name=message.from_user.full_name,
                        is_admin=await is_admin(user_id))
            session.add(user)
            await session.flush()
        else:
            user.username = message.from_user.username
            user.first_name = message.from_user.full_name

        await _sync_user_birthdate_from_telegram(bot, user)
        await session.commit()
        await session.refresh(user, ['subscription'])

        sub_config = await session.get(SubscriptionConfig, 1)
        subscriptions_active = sub_config.subscriptions_enabled if sub_config else True
        is_user_admin = await is_admin(user_id)

        if not is_user_admin and subscriptions_active:
            if not user.subscription or user.subscription.end_date < datetime.utcnow():
                await message.answer(
                    "Чтобы продолжить диалог, активируйте подписку / бонусные дни.",
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(text="Начать пользоваться ботом",
                                              callback_data="show_subscription_info_from_chat")]
                    ])
                )
                return

        if not user.name:
            await state.set_state(UserStates.awaiting_name)
            await state.update_data(initial_prompt=message.text)
            await message.answer("Прежде чем мы начнем, подскажите, как я могу к вам обращаться?")
            return

        if not user.accepted_disclaimer:
            disclaimer_content = await get_content_from_db("disclaimer")
            if disclaimer_content.get('is_visible', True):
                await state.set_state(UserStates.awaiting_disclaimer_acceptance)
                await state.update_data(initial_prompt=message.text)
                text_to_send = disclaimer_content.get('text') or "Текст дисклеймера не задан."
                await message.answer(text_to_send, reply_markup=kb.confirm_disclaimer_keyboard())
                return
            else:
                stmt = update(User).where(User.id == user_id).values(accepted_disclaimer=True)
                await session.execute(stmt)
                await session.commit()

    if user_id not in user_message_buffers:
        user_message_buffers[user_id] = []

    user_message_buffers[user_id].append(message.text)

    if user_id in user_processing_tasks and not user_processing_tasks[user_id].done():
        return

    async def debounced_process():
        await asyncio.sleep(0.8)
        await process_buffered_messages(user_id, bot)

    user_processing_tasks[user_id] = asyncio.create_task(debounced_process())


@router.callback_query(F.data == "admin_export_mode_start")
async def admin_export_mode_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.selecting_for_export)
    await state.update_data(selected_export_ids=[])
    await admin_clients_list(callback, state)


@router.callback_query(F.data.startswith("toggle_export_"))
async def toggle_client_selection(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    user_id = int(parts[2])
    page = int(parts[3])

    data = await state.get_data()
    selected_ids = data.get("selected_export_ids", [])

    if selected_ids is None:
        selected_ids = []

    if user_id in selected_ids:
        selected_ids.remove(user_id)
    else:
        selected_ids.append(user_id)

    await state.update_data(selected_export_ids=selected_ids)

    new_callback = callback.model_copy(update={"data": f"admin_export_page_{page}"})

    await admin_clients_list(new_callback, state)


@router.callback_query(F.data.startswith("toggle_export_"))
async def toggle_export_selection(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    user_id = int(parts[2])
    page = int(parts[3])

    data = await state.get_data()
    selected_ids = data.get("selected_export_ids", [])
    if selected_ids is None:
        selected_ids = []

    if user_id in selected_ids:
        selected_ids.remove(user_id)
    else:
        selected_ids.append(user_id)

    await state.update_data(selected_export_ids=selected_ids)

    callback_mock = type('obj', (object,), {
        'message': callback.message,
        'data': f"admin_export_page_{page}",
        'from_user': callback.from_user,
        'answer': callback.answer
    })()

    await admin_clients_list(callback_mock, state)


@router.callback_query(F.data == "export_select_all_no_admins")
async def export_select_all_no_admins(callback: CallbackQuery, state: FSMContext):
    async with async_session_maker() as session:
        result = await session.execute(
            select(User.id).where(
                User.is_admin == False,
                User.id.notin_(OWNER_IDS)
            )
        )
        all_ids = [row[0] for row in result]
    await state.update_data(selected_export_ids=all_ids)
    new_callback = callback.model_copy(update={"data": "admin_export_page_0"})
    await admin_clients_list(new_callback, state)
    await callback.answer(f"Выбрано {len(all_ids)} пользователей (без админов)")


@router.callback_query(F.data == "admin_export_all_confirm")
async def admin_export_all_confirm(callback: CallbackQuery, state: FSMContext):
    await state.update_data(export_all=True, export_date_from=None, export_date_to=None)
    await callback.message.edit_text(
        "📅 Фильтр по датам для ВСЕХ пользователей:\n\nВыберите диапазон сообщений для экспорта:",
        reply_markup=kb.export_date_filter_keyboard()
    )


@router.callback_query(F.data == "admin_export_confirm_options")
async def admin_export_confirm_options(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    selected_ids = data.get("selected_export_ids", [])
    if not selected_ids:
        await callback.answer("Вы не выбрали ни одного клиента!", show_alert=True)
        return
    await state.update_data(export_all=False, export_date_from=None, export_date_to=None)
    await callback.message.edit_text(
        f"📅 Фильтр по датам для {len(selected_ids)} пользователей:\n\nВыберите диапазон сообщений для экспорта:",
        reply_markup=kb.export_date_filter_keyboard()
    )


@router.callback_query(F.data.startswith("export_date_preset_"))
async def export_date_preset(callback: CallbackQuery, state: FSMContext):
    days = int(callback.data.split("_")[-1])
    if days == 0:
        await state.update_data(export_date_from=None, export_date_to=None)
        date_label = "без фильтра (все даты)"
    else:
        date_from = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")
        await state.update_data(export_date_from=date_from, export_date_to=None)
        date_label = f"за последние {days} дней (с {date_from})"

    data = await state.get_data()
    export_all = data.get("export_all", False)
    selected_ids = data.get("selected_export_ids", [])
    target = "ВСЕХ пользователей" if export_all else f"{len(selected_ids)} пользователей"

    await callback.message.edit_text(
        f"✅ Фильтр дат: {date_label}\n\nВыберите формат экспорта для {target}:",
        reply_markup=kb.mass_export_options_keyboard()
    )


@router.callback_query(F.data == "export_date_manual")
async def export_date_manual(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.export_date_from)
    await callback.message.edit_text(
        "✏️ Введите дату начала в формате <b>ДД-ММ-ГГГГ</b>\n"
        "Пример: <code>01-09-2025</code>\n\n"
        "Или отправьте <code>0</code> чтобы не ограничивать начало."
    )


@router.message(AdminStates.export_date_from)
async def export_date_from_input(message: Message, state: FSMContext):
    text = message.text.strip()
    if text == "0":
        await state.update_data(export_date_from=None)
    else:
        try:
            datetime.strptime(text, "%d-%m-%Y")
            await state.update_data(export_date_from=text)
        except ValueError:
            await message.answer("❌ Неверный формат. Введите дату как <code>ДД-ММ-ГГГГ</code>, например <code>01-09-2025</code>")
            return

    await state.set_state(AdminStates.export_date_to)
    await message.answer(
        "✏️ Теперь введите дату окончания в формате <b>ДД-ММ-ГГГГ</b>\n"
        "Пример: <code>31-12-2025</code>\n\n"
        "Или отправьте <code>0</code> чтобы не ограничивать конец (до сегодня)."
    )


@router.message(AdminStates.export_date_to)
async def export_date_to_input(message: Message, state: FSMContext):
    text = message.text.strip()
    if text == "0":
        await state.update_data(export_date_to=None)
    else:
        try:
            datetime.strptime(text, "%d-%m-%Y")
            await state.update_data(export_date_to=text)
        except ValueError:
            await message.answer("❌ Неверный формат. Введите дату как <code>ДД-ММ-ГГГГ</code>, например <code>31-12-2025</code>")
            return

    await state.set_state(AdminStates.selecting_for_export)
    data = await state.get_data()
    date_from = data.get("export_date_from", "начала")
    date_to = data.get("export_date_to", "сегодня")
    export_all = data.get("export_all", False)
    selected_ids = data.get("selected_export_ids", [])
    target = "ВСЕХ пользователей" if export_all else f"{len(selected_ids)} пользователей"

    await message.answer(
        f"✅ Фильтр дат: с <b>{date_from}</b> по <b>{date_to}</b>\n\nВыберите формат экспорта для {target}:",
        reply_markup=kb.mass_export_options_keyboard()
    )


@router.callback_query(F.data.startswith("run_export_"))
async def process_mass_export(callback: CallbackQuery, state: FSMContext, bot: Bot):
    parts = callback.data.split("_")
    fmt = parts[2]
    anonymize = parts[3] == "yes"

    data = await state.get_data()
    export_all = data.get("export_all", False)

    date_from_str = data.get("export_date_from")
    date_to_str = data.get("export_date_to")
    date_from_dt = datetime.strptime(date_from_str, "%d-%m-%Y") if date_from_str else None
    date_to_dt = datetime.strptime(date_to_str, "%d-%m-%Y").replace(hour=23, minute=59, second=59) if date_to_str else None

    date_label = ""
    if date_from_dt or date_to_dt:
        date_label = f" | фильтр: {date_from_str or 'начало'} → {date_to_str or 'сегодня'}"

    await callback.message.edit_text(f"⏳ Начинаю сбор данных и формирование файла{date_label}. Это может занять время...")

    async with async_session_maker() as session:
        if export_all:
            users_res = await session.execute(select(User))
            users = users_res.scalars().all()
        else:
            selected_ids = data.get("selected_export_ids", [])
            users_res = await session.execute(select(User).where(User.id.in_(selected_ids)))
            users = users_res.scalars().all()

        if not users:
            await callback.message.edit_text("Ошибка: Пользователи не найдены.")
            return

        topics_res = await session.execute(select(Topic))
        topic_map = {t.id: t.name for t in topics_res.scalars().all()}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        def build_msg_stmt(user_id):
            stmt = select(DBMessage).where(DBMessage.user_id == user_id)
            if date_from_dt:
                stmt = stmt.where(DBMessage.timestamp >= date_from_dt)
            if date_to_dt:
                stmt = stmt.where(DBMessage.timestamp <= date_to_dt)
            return stmt.order_by(DBMessage.timestamp)

        if fmt == "txt":
            full_content = f"MASS EXPORT - {timestamp}{date_label}\n"
            full_content += "=" * 60 + "\n\n"

            for i, user in enumerate(users, 1):
                user_label = f"user_{i}" if anonymize else f"{user.name or user.first_name} (ID: {user.id}, @{user.username})"

                messages = (await session.execute(build_msg_stmt(user.id))).scalars().all()

                if not messages:
                    continue

                full_content += f"ДАННЫЕ КЛИЕНТА: {user_label}\n"
                full_content += "-" * 40 + "\n"

                for m in messages:
                    t_name = topic_map.get(m.topic_id, "General")
                    role = "Client" if m.role == "user" else "Bot"
                    full_content += f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}] [{t_name}] {role}: {m.content}\n"

                full_content += "\n" + "=" * 60 + "\n\n"

            final_bytes = full_content.encode("utf-8")
            extension = "txt"

        else:
            export_data = []
            for i, user in enumerate(users, 1):
                user_label = f"user_{i}" if anonymize else str(user.id)

                messages = (await session.execute(build_msg_stmt(user.id))).scalars().all()

                if not messages:
                    continue

                user_history = {
                    "user_info": {
                        "label": user_label,
                        "id": None if anonymize else user.id,
                        "name": None if anonymize else (user.name or user.first_name),
                        "username": None if anonymize else user.username
                    },
                    "history": []
                }

                for m in messages:
                    user_history["history"].append({
                        "timestamp": m.timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                        "topic": topic_map.get(m.topic_id, "General"),
                        "role": m.role,
                        "content": m.content
                    })

                export_data.append(user_history)

            final_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")
            extension = "json"

    if not final_bytes:
        await callback.message.edit_text("История переписок пуста у выбранных пользователей.")
        return

    limit_50mb = 50 * 1024 * 1024

    if len(final_bytes) > limit_50mb:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"export_{timestamp}.{extension}", final_bytes)

        zip_buffer.seek(0)
        input_file = BufferedInputFile(zip_buffer.read(), filename=f"export_{timestamp}.zip")
        caption = "📦 Файл превысил 50МБ и был сжат в ZIP."
    else:
        input_file = BufferedInputFile(final_bytes, filename=f"export_{timestamp}.{extension}")
        caption = f"✅ Экспорт завершен успешно ({extension})."

    await callback.message.answer_document(input_file, caption=caption)
    await bot.delete_message(callback.message.chat.id, callback.message.message_id)
    await state.clear()


@router.callback_query(F.data == "admin_change_vision_model")
async def admin_change_vision_model_list(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        provider = config.vision_provider

    builder = InlineKeyboardBuilder()
    if provider == "Gemini":
        models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-3-flash-preview"]
    elif provider == "KIE":
        models = ["gemini-2.5-flash", "gemini-3-flash"]
    elif provider == "Claude":
        models = [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            "claude-haiku-4-5-20251001",
        ]
    else:
        models = ["gpt-4o", "gpt-4o-mini"]

    for m in models:
        builder.button(text=m, callback_data=f"save_vision_model_{m}")

    builder.button(text="⬅️ Назад", callback_data="admin_ai_keys")
    builder.adjust(1)

    await callback.message.edit_text(f"Выберите модель для {provider}:", reply_markup=builder.as_markup())


@router.callback_query(F.data.startswith("save_vision_model_"))
async def save_vision_model(callback: CallbackQuery):
    model_name = callback.data.replace("save_vision_model_", "")
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        config.vision_model = model_name
        await session.commit()

    await callback.answer(f"✅ Модель для фото изменена на {model_name}")


@router.callback_query(F.data == "admin_change_image_generation_model")
async def admin_change_image_generation_model_list(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        provider = getattr(config, "image_generation_provider", "Gemini")

    builder = InlineKeyboardBuilder()
    if provider == "Gemini":
        models = ["imagen-4.0-generate-001"]
    elif provider == "KIE":
        models = [
            "seedream/4.5-text-to-image",
            "bytedance/seedream-v4-text-to-image",
            "google/imagen4-fast",
            "google/imagen4-ultra",
        ]
    else:
        models = ["gpt-image-1.5"]

    for m in models:
        builder.button(text=m, callback_data=f"save_image_generation_model_{m}")

    builder.button(text="⬅️ Назад", callback_data="admin_ai_keys")
    builder.adjust(1)
    await callback.message.edit_text(f"Выберите модель генерации для {provider}:", reply_markup=builder.as_markup())


@router.callback_query(F.data.startswith("save_image_generation_model_"))
async def save_image_generation_model(callback: CallbackQuery):
    model_name = callback.data.replace("save_image_generation_model_", "")
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        config.image_generation_model = model_name
        await session.commit()
    await callback.answer(f"✅ Модель генерации изменена на {model_name}")


@router.callback_query(F.data == "admin_change_image_edit_model")
async def admin_change_image_edit_model_list(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        provider = getattr(config, "image_edit_provider", "Gemini")

    builder = InlineKeyboardBuilder()
    if provider == "Gemini":
        models = ["gemini-3-pro-image-preview"]
    else:
        models = [
            "seedream/4.5-edit",
            "bytedance/seedream-v4-edit",
            "google/nano-banana-edit",
        ]

    for m in models:
        builder.button(text=m, callback_data=f"save_image_edit_model_{m}")

    builder.button(text="⬅️ Назад", callback_data="admin_ai_keys")
    builder.adjust(1)
    await callback.message.edit_text(f"Выберите модель редактирования для {provider}:", reply_markup=builder.as_markup())


@router.callback_query(F.data.startswith("save_image_edit_model_"))
async def save_image_edit_model(callback: CallbackQuery):
    model_name = callback.data.replace("save_image_edit_model_", "")
    async with async_session_maker() as session:
        config = await session.get(AIConfig, 1)
        config.image_edit_model = model_name
        await session.commit()
    await callback.answer(f"✅ Модель редактирования изменена на {model_name}")


# ══════════════════════════════════════════════════════════════
#  РЕФЕРАЛЬНАЯ ПРОГРАММА — пользовательские хэндлеры
# ══════════════════════════════════════════════════════════════

async def _get_referral_screen_text(user_id: int, bot: Bot) -> str:
    """Формирует текст экрана реферальной программы для пользователя."""
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if not config or not config.referral_enabled:
            return "Реферальная программа недоступна."

        count_result = await session.execute(
            select(func.count()).select_from(User).where(User.referred_by == user_id)
        )
        referral_count = count_result.scalar() or 0

    bot_info = await bot.get_me()
    link = f"https://t.me/{bot_info.username}?start=ref_{user_id}"

    text = (
        f"🔗 <b>Реферальная программа</b>\n\n"
        f"Пригласите друга по ссылке и получите бонусные дни!\n\n"
        f"<b>Ваша ссылка:</b>\n{link}\n\n"
        f"👥 <b>Приглашено:</b> {referral_count} чел.\n\n"
        f"За каждого зарегистрировавшегося по ссылке — "
        f"вы и ваш друг получаете по <b>{config.referral_bonus_days_referrer} дн.</b> бонуса."
    )
    return text


# ══════════════════════════════════════════════════════════════
#  РЕФЕРАЛЬНАЯ ПРОГРАММА — admin хэндлеры
# ══════════════════════════════════════════════════════════════

REFERRAL_REFERRERS_PAGE_SIZE = 10
REFERRAL_REFERRED_PAGE_SIZE = 5


@router.callback_query(F.data == "admin_referral_menu")
async def admin_referral_menu(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        referrers_count = await session.scalar(
            select(func.count(func.distinct(User.referred_by))).where(User.referred_by != None)
        ) or 0
        referrals_count = await session.scalar(
            select(func.count()).select_from(User).where(User.referred_by != None)
        ) or 0
        total_turnover = await session.scalar(
            select(func.coalesce(func.sum(ReferralPaymentLog.amount), 0.0))
        ) or 0.0

    status = "✅ Включена" if (config and config.referral_enabled) else "❌ Выключена"
    text = (
        f"<b>👫 Реферальная программа</b>\n\n"
        f"Статус: {status}\n"
        f"Рефереров: {referrers_count}\n"
        f"Рефералов: {referrals_count}\n"
        f"Общий оборот: {total_turnover:.2f} руб."
    )
    await callback.message.edit_text(text, reply_markup=kb.admin_referral_menu_keyboard(), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "admin_referral_settings")
async def admin_referral_settings(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)

    await callback.message.edit_text(
        "<b>⚙️ Настройки реферальной программы</b>",
        reply_markup=kb.admin_referral_settings_keyboard(config),
        parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(F.data == "admin_referral_toggle_enabled")
async def admin_referral_toggle_enabled(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_enabled = not config.referral_enabled
        await session.commit()
        config_fresh = await session.get(SubscriptionConfig, 1)

    await callback.message.edit_reply_markup(reply_markup=kb.admin_referral_settings_keyboard(config_fresh))
    status = "включена" if config_fresh.referral_enabled else "выключена"
    await callback.answer(f"Программа {status}")


@router.callback_query(F.data == "admin_referral_toggle_pay_bonus")
async def admin_referral_toggle_pay_bonus(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_pay_bonus_enabled = not config.referral_pay_bonus_enabled
        await session.commit()
        config_fresh = await session.get(SubscriptionConfig, 1)

    await callback.message.edit_reply_markup(reply_markup=kb.admin_referral_settings_keyboard(config_fresh))
    await callback.answer()


@router.callback_query(F.data == "admin_referral_toggle_pay_first_only")
async def admin_referral_toggle_pay_first_only(callback: CallbackQuery):
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_pay_bonus_first_only = not config.referral_pay_bonus_first_only
        await session.commit()
        config_fresh = await session.get(SubscriptionConfig, 1)

    await callback.message.edit_reply_markup(reply_markup=kb.admin_referral_settings_keyboard(config_fresh))
    await callback.answer()


@router.callback_query(F.data == "admin_referral_set_bonus_referrer")
async def admin_referral_set_bonus_referrer(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_referral_bonus_referrer)
    await callback.message.answer("Введите количество бонусных дней рефереру (за каждого приведённого):")
    await callback.answer()


@router.message(AdminStates.set_referral_bonus_referrer)
async def save_referral_bonus_referrer(message: Message, state: FSMContext):
    try:
        days = int(message.text.strip())
        if days < 0:
            raise ValueError
    except ValueError:
        await message.answer("Введите целое неотрицательное число.")
        return

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_bonus_days_referrer = days
        await session.commit()

    await state.clear()
    await message.answer(f"✅ Бонус рефереру установлен: {days} дн.")


@router.callback_query(F.data == "admin_referral_set_bonus_referral")
async def admin_referral_set_bonus_referral(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_referral_bonus_referral)
    await callback.message.answer("Введите количество бонусных дней рефералу (новому пользователю):")
    await callback.answer()


@router.message(AdminStates.set_referral_bonus_referral)
async def save_referral_bonus_referral(message: Message, state: FSMContext):
    try:
        days = int(message.text.strip())
        if days < 0:
            raise ValueError
    except ValueError:
        await message.answer("Введите целое неотрицательное число.")
        return

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_bonus_days_referral = days
        await session.commit()

    await state.clear()
    await message.answer(f"✅ Бонус рефералу установлен: {days} дн.")


@router.callback_query(F.data == "admin_referral_set_pay_days")
async def admin_referral_set_pay_days(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_referral_pay_days)
    await callback.message.answer("Введите количество бонусных дней рефереру за оплату реферала:")
    await callback.answer()


@router.message(AdminStates.set_referral_pay_days)
async def save_referral_pay_days(message: Message, state: FSMContext):
    try:
        days = int(message.text.strip())
        if days < 0:
            raise ValueError
    except ValueError:
        await message.answer("Введите целое неотрицательное число.")
        return

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_pay_bonus_days = days
        await session.commit()

    await state.clear()
    await message.answer(f"✅ Дней за оплату реферала: {days}.")


@router.callback_query(F.data == "admin_referral_set_btn_name")
async def admin_referral_set_btn_name(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_referral_btn_name)
    await callback.message.answer(
        "Введите название кнопки реферальной программы в главном меню:",
        reply_markup=kb.admin_referral_input_cancel_keyboard()
    )
    await callback.answer()


@router.message(AdminStates.set_referral_btn_name)
async def save_referral_btn_name(message: Message, state: FSMContext):
    name = message.text.strip()
    if not name:
        await message.answer("Название не может быть пустым.")
        return

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_btn_name = name
        await session.commit()

    await state.clear()
    await message.answer(f"✅ Название кнопки меню: «{name}»")
    async with async_session_maker() as session:
        cfg = await session.get(SubscriptionConfig, 1)
    await message.answer(
        "<b>⚙️ Настройки реферальной программы</b>",
        reply_markup=kb.admin_referral_settings_keyboard(cfg),
        parse_mode="HTML"
    )


@router.callback_query(F.data == "admin_referral_set_sub_btn_name")
async def admin_referral_set_sub_btn_name(callback: CallbackQuery, state: FSMContext):
    await state.set_state(AdminStates.set_referral_sub_btn_name)
    await callback.message.answer(
        "Введите название кнопки реферальной программы в меню подписки:",
        reply_markup=kb.admin_referral_input_cancel_keyboard()
    )
    await callback.answer()


@router.message(AdminStates.set_referral_sub_btn_name)
async def save_referral_sub_btn_name(message: Message, state: FSMContext):
    name = message.text.strip()
    if not name:
        await message.answer("Название не может быть пустым.")
        return

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.referral_sub_btn_name = name
        await session.commit()

    await state.clear()
    await message.answer(f"✅ Название кнопки подписки: «{name}»")
    async with async_session_maker() as session:
        cfg = await session.get(SubscriptionConfig, 1)
    await message.answer(
        "<b>⚙️ Настройки реферальной программы</b>",
        reply_markup=kb.admin_referral_settings_keyboard(cfg),
        parse_mode="HTML"
    )


@router.callback_query(F.data == "admin_referral_cancel_input")
async def admin_referral_cancel_input(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    async with async_session_maker() as session:
        cfg = await session.get(SubscriptionConfig, 1)
    await callback.message.edit_text(
        "<b>⚙️ Настройки реферальной программы</b>",
        reply_markup=kb.admin_referral_settings_keyboard(cfg),
        parse_mode="HTML"
    )
    await callback.answer("Ввод отменён.")


@router.callback_query(F.data.startswith("admin_referral_referrers_"))
async def admin_referral_referrers(callback: CallbackQuery):
    try:
        page = int(callback.data.split("_")[-1])
    except (ValueError, IndexError):
        page = 0

    async with async_session_maker() as session:
        # Fetch referrers with stats
        referrer_ids_result = await session.execute(
            select(User.referred_by, func.count(User.id).label('cnt'))
            .where(User.referred_by != None)
            .group_by(User.referred_by)
            .order_by(desc('cnt'))
        )
        referrer_rows = referrer_ids_result.all()

    total = len(referrer_rows)
    total_pages = max(1, math.ceil(total / REFERRAL_REFERRERS_PAGE_SIZE))
    page = max(0, min(page, total_pages - 1))
    page_rows = referrer_rows[page * REFERRAL_REFERRERS_PAGE_SIZE:(page + 1) * REFERRAL_REFERRERS_PAGE_SIZE]

    referrers = []
    async with async_session_maker() as session:
        for row in page_rows:
            ref_id = row.referred_by
            count = row.cnt
            referrer = await session.get(User, ref_id)
            if not referrer:
                continue
            name = referrer.first_name or str(ref_id)
            if referrer.username:
                name += f" (@{referrer.username})"

            turnover = await session.scalar(
                select(func.coalesce(func.sum(ReferralPaymentLog.amount), 0.0))
                .where(ReferralPaymentLog.referrer_id == ref_id)
            ) or 0.0

            referrers.append({'id': ref_id, 'name': name, 'count': count, 'total': turnover})

    text = f"<b>👫 Рефереры · стр. {page + 1}/{total_pages}</b>\n\nВсего: {total}"
    markup = kb.admin_referral_referrers_keyboard(referrers, page, total_pages)
    try:
        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
    except Exception:
        await callback.message.answer(text, reply_markup=markup, parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("admin_referral_referrer_"))
async def admin_referral_referrer_detail(callback: CallbackQuery):
    # callback_data format: admin_referral_referrer_{uid}_{page}
    parts = callback.data.split("_")
    try:
        referrer_id = int(parts[3])
        page = int(parts[4])
    except (IndexError, ValueError):
        await callback.answer("Ошибка параметров.")
        return

    async with async_session_maker() as session:
        referrer = await session.get(User, referrer_id)
        if not referrer:
            await callback.answer("Реферер не найден.")
            return

        referred_result = await session.execute(
            select(User).where(User.referred_by == referrer_id).order_by(User.created_at.asc())
        )
        referred_users = referred_result.scalars().all()

        total_turnover = await session.scalar(
            select(func.coalesce(func.sum(ReferralPaymentLog.amount), 0.0))
            .where(ReferralPaymentLog.referrer_id == referrer_id)
        ) or 0.0

        # Collect per-referred turnover
        per_user_turnover = {}
        for ru in referred_users:
            amt = await session.scalar(
                select(func.coalesce(func.sum(ReferralPaymentLog.amount), 0.0))
                .where(ReferralPaymentLog.referrer_id == referrer_id,
                       ReferralPaymentLog.referred_user_id == ru.id)
            ) or 0.0
            per_user_turnover[ru.id] = amt

    name_r = referrer.first_name or str(referrer_id)
    if referrer.username:
        name_r += f" (@{referrer.username})"

    total_referred = len(referred_users)
    total_pages = max(1, math.ceil(total_referred / REFERRAL_REFERRED_PAGE_SIZE))
    page = max(0, min(page, total_pages - 1))
    page_slice = referred_users[page * REFERRAL_REFERRED_PAGE_SIZE:(page + 1) * REFERRAL_REFERRED_PAGE_SIZE]

    lines = [
        f"<b>👤 Реферер: {html.escape(name_r)} [id={referrer_id}]</b>",
        f"Всего приглашено: {total_referred}",
        f"Общий оборот: {total_turnover:.2f} руб.",
        "",
        "<b>Рефералы:</b>",
    ]
    for ru in page_slice:
        ru_name = html.escape(ru.first_name or str(ru.id))
        if ru.username:
            ru_name += f" (@{html.escape(ru.username)})"
        reg_date = ru.created_at.strftime('%d.%m.%Y') if ru.created_at else "?"
        turnover_val = per_user_turnover.get(ru.id, 0.0)
        lines.append(f"━━━━━━━━━━━━━━━━")
        lines.append(f"👤 {ru_name} [id={ru.id}] — зарег. {reg_date}")
        lines.append(f"   💰 Оборот: {turnover_val:.2f} руб.")

    text = "\n".join(lines)
    markup = kb.admin_referral_referrer_detail_keyboard(referrer_id, page, total_pages)
    try:
        await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")
    except Exception:
        await callback.message.answer(text, reply_markup=markup, parse_mode="HTML")
    await callback.answer()
