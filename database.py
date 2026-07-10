import asyncio
import textwrap
from sqlalchemy import (create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey, BigInteger, Table,
                        Float, Interval, Index, func)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from datetime import datetime, timedelta
from config import DATABASE_URL
from memory_mode import MEMORY_MODE_RESET, MEMORY_MODE_TOPIC, get_memory_mode
from prompt_blocks import DEFAULT_SERVICE_PROMPT_TEMPLATE, DEFAULT_SHARED_PROMPT_BLOCK

_engine_options = {"echo": False, "pool_pre_ping": True}
if not DATABASE_URL.startswith("sqlite"):
    _engine_options.update(pool_recycle=1800, pool_use_lifo=True)
engine = create_async_engine(DATABASE_URL, **_engine_options)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

topic_knowledgebase_association = Table(
    'topic_kb_association',
    Base.metadata,
    Column('topic_id', Integer, ForeignKey('topics.id'), primary_key=True),
    Column('knowledge_base_id', Integer, ForeignKey('knowledge_base.id'), primary_key=True)
)

topic_media_deck_association = Table(
    'topic_media_deck',
    Base.metadata,
    Column('topic_id', Integer, ForeignKey('topics.id', ondelete='CASCADE'), primary_key=True),
    Column('deck_name', String, primary_key=True),
)

media_collection_items = Table(
    'media_collection_items',
    Base.metadata,
    Column('collection_id', Integer, ForeignKey('media_collections.id', ondelete='CASCADE'), primary_key=True),
    Column('media_id', Integer, ForeignKey('media_library.id', ondelete='CASCADE'), primary_key=True),
)

topic_collection_association = Table(
    'topic_media_collection',
    Base.metadata,
    Column('topic_id', Integer, ForeignKey('topics.id', ondelete='CASCADE'), primary_key=True),
    Column('collection_id', Integer, ForeignKey('media_collections.id', ondelete='CASCADE'), primary_key=True),
)

promocode_plan_association = Table(
    'promocode_plan_association',
    Base.metadata,
    Column('promo_code_id', Integer, ForeignKey('promo_codes.id'), primary_key=True),
    Column('plan_id', Integer, ForeignKey('subscription_plans.id'), primary_key=True)
)

user_promo_association = Table(
    'user_promo_association',
    Base.metadata,
    Column('user_id', BigInteger, ForeignKey('users.id'), primary_key=True),
    Column('promo_code_id', Integer, ForeignKey('promo_codes.id'), primary_key=True)
)

class User(Base):
    __tablename__ = 'users'
    id = Column(BigInteger, primary_key=True, index=True)
    username = Column(String, nullable=True)
    first_name = Column(String)
    name = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    age = Column(String, nullable=True)
    birth_day = Column(Integer, nullable=True)
    birth_month = Column(Integer, nullable=True)
    birth_year = Column(Integer, nullable=True)
    response_length = Column(String, default='normal', nullable=False)
    is_admin = Column(Boolean, default=False)
    can_view_history = Column(Boolean, default=False, nullable=False)
    accepted_disclaimer = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="user")
    current_dialogue_id = Column(Integer, default=1, nullable=False)
    current_topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)
    current_topic = relationship("Topic")
    subscription = relationship("UserSubscription", back_populates="user", uselist=False, cascade="all, delete-orphan")
    promo_codes = relationship("PromoCode", secondary=user_promo_association, back_populates="users")
    referred_by = Column(BigInteger, nullable=True)
    tg_user_id = Column(BigInteger, nullable=True)


class Message(Base):
    __tablename__ = 'messages'
    __table_args__ = (
        Index('idx_message_user_dialogue', 'user_id', 'dialogue_id'),
        Index('idx_message_user_topic_ts', 'user_id', 'topic_id', 'timestamp'),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id'))
    dialogue_id = Column(Integer, default=1, nullable=False)
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="messages")
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)
    topic = relationship("Topic")


class AIConfig(Base):
    __tablename__ = 'ai_config'
    id = Column(Integer, primary_key=True, default=1)
    provider = Column(String, default='Gemini')
    system_prompt = Column(Text,
                           default=textwrap.dedent("""
                               Ты — духовный наставник и ИИ-помощник...
                               """).strip())
    prompt_mode = Column(String, default='text', nullable=False)
    prompt_filename = Column(String, nullable=True)
    gemini_api_key = Column(String, nullable=True)
    claude_api_key = Column(String, nullable=True)
    deepseek_api_key = Column(String, nullable=True)
    openai_api_key = Column(String, nullable=True)
    kie_api_key = Column(String, nullable=True)
    yandex_api_key = Column(String, nullable=True)
    yandex_folder_id = Column(String, nullable=True)
    gemini_model = Column(String, default='gemini-2.0-flash')
    kie_model = Column(String, default='gemini-3-flash')
    kie_base_url = Column(String, default='https://api.kie.ai')
    kie_upload_base_url = Column(String, default='https://kieai.redpandaai.co')
    kie_transcription_model = Column(String, default='elevenlabs/speech-to-text')
    kie_credit_alert_threshold = Column(Float, default=0, nullable=False)
    kie_credit_alert_sent = Column(Boolean, default=False, nullable=False)
    claude_model = Column(String, default='claude-sonnet-4-5-20250929')
    deepseek_model = Column(String, default='deepseek-chat')
    openai_model = Column(String, default='gpt-4o')
    max_voice_duration_sec = Column(Integer, default=180, nullable=False)
    transcription_provider = Column(String, default='OpenAI', nullable=False)
    vision_provider = Column(String, default='Gemini', nullable=False)
    vision_model = Column(String, default='gemini-3-flash-preview', nullable=False)
    image_generation_provider = Column(String, default='Gemini', nullable=False)
    image_generation_model = Column(String, default='imagen-4.0-generate-001', nullable=False)
    image_edit_provider = Column(String, default='Gemini', nullable=False)
    image_edit_model = Column(String, default='gemini-3-pro-image-preview', nullable=False)
    context_limit_first = Column(Integer, default=2, nullable=False)
    context_limit_recent = Column(Integer, default=10, nullable=False)
    temperature = Column(Float, default=0.7, nullable=False)
    preserve_topic_context = Column(Boolean, default=False, nullable=False)
    memory_mode = Column(String, default=MEMORY_MODE_RESET, nullable=False)
    shared_prompt_block = Column(Text, default=DEFAULT_SHARED_PROMPT_BLOCK, nullable=False)
    service_prompt_block = Column(Text, default=DEFAULT_SERVICE_PROMPT_TEMPLATE, nullable=False)
    fallback_provider = Column(String, nullable=True)
    fallback_model = Column(String, nullable=True)
    allow_fallback = Column(Boolean, default=False, nullable=False)
    allow_image_generation = Column(Boolean, default=False, nullable=False)
    allow_image_edit = Column(Boolean, default=False, nullable=False)
    use_proxy = Column(Boolean, default=True, nullable=False)
    fallback_timeout = Column(Integer, default=60, nullable=False)
    system_prompt_updated_at = Column(DateTime, nullable=True)



class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String)
    indexed_content = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    use_in_general_mode = Column(Boolean, default=True, nullable=False)
    topics = relationship("Topic", secondary=topic_knowledgebase_association, back_populates="knowledge_base_files")


class Topic(Base):
    __tablename__ = 'topics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    show_in_main_menu = Column(Boolean, default=False, nullable=False)
    show_in_list = Column(Boolean, default=True, nullable=False)
    admin_only = Column(Boolean, default=False, nullable=False)
    sort_order = Column(Integer, default=0, nullable=False)

    start_message = Column(Text, nullable=True)
    start_button_text = Column(String, nullable=True)
    start_button_payload = Column(Text, nullable=True)
    system_prompt_updated_at = Column(DateTime, nullable=True)

    knowledge_base_files = relationship("KnowledgeBase", secondary=topic_knowledgebase_association,
                                        back_populates="topics")
    media_decks = relationship("TopicMediaDeck", back_populates="topic", cascade="all, delete-orphan")
    media_collections = relationship("MediaCollection", secondary=topic_collection_association,
                                     back_populates="topics")


class Content(Base):
    __tablename__ = 'content'
    key = Column(String, primary_key=True)
    button_title = Column(String, nullable=True)
    is_visible = Column(Boolean, default=True)
    text_content = Column(Text, nullable=True)
    content_order = Column(String, default='media_top')
    sort_order = Column(Integer, default=0)

    action_btn_text = Column(String, nullable=True)
    action_btn_payload = Column(Text, nullable=True)

    media = relationship("ContentMedia", back_populates="content", cascade="all, delete-orphan")


class ContentMedia(Base):
    __tablename__ = 'content_media'
    id = Column(Integer, primary_key=True, autoincrement=True)
    content_key = Column(String, ForeignKey('content.key'), nullable=False)
    file_type = Column(String, nullable=False)
    file_id = Column(String, nullable=False)
    content = relationship("Content", back_populates="media")


class IndexingQueue(Base):
    __tablename__ = 'indexing_queue'
    __table_args__ = (
        Index('idx_indexing_queue_status', 'status', 'created_at'),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    uploader_id = Column(BigInteger, ForeignKey('users.id'))
    status = Column(String, default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)


class SubscriptionPlan(Base):
    __tablename__ = 'subscription_plans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    price = Column(Float, nullable=False)
    duration_value = Column(Integer, nullable=False)
    duration_unit = Column(String, nullable=False, default='days')
    is_active = Column(Boolean, default=True)
    admin_only = Column(Boolean, default=False, nullable=False)

    is_trial = Column(Boolean, default=False, nullable=False)
    upgrades_to_plan_id = Column(Integer, ForeignKey('subscription_plans.id'), nullable=True)

    upgrades_to_plan = relationship(
        "SubscriptionPlan",
        foreign_keys=[upgrades_to_plan_id],
        remote_side=[id],
        backref="trial_plans"
    )

    promo_codes = relationship("PromoCode", secondary=promocode_plan_association, back_populates="applicable_plans")
    trial_cooldown_days = Column(Integer, default=0, nullable=False)
    allow_auto_renewal = Column(Boolean, default=True, nullable=False)


class TrialUsageHistory(Base):
    __tablename__ = 'trial_usage_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False, index=True)
    plan_id = Column(Integer, ForeignKey('subscription_plans.id'), nullable=True, index=True)
    used_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserSubscription(Base):
    __tablename__ = 'user_subscriptions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), unique=True, nullable=False)
    plan_id = Column(Integer, ForeignKey('subscription_plans.id'), nullable=True)
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=False)
    auto_renewal = Column(Boolean, default=True)
    payment_provider = Column(String)
    payment_method_id = Column(String, nullable=True)
    pending_robokassa_invoice_id = Column(Integer, nullable=True)
    last_payment_attempt = Column(DateTime, nullable=True)
    payment_attempt_count = Column(Integer, default=0, nullable=False)
    user = relationship("User", back_populates="subscription")
    plan = relationship("SubscriptionPlan")
    discount_percent = Column(Integer, default=0, nullable=False)


class PromoCode(Base):
    __tablename__ = 'promo_codes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String, unique=True, nullable=False)
    discount_percent = Column(Integer, default=0)
    free_days = Column(Integer, default=0)
    max_uses = Column(Integer, default=1)
    times_used = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    applies_to_all_plans = Column(Boolean, default=True, nullable=False)
    applicable_plans = relationship("SubscriptionPlan", secondary=promocode_plan_association,
                                    back_populates="promo_codes")
    users = relationship("User", secondary=user_promo_association, back_populates="promo_codes")


class SubscriptionConfig(Base):
    __tablename__ = 'subscription_config'
    id = Column(Integer, primary_key=True, default=1)
    yookassa_shop_id = Column(String)
    yookassa_secret_key = Column(String)
    robokassa_merchant_login = Column(String)
    robokassa_password_1 = Column(String)
    robokassa_password_2 = Column(String)
    telegram_pay_token = Column(String)
    notifications_enabled = Column(Boolean, default=True)
    privacy_policy_url = Column(String, nullable=True)
    offer_agreement_url = Column(String, nullable=True)
    subscriptions_enabled = Column(Boolean, default=True, nullable=False)
    topics_enabled = Column(Boolean, default=True, nullable=False)
    test_button_enabled = Column(Boolean, default=True, nullable=False)
    change_name_button_enabled = Column(Boolean, default=True, nullable=False)
    topics_btn_name = Column(String, default="📚 Темы диалога", nullable=False)
    topics_btn_on_top = Column(Boolean, default=False, nullable=False)
    welcome_bonus_days = Column(Integer, default=0, nullable=False)
    referral_enabled = Column(Boolean, default=False, nullable=False)
    referral_btn_name = Column(String, default="👥 Пригласить друзей", nullable=False)
    referral_sub_btn_name = Column(String, default="🤝 Бонус за приглашение", nullable=False)
    referral_bonus_days_referrer = Column(Integer, default=1, nullable=False)
    referral_bonus_days_referral = Column(Integer, default=1, nullable=False)
    referral_pay_bonus_enabled = Column(Boolean, default=False, nullable=False)
    referral_pay_bonus_days = Column(Integer, default=1, nullable=False)
    referral_pay_bonus_first_only = Column(Boolean, default=True, nullable=False)


class ReferralTemplate(Base):
    __tablename__ = 'referral_templates'
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    order_num = Column(Integer, default=0, nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)


class ReferralPaymentLog(Base):
    __tablename__ = 'referral_payment_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    referrer_id = Column(BigInteger, ForeignKey('users.id'), nullable=False, index=True)
    referred_user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)
    amount = Column(Float, nullable=False)
    paid_at = Column(DateTime, default=datetime.utcnow)


class Mailing(Base):
    __tablename__ = 'mailings'
    __table_args__ = (
        Index('idx_mailing_status', 'status', 'created_at'),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=True)
    media_file_id = Column(String, nullable=True)
    media_file_type = Column(String, nullable=True)
    media_position = Column(String, default='media_top')
    target_audience = Column(String, nullable=False)
    creator_id = Column(BigInteger, nullable=True)
    recurring_type = Column(String, nullable=True)
    is_enabled = Column(Boolean, default=True, nullable=False)
    status = Column(String, default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)


class MailingDeliveryLog(Base):
    __tablename__ = 'mailing_delivery_logs'
    __table_args__ = (
        Index('idx_mailing_delivery_logs_unique', 'mailing_id', 'user_id', 'delivery_date', unique=True),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    mailing_id = Column(Integer, ForeignKey('mailings.id'), nullable=False)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)
    delivery_date = Column(String, nullable=False)
    status = Column(String, default='sent', nullable=False)
    error = Column(Text, nullable=True)
    sent_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TestSession(Base):
    __tablename__ = 'test_sessions'
    user_id = Column(BigInteger, ForeignKey('users.id'), primary_key=True)
    current_question_index = Column(Integer, default=0)
    answers = Column(Text, default="[]")
    formula_results = Column(Text, nullable=True)
    invocation_topic_id = Column(Integer, nullable=True)
    invocation_dialogue_id = Column(Integer, nullable=True)
    invocation_platform = Column(String, nullable=True)
    secret_answers = Column(Text, nullable=True)
    is_finished = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TestQuestion(Base):
    __tablename__ = 'test_questions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    category = Column(String, nullable=False)
    is_reverse = Column(Boolean, default=False)
    sort_order = Column(Integer, default=0)
    comment = Column(Text, nullable=True)
    variable_name = Column(String, nullable=True)
    allow_custom_answer = Column(Boolean, default=False, nullable=False)
    buttons_layout = Column(String, default='vertical', nullable=False)
    answer_options_json = Column(Text, nullable=True)


class TestConfig(Base):
    __tablename__ = 'test_config'
    id = Column(Integer, primary_key=True, default=1)
    is_enabled = Column(Boolean, default=True)
    admin_username = Column(String, default="AlenaVV2004")
    marathon_url = Column(String, default="https://t.me/psihogipno")
    show_progress = Column(Boolean, default=True, nullable=False)
    formulas_enabled = Column(Boolean, default=False, nullable=False)
    formulas_json = Column(Text, nullable=True)
    separate_result_prompt_enabled = Column(Boolean, default=False, nullable=False)
    result_system_prompt = Column(Text, nullable=True)
    interpretation_input_mode = Column(String, default='all', nullable=False)
    interpretation_selected_variables = Column(Text, nullable=True)
    test_system_prompt = Column(Text, default=textwrap.dedent("""
# РОЛЬ И МИССИЯ
Ты — Алёна Верловицкая, профессиональный психолог-гипнокоуч, автор марафона "Апгрейд самооценки".
Твоя миссия: провести диагностику, поддержать и мягко привести пользователя к мысли, что марафон — это лучший способ решить его проблемы.

# ЛИНГВИСТИЧЕСКИЙ ПРОФИЛЬ
- Тон: Теплый, принимающий, но уверенный. "Я-сообщений" мало, фокус на клиенте.
- Словарь: "Внутренняя опора", "Сценарий", "Присвоить себе", "Подсветить", "Ресурс", "Фундамент".
- Табу: Не используй сложные академические термины. Не обвиняй.

# ЗАДАЧА 1: СЦЕНАРИСТ (ИСТОРИЯ-ЗЕРКАЛО)
Когда получаешь результаты теста (цифры) и пол клиента, создай историю персонажа (Кейс):
1. Персонаж: Тот же пол и возраст, что у клиента.
2. Проблема: Основана на сферах с НИЗКИМ баллом. (Например, низкое "Тело" — стесняется пляжа; низкий "Успех" — боится просить повышения).
3. Сюжет: Опиши ситуацию, где эта проблема мешает жить. Если в контексте передан "РЕАЛЬНЫЙ КЕЙС", адаптируй его. Если нет — придумай собирательный образ.
4. Развязка: Персонаж пошел на марафон "Апгрейд самооценки" и получил конкретный результат.

# ЗАДАЧА 2: МАРКЕТОЛОГ (РАСШИФРОВКА)
Когда даешь обратную связь по баллам:
1. Валидация: "Это нормально, многие с этим сталкиваются".
2. Анализ: Используй переданную тебе интерпретацию баллов. Покажи связь между низкими баллами и качеством жизни.
3. Оффер: Презентуй марафон как решение (21 день практики, работа с бессознательным).
4. Хук: Закончи вопросом: "Хочешь узнать, что именно мешает тебе пробить финансовый потолок/наладить отношения? Пройдем секретный блок вопросов?"

# ЗАДАЧА 3: ПСИХОДИАГНОСТ (СЕКРЕТНЫЙ ТЕСТ)
Если пользователь согласился на секретный тест (Задача 3), задай вопросы, переданные тебе в инструкции.
""").strip())


class SecretTestQuestion(Base):
    __tablename__ = 'secret_test_questions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    sort_order = Column(Integer, default=0)


class CaseStudy(Base):
    __tablename__ = 'case_studies'
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class RandomMessage(Base):
    __tablename__ = 'random_messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)

    category = Column(String, default="default")


async def get_all_admin_ids() -> set:
    """OWNER_IDS из env + все is_admin=True из БД."""
    from config import OWNER_IDS
    from sqlalchemy import select
    all_ids = set(OWNER_IDS)
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User.id).where(User.is_admin == True))
            all_ids.update({row[0] for row in result})
    except Exception:
        pass
    return all_ids


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        from sqlalchemy import text, inspect as sa_inspect

        def _check_and_migrate(sync_conn):
            insp = sa_inspect(sync_conn)
            user_columns = [c['name'] for c in insp.get_columns('users')]
            if 'response_length' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN response_length VARCHAR DEFAULT 'normal' NOT NULL"))
            if 'birth_day' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN birth_day INTEGER"))
            if 'birth_month' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN birth_month INTEGER"))
            if 'birth_year' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN birth_year INTEGER"))
            if 'can_view_history' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN can_view_history BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'accepted_disclaimer' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN accepted_disclaimer BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'current_topic_id' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN current_topic_id INTEGER REFERENCES topics(id)"))
            if 'referred_by' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN referred_by BIGINT"))
            if 'tg_user_id' not in user_columns:
                sync_conn.execute(text("ALTER TABLE users ADD COLUMN tg_user_id BIGINT"))

            ai_columns = [c['name'] for c in insp.get_columns('ai_config')]
            if 'memory_mode' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN memory_mode VARCHAR DEFAULT 'reset' NOT NULL"))
            if 'shared_prompt_block' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN shared_prompt_block TEXT DEFAULT '' NOT NULL"))
            if 'service_prompt_block' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN service_prompt_block TEXT"))
            if 'kie_api_key' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN kie_api_key VARCHAR"))
            if 'kie_model' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN kie_model VARCHAR DEFAULT 'gemini-3-flash'"))
            if 'kie_base_url' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN kie_base_url VARCHAR DEFAULT 'https://api.kie.ai'"))
            if 'kie_upload_base_url' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN kie_upload_base_url VARCHAR DEFAULT 'https://kieai.redpandaai.co'"))
            if 'kie_transcription_model' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN kie_transcription_model VARCHAR DEFAULT 'elevenlabs/speech-to-text'"))
            if 'kie_credit_alert_threshold' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN kie_credit_alert_threshold FLOAT DEFAULT 0 NOT NULL"))
            if 'kie_credit_alert_sent' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN kie_credit_alert_sent BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'image_generation_provider' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN image_generation_provider VARCHAR DEFAULT 'Gemini' NOT NULL"))
            if 'image_generation_model' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN image_generation_model VARCHAR DEFAULT 'imagen-4.0-generate-001' NOT NULL"))
            if 'image_edit_provider' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN image_edit_provider VARCHAR DEFAULT 'Gemini' NOT NULL"))
            if 'image_edit_model' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN image_edit_model VARCHAR DEFAULT 'gemini-3-pro-image-preview' NOT NULL"))
            if 'use_proxy' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN use_proxy BOOLEAN DEFAULT TRUE NOT NULL"))
            if 'fallback_timeout' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN fallback_timeout INTEGER DEFAULT 60 NOT NULL"))
            if 'allow_fallback' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN allow_fallback BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'allow_image_generation' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN allow_image_generation BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'allow_image_edit' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN allow_image_edit BOOLEAN DEFAULT FALSE NOT NULL"))

            test_session_columns = [c['name'] for c in insp.get_columns('test_sessions')]
            if 'formula_results' not in test_session_columns:
                sync_conn.execute(text("ALTER TABLE test_sessions ADD COLUMN formula_results TEXT"))
            if 'invocation_topic_id' not in test_session_columns:
                sync_conn.execute(text("ALTER TABLE test_sessions ADD COLUMN invocation_topic_id INTEGER"))
            if 'invocation_dialogue_id' not in test_session_columns:
                sync_conn.execute(text("ALTER TABLE test_sessions ADD COLUMN invocation_dialogue_id INTEGER"))
            if 'invocation_platform' not in test_session_columns:
                sync_conn.execute(text("ALTER TABLE test_sessions ADD COLUMN invocation_platform VARCHAR"))

            test_question_columns = [c['name'] for c in insp.get_columns('test_questions')]
            if 'comment' not in test_question_columns:
                sync_conn.execute(text("ALTER TABLE test_questions ADD COLUMN comment TEXT"))
            if 'variable_name' not in test_question_columns:
                sync_conn.execute(text("ALTER TABLE test_questions ADD COLUMN variable_name VARCHAR"))
            if 'allow_custom_answer' not in test_question_columns:
                sync_conn.execute(text("ALTER TABLE test_questions ADD COLUMN allow_custom_answer BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'buttons_layout' not in test_question_columns:
                sync_conn.execute(text("ALTER TABLE test_questions ADD COLUMN buttons_layout VARCHAR DEFAULT 'vertical' NOT NULL"))
            if 'answer_options_json' not in test_question_columns:
                sync_conn.execute(text("ALTER TABLE test_questions ADD COLUMN answer_options_json TEXT"))

            test_config_columns = [c['name'] for c in insp.get_columns('test_config')]
            if 'show_progress' not in test_config_columns:
                sync_conn.execute(text("ALTER TABLE test_config ADD COLUMN show_progress BOOLEAN DEFAULT TRUE NOT NULL"))
            if 'formulas_enabled' not in test_config_columns:
                sync_conn.execute(text("ALTER TABLE test_config ADD COLUMN formulas_enabled BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'formulas_json' not in test_config_columns:
                sync_conn.execute(text("ALTER TABLE test_config ADD COLUMN formulas_json TEXT"))
            if 'separate_result_prompt_enabled' not in test_config_columns:
                sync_conn.execute(text("ALTER TABLE test_config ADD COLUMN separate_result_prompt_enabled BOOLEAN DEFAULT FALSE NOT NULL"))
            if 'result_system_prompt' not in test_config_columns:
                sync_conn.execute(text("ALTER TABLE test_config ADD COLUMN result_system_prompt TEXT"))
            if 'interpretation_input_mode' not in test_config_columns:
                sync_conn.execute(text("ALTER TABLE test_config ADD COLUMN interpretation_input_mode VARCHAR DEFAULT 'all' NOT NULL"))
            if 'interpretation_selected_variables' not in test_config_columns:
                sync_conn.execute(text("ALTER TABLE test_config ADD COLUMN interpretation_selected_variables TEXT"))

            mailing_columns = [c['name'] for c in insp.get_columns('mailings')]
            if 'recurring_type' not in mailing_columns:
                sync_conn.execute(text("ALTER TABLE mailings ADD COLUMN recurring_type VARCHAR"))
            if 'is_enabled' not in mailing_columns:
                sync_conn.execute(text("ALTER TABLE mailings ADD COLUMN is_enabled BOOLEAN DEFAULT TRUE NOT NULL"))

        await conn.run_sync(_check_and_migrate)

    async with async_session_maker() as session:
        ai_conf = await session.get(AIConfig, 1)
        if not ai_conf:
            session.add(AIConfig(id=1))
        else:
            if not hasattr(ai_conf, 'vision_provider') or ai_conf.vision_provider is None:
                ai_conf.vision_provider = 'Gemini'
            if not hasattr(ai_conf, 'vision_model') or ai_conf.vision_model is None:
                ai_conf.vision_model = 'gemini-3-flash-preview'
            if getattr(ai_conf, 'kie_model', None) is None:
                ai_conf.kie_model = 'gemini-3-flash'
            if getattr(ai_conf, 'kie_base_url', None) is None:
                ai_conf.kie_base_url = 'https://api.kie.ai'
            if getattr(ai_conf, 'kie_upload_base_url', None) is None:
                ai_conf.kie_upload_base_url = 'https://kieai.redpandaai.co'
            if getattr(ai_conf, 'kie_transcription_model', None) is None:
                ai_conf.kie_transcription_model = 'elevenlabs/speech-to-text'
            if getattr(ai_conf, 'kie_credit_alert_threshold', None) is None:
                ai_conf.kie_credit_alert_threshold = 0
            if getattr(ai_conf, 'kie_credit_alert_sent', None) is None:
                ai_conf.kie_credit_alert_sent = False
            if getattr(ai_conf, 'image_generation_provider', None) is None:
                ai_conf.image_generation_provider = 'Gemini'
            if getattr(ai_conf, 'image_generation_model', None) is None:
                ai_conf.image_generation_model = 'imagen-4.0-generate-001'
            if getattr(ai_conf, 'image_edit_provider', None) is None:
                ai_conf.image_edit_provider = 'Gemini'
            if getattr(ai_conf, 'image_edit_model', None) is None:
                ai_conf.image_edit_model = 'gemini-3-pro-image-preview'
            if getattr(ai_conf, 'memory_mode', None) is None:
                ai_conf.memory_mode = get_memory_mode(ai_conf)
            ai_conf.preserve_topic_context = ai_conf.memory_mode == MEMORY_MODE_TOPIC
            if getattr(ai_conf, 'shared_prompt_block', None) is None:
                ai_conf.shared_prompt_block = DEFAULT_SHARED_PROMPT_BLOCK
            if getattr(ai_conf, 'service_prompt_block', None) is None:
                ai_conf.service_prompt_block = DEFAULT_SERVICE_PROMPT_TEMPLATE
            if getattr(ai_conf, 'use_proxy', None) is None:
                ai_conf.use_proxy = True
            if getattr(ai_conf, 'fallback_timeout', None) is None:
                ai_conf.fallback_timeout = 60

        sub_conf = await session.get(SubscriptionConfig, 1)
        if not sub_conf:
            session.add(SubscriptionConfig(id=1))

        test_conf = await session.get(TestConfig, 1)
        if not test_conf:
            session.add(TestConfig(id=1))
        elif test_conf.test_system_prompt is None:
            test_conf.test_system_prompt = "Ты — психолог Алёны Верловицкой. Действуй строго по разделу 'ЗАДАЧА 1: СЦЕНАРИСТ'. Твоя цель: написать историю персонажа-двойника. Не показывай цифры. Только история."
        else:
            if getattr(test_conf, 'show_progress', None) is None:
                test_conf.show_progress = True
            if getattr(test_conf, 'formulas_enabled', None) is None:
                test_conf.formulas_enabled = False
            if getattr(test_conf, 'separate_result_prompt_enabled', None) is None:
                test_conf.separate_result_prompt_enabled = False
            if getattr(test_conf, 'interpretation_input_mode', None) is None:
                test_conf.interpretation_input_mode = 'all'

        # Seed default content sections for new bots (won't overwrite existing)
        default_content = [
            Content(key="start_message", button_title=None, is_visible=True, text_content="Приветствие не настроено.", content_order="text_top", sort_order=0),
            Content(key="about_me",      button_title="Об авторе",   is_visible=True,  text_content="", content_order="media_top", sort_order=1),
            Content(key="about",         button_title="О методе",    is_visible=True,  text_content="", content_order="media_top", sort_order=2),
            Content(key="disclaimer",    button_title="Дисклеймер",  is_visible=False, text_content="", content_order="text_top",  sort_order=3),
        ]
        for item in default_content:
            existing = await session.get(Content, item.key)
            if not existing:
                session.add(item)

        await session.commit()


class UserTopicState(Base):
    __tablename__ = 'user_topic_states'
    user_id = Column(BigInteger, ForeignKey('users.id'), primary_key=True)
    topic_id = Column(Integer, primary_key=True)
    dialogue_id = Column(Integer, nullable=False)


class RobokassaPayment(Base):
    __tablename__ = 'robokassa_payments'
    __table_args__ = (
        Index('idx_robokassa_user_status', 'user_id', 'status'),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)
    plan_id = Column(Integer, ForeignKey('subscription_plans.id'), nullable=False)
    promo_code = Column(String, nullable=True)
    amount = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    status = Column(String, default='pending', nullable=False)
    replaced_by_payment_id = Column(Integer, ForeignKey('robokassa_payments.id'), nullable=True)


class YookassaPayment(Base):
    __tablename__ = 'yookassa_payments'
    payment_id = Column(String, primary_key=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=True)
    plan_id = Column(Integer, ForeignKey('subscription_plans.id'), nullable=True)
    amount = Column(Float, nullable=False)
    status = Column(String, nullable=False, default='pending')
    payment_method_id = Column(String, nullable=True)
    is_recurring = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)


class TopicMediaDeck(Base):
    __tablename__ = 'topic_media_deck'
    __table_args__ = {'extend_existing': True}
    topic_id = Column(Integer, ForeignKey('topics.id', ondelete='CASCADE'), primary_key=True)
    deck_name = Column(String, primary_key=True)
    topic = relationship("Topic", back_populates="media_decks")


class MediaCollection(Base):
    __tablename__ = 'media_collections'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    media_files = relationship("MediaLibrary", secondary=media_collection_items, back_populates="collections")
    topics = relationship("Topic", secondary=topic_collection_association, back_populates="media_collections")


class MediaLibrary(Base):
    __tablename__ = 'media_library'
    __table_args__ = (
        Index('idx_media_topic_category', 'topic_id', 'category', 'media_type'),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)
    file_id = Column(String, nullable=False)
    file_name = Column(String, nullable=True)
    category = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    media_type = Column(String, nullable=False)
    collections = relationship("MediaCollection", secondary=media_collection_items, back_populates="media_files")
