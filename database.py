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

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

topic_knowledgebase_association = Table(
    'topic_kb_association',
    Base.metadata,
    Column('topic_id', Integer, ForeignKey('topics.id'), primary_key=True),
    Column('knowledge_base_id', Integer, ForeignKey('knowledge_base.id'), primary_key=True)
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
    yandex_api_key = Column(String, nullable=True)
    yandex_folder_id = Column(String, nullable=True)
    gemini_model = Column(String, default='gemini-2.0-flash')
    claude_model = Column(String, default='claude-sonnet-4-5-20250929')
    deepseek_model = Column(String, default='deepseek-chat')
    openai_model = Column(String, default='gpt-4o')
    max_voice_duration_sec = Column(Integer, default=180, nullable=False)
    transcription_provider = Column(String, default='OpenAI', nullable=False)
    vision_provider = Column(String, default='Gemini', nullable=False)
    vision_model = Column(String, default='gemini-3-flash-preview', nullable=False)
    context_limit_first = Column(Integer, default=2, nullable=False)
    context_limit_recent = Column(Integer, default=10, nullable=False)
    temperature = Column(Float, default=0.7, nullable=False)
    preserve_topic_context = Column(Boolean, default=False, nullable=False)
    memory_mode = Column(String, default=MEMORY_MODE_RESET, nullable=False)
    shared_prompt_block = Column(Text, default=DEFAULT_SHARED_PROMPT_BLOCK, nullable=False)
    service_prompt_block = Column(Text, default=DEFAULT_SERVICE_PROMPT_TEMPLATE, nullable=False)


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

    knowledge_base_files = relationship("KnowledgeBase", secondary=topic_knowledgebase_association,
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


class TestConfig(Base):
    __tablename__ = 'test_config'
    id = Column(Integer, primary_key=True, default=1)
    is_enabled = Column(Boolean, default=True)
    admin_username = Column(String, default="AlenaVV2004")
    marathon_url = Column(String, default="https://t.me/psihogipno")
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

            ai_columns = [c['name'] for c in insp.get_columns('ai_config')]
            if 'memory_mode' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN memory_mode VARCHAR DEFAULT 'reset' NOT NULL"))
            if 'shared_prompt_block' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN shared_prompt_block TEXT DEFAULT '' NOT NULL"))
            if 'service_prompt_block' not in ai_columns:
                sync_conn.execute(text("ALTER TABLE ai_config ADD COLUMN service_prompt_block TEXT"))

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
            if getattr(ai_conf, 'memory_mode', None) is None:
                ai_conf.memory_mode = get_memory_mode(ai_conf)
            ai_conf.preserve_topic_context = ai_conf.memory_mode == MEMORY_MODE_TOPIC
            if getattr(ai_conf, 'shared_prompt_block', None) is None:
                ai_conf.shared_prompt_block = DEFAULT_SHARED_PROMPT_BLOCK
            if getattr(ai_conf, 'service_prompt_block', None) is None:
                ai_conf.service_prompt_block = DEFAULT_SERVICE_PROMPT_TEMPLATE

        sub_conf = await session.get(SubscriptionConfig, 1)
        if not sub_conf:
            session.add(SubscriptionConfig(id=1))

        test_conf = await session.get(TestConfig, 1)
        if not test_conf:
            session.add(TestConfig(id=1))
        elif test_conf.test_system_prompt is None:
            test_conf.test_system_prompt = "Ты — психолог Алёны Верловицкой. Действуй строго по разделу 'ЗАДАЧА 1: СЦЕНАРИСТ'. Твоя цель: написать историю персонажа-двойника. Не показывай цифры. Только история."

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
