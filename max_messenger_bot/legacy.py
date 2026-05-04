from __future__ import annotations

from .settings import apply_legacy_env_defaults

apply_legacy_env_defaults()

from database import (  # noqa: E402
    AIConfig,
    Base,
    CaseStudy,
    Content,
    ContentMedia,
    KnowledgeBase,
    Mailing,
    MediaCollection,
    MediaLibrary,
    Message,
    PromoCode,
    RandomMessage,
    ReferralPaymentLog,
    ReferralTemplate,
    RobokassaPayment,
    SecretTestQuestion,
    SubscriptionConfig,
    SubscriptionPlan,
    TestConfig,
    TestQuestion,
    TestSession,
    Topic,
    TopicMediaDeck,
    TrialUsageHistory,
    User,
    UserSubscription,
    UserTopicState,
    YookassaPayment,
    async_session_maker,
    engine,
    get_all_admin_ids,
    init_db,
    media_collection_items,
    topic_collection_association,
)
