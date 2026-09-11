import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import sqlalchemy.ext.asyncio as sqlalchemy_asyncio

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

original_create_async_engine = sqlalchemy_asyncio.create_async_engine


def _sqlite_compatible_engine(*args, **kwargs):
    kwargs.pop("pool_recycle", None)
    kwargs.pop("pool_use_lifo", None)
    return original_create_async_engine(*args, **kwargs)


with patch.object(sqlalchemy_asyncio, "create_async_engine", _sqlite_compatible_engine):
    from effective_subscription import (
        ActiveSubscription,
        EffectiveSubscriptionFilters,
        choose_active_subscription,
        effective_subscription_filters,
        is_active_subscription,
        load_active_subscription,
    )
    import max_messenger_bot.services.subscription_access as sub_access_compat
    from handlers import _build_client_payment_info_text
    from database import (
        User,
        UserSubscription,
        SubscriptionPlan,
        SubscriptionConfig,
        PromoCode,
        RobokassaPayment,
        YookassaPayment,
        async_session_maker,
        init_db,
    )


class SubscriptionVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        self.now = datetime(2026, 9, 11, 12, 0, 0)

    def test_backward_compatibility_symbols(self):
        """Verify max_messenger_bot.services.subscription_access re-exports all 6 symbols."""
        self.assertIs(sub_access_compat.ActiveSubscription, ActiveSubscription)
        self.assertIs(sub_access_compat.EffectiveSubscriptionFilters, EffectiveSubscriptionFilters)
        self.assertIs(sub_access_compat.is_active_subscription, is_active_subscription)
        self.assertIs(sub_access_compat.choose_active_subscription, choose_active_subscription)
        self.assertIs(sub_access_compat.load_active_subscription, load_active_subscription)
        self.assertIs(sub_access_compat.effective_subscription_filters, effective_subscription_filters)

    def test_is_active_subscription_discount_does_not_grant_access(self):
        """discount_percent > 0 on expired subscription does not grant active access."""
        expired_sub = SimpleNamespace(
            end_date=self.now - timedelta(days=1),
            discount_percent=20,
        )
        self.assertFalse(is_active_subscription(expired_sub, self.now))

        active_sub = SimpleNamespace(
            end_date=self.now + timedelta(days=5),
            discount_percent=0,
        )
        self.assertTrue(is_active_subscription(active_sub, self.now))

    def test_linked_source_matrix(self):
        """Active Telegram wins; otherwise active MAX; otherwise None."""
        tg_active = SimpleNamespace(end_date=self.now + timedelta(days=5))
        tg_expired = SimpleNamespace(end_date=self.now - timedelta(days=5))
        max_active = SimpleNamespace(end_date=self.now + timedelta(days=10))
        max_expired = SimpleNamespace(end_date=self.now - timedelta(days=2))

        # 1. Both active -> Telegram wins
        res = choose_active_subscription(max_active, tg_active, self.now)
        self.assertIsNotNone(res)
        self.assertIs(res.subscription, tg_active)
        self.assertEqual(res.source, "telegram")

        # 2. TG expired, MAX active -> MAX wins
        res = choose_active_subscription(max_active, tg_expired, self.now)
        self.assertIsNotNone(res)
        self.assertIs(res.subscription, max_active)
        self.assertEqual(res.source, "max")

        # 3. Both expired -> None
        res = choose_active_subscription(max_expired, tg_expired, self.now)
        self.assertIsNone(res)

    async def test_denis_class_expired_with_discount_in_payment_info_text(self):
        """Denis-class regression: expired subscription + discount_percent > 0 must be Неактивен."""
        user_id = 99912345
        async with async_session_maker() as session:
            user = User(id=user_id, first_name="Denis", username="denis_test")
            session.add(user)
            plan = SubscriptionPlan(
                id=901,
                name="Месячный",
                price=1000.0,
                duration_value=1,
                duration_unit="months",
            )
            session.add(plan)
            sub = UserSubscription(
                user_id=user_id,
                plan_id=plan.id,
                start_date=self.now - timedelta(days=35),
                end_date=self.now - timedelta(days=5),  # expired 5 days ago
                discount_percent=30,  # has active discount
                auto_renewal=False,
                payment_provider="Yookassa",
            )
            session.add(sub)
            await session.commit()

        text = await _build_client_payment_info_text(user_id)
        # Status MUST be Неактивен
        self.assertIn("<b>Статус доступа:</b> ❌ Неактивен", text)
        self.assertNotIn("✅ Активен", text)
        # Discount must NOT be listed under 'Основание:'
        self.assertNotIn("Основание:", text)
        # No active paid plan
        self.assertIn("Активного платного тарифа нет", text)
        # But discount is preserved in promo section
        self.assertIn("Активная скидка сохранена в аккаунте (30%)", text)

    async def test_max_user_linked_tg_view_consistency(self):
        """MAX user with active linked TG subscription is shown as active in TG admin view."""
        max_user_id = 100_000_000_123
        tg_user_id = 888777
        async with async_session_maker() as session:
            tg_user = User(id=tg_user_id, first_name="TG_Alice", username="alice_tg")
            max_user = User(id=max_user_id, first_name="MAX_Alice", tg_user_id=tg_user_id)
            session.add_all([tg_user, max_user])

            plan = SubscriptionPlan(
                id=902,
                name="Премиум Год",
                price=5000.0,
                duration_value=1,
                duration_unit="months",
            )
            session.add(plan)
            # TG subscription is active
            tg_sub = UserSubscription(
                user_id=tg_user_id,
                plan_id=plan.id,
                start_date=self.now - timedelta(days=10),
                end_date=self.now + timedelta(days=20),
                auto_renewal=True,
                payment_provider="Yookassa",
            )
            # MAX user subscription is expired or missing
            max_sub = UserSubscription(
                user_id=max_user_id,
                plan_id=plan.id,
                start_date=self.now - timedelta(days=60),
                end_date=self.now - timedelta(days=30),
                auto_renewal=False,
                payment_provider="Robokassa",
            )
            session.add_all([tg_sub, max_sub])
            await session.commit()

        # Both resolver and TG admin view should agree
        async with async_session_maker() as session:
            active_sub = await load_active_subscription(session, max_user_id, self.now)
            self.assertIsNotNone(active_sub)
            self.assertEqual(active_sub.source, "telegram")
            self.assertEqual(active_sub.subscription.user_id, tg_user_id)

        text = await _build_client_payment_info_text(max_user_id)
        self.assertIn("<b>Статус доступа:</b> ✅ Активен", text)
        self.assertIn("Премиум Год", text)
        self.assertIn("Автопродление: ✅ Включено", text)

    async def test_mixed_source_pricing_cases_a_b_c(self):
        """Regression for linked TG/MAX mixed-source pricing:
        A: TG active 1000 RUB plan, discount 0; linked MAX expired sub, discount 30% -> price 1000.00 руб. (NOT 700.00).
        B: TG active 1000 RUB plan, discount 25%; linked MAX no discount -> price 750.00 руб.
        C: Direct active MAX subscription with 20% discount -> price 800.00 руб.
        """
        plan_id = 910
        async with async_session_maker() as session:
            plan = SubscriptionPlan(
                id=plan_id,
                name="Тариф 1000",
                price=1000.0,
                duration_value=1,
                duration_unit="months",
            )
            session.add(plan)
            await session.commit()

        # --- Case A ---
        max_user_a = 100_000_000_201
        tg_user_a = 777201
        async with async_session_maker() as session:
            session.add_all([
                User(id=tg_user_a, first_name="TG_A"),
                User(id=max_user_a, first_name="MAX_A", tg_user_id=tg_user_a),
            ])
            # TG active sub, discount 0
            session.add(UserSubscription(
                user_id=tg_user_a,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=10),
                end_date=self.now + timedelta(days=20),
                auto_renewal=True,
                payment_provider="Yookassa",
                discount_percent=0,
            ))
            # MAX expired sub, discount 30%
            session.add(UserSubscription(
                user_id=max_user_a,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=60),
                end_date=self.now - timedelta(days=30),
                auto_renewal=False,
                payment_provider="Robokassa",
                discount_percent=30,
            ))
            await session.commit()

        text_a = await _build_client_payment_info_text(max_user_a)
        self.assertIn("<b>Статус доступа:</b> ✅ Активен", text_a)
        self.assertIn("• Стоимость со скидкой: 1000.00 руб.", text_a)
        self.assertNotIn("700.00 руб.", text_a)
        # Verify MAX account data preserves its own discount in explicit promo section
        self.assertIn("Активная скидка сохранена в аккаунте (30%)", text_a)

        # --- Case B ---
        max_user_b = 100_000_000_202
        tg_user_b = 777202
        async with async_session_maker() as session:
            session.add_all([
                User(id=tg_user_b, first_name="TG_B"),
                User(id=max_user_b, first_name="MAX_B", tg_user_id=tg_user_b),
            ])
            # TG active sub, discount 25%
            session.add(UserSubscription(
                user_id=tg_user_b,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=10),
                end_date=self.now + timedelta(days=20),
                auto_renewal=True,
                payment_provider="Yookassa",
                discount_percent=25,
            ))
            # MAX user no discount
            session.add(UserSubscription(
                user_id=max_user_b,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=60),
                end_date=self.now - timedelta(days=30),
                auto_renewal=False,
                payment_provider="Robokassa",
                discount_percent=0,
            ))
            await session.commit()

        text_b = await _build_client_payment_info_text(max_user_b)
        self.assertIn("<b>Статус доступа:</b> ✅ Активен", text_b)
        self.assertIn("• Стоимость со скидкой: 750.00 руб.", text_b)
        self.assertNotIn("1000.00 руб.", text_b)

        # --- Case C: Direct active MAX subscription ---
        max_user_c = 100_000_000_203
        async with async_session_maker() as session:
            session.add(User(id=max_user_c, first_name="MAX_C", tg_user_id=None))
            session.add(UserSubscription(
                user_id=max_user_c,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=5),
                end_date=self.now + timedelta(days=25),
                auto_renewal=True,
                payment_provider="Robokassa",
                discount_percent=20,
            ))
            await session.commit()

        text_c = await _build_client_payment_info_text(max_user_c)
        self.assertIn("<b>Статус доступа:</b> ✅ Активен", text_c)
        self.assertIn("• Стоимость со скидкой: 800.00 руб.", text_c)
