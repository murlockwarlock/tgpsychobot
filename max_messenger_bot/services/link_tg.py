from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..legacy import User, UserSubscription, async_session_maker
from ..models import MAX_ID_OFFSET
from ..storage import StateStore
from ..time_utils import utc_now


async def show_link_tg_prompt(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    """Show prompt asking user to enter their TG user ID."""
    text = (
        "🔗 <b>Привязка Telegram аккаунта</b>\n\n"
        "Если у вас есть активная подписка в нашем Telegram боте, вы можете привязать ваш TG аккаунт "
        "и использовать подписку в MAX.\n\n"
        "📱 <b>Как узнать ваш Telegram ID?</b>\n"
        "1. Откройте Telegram\n"
        "2. Найдите бота @userinfobot\n"
        "3. Отправьте ему любое сообщение\n"
        "4. Он покажет ваш ID — введите его здесь\n\n"
        "Введите ваш Telegram ID (только цифры):"
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard([[callback_button("❌ Отмена", "sub_info")]]),
    )


async def process_tg_link(
    client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, tg_id_str: str
) -> None:
    """Process the TG ID entered by the user."""
    await states.clear(user_id)

    tg_id_str = tg_id_str.strip()
    retry_keyboard = inline_keyboard(
        [[callback_button("🔗 Попробовать снова", "link_tg_start"), callback_button("❌ Отмена", "sub_info")]]
    )
    if not tg_id_str.isdigit():
        await client.send_message(
            chat_id=chat_id,
            text="❌ Неверный формат. Введите только цифры (ваш Telegram ID).",
            attachments=retry_keyboard,
        )
        return

    tg_id = int(tg_id_str)
    if tg_id >= MAX_ID_OFFSET:
        await client.send_message(
            chat_id=chat_id,
            text="❌ Это не похоже на Telegram ID. Попробуйте ещё раз.",
            attachments=retry_keyboard,
        )
        return

    async with async_session_maker() as session:
        tg_user = await session.get(
            User,
            tg_id,
            options=[selectinload(User.subscription).selectinload(UserSubscription.plan)],
        )
        if not tg_user:
            await client.send_message(
                chat_id=chat_id,
                text=(
                    "❌ Telegram аккаунт с таким ID не найден в нашей базе.\n\n"
                    "Убедитесь, что вы хотя бы раз запускали наш Telegram бот, и введите правильный ID."
                ),
                attachments=retry_keyboard,
            )
            return

        linked_max_user = await session.scalar(
            select(User).where(User.id >= MAX_ID_OFFSET, User.tg_user_id == tg_id, User.id != user_id)
        )
        if linked_max_user:
            await client.send_message(
                chat_id=chat_id,
                text="❌ Этот Telegram аккаунт уже привязан к другому пользователю MAX.",
                attachments=retry_keyboard,
            )
            return

        max_user = await session.get(
            User,
            user_id,
            options=[selectinload(User.subscription).selectinload(UserSubscription.plan)],
        )
        if not max_user:
            return

        max_user.tg_user_id = tg_id

        now = utc_now()
        tg_has_sub = (
            tg_user.subscription
            and tg_user.subscription.end_date
            and tg_user.subscription.end_date > now
        )
        max_has_sub = (
            max_user.subscription
            and max_user.subscription.end_date
            and max_user.subscription.end_date > now
        )

        subscription_msg = ""
        if tg_has_sub and not max_has_sub:
            if max_user.subscription:
                max_user.subscription.plan_id = tg_user.subscription.plan_id
                max_user.subscription.start_date = now
                max_user.subscription.end_date = tg_user.subscription.end_date
                max_user.subscription.auto_renewal = False
                max_user.subscription.payment_provider = "Telegram Link"
                max_user.subscription.payment_method_id = None
                max_user.subscription.pending_robokassa_invoice_id = None
                max_user.subscription.last_payment_attempt = None
                max_user.subscription.payment_attempt_count = 0
            else:
                session.add(
                    UserSubscription(
                        user_id=user_id,
                        plan_id=tg_user.subscription.plan_id,
                        start_date=now,
                        end_date=tg_user.subscription.end_date,
                        auto_renewal=False,
                        payment_provider="Telegram Link",
                        payment_attempt_count=0,
                    )
                )
            plan_name = tg_user.subscription.plan.name if tg_user.subscription.plan else "подписка"
            subscription_msg = f"\n\n✅ Ваша подписка <b>{plan_name}</b> активирована в MAX!"
        elif tg_has_sub and max_has_sub:
            subscription_msg = "\n\nℹ️ У вас уже есть активная подписка в MAX."
        elif not tg_has_sub:
            subscription_msg = "\n\nℹ️ В Telegram боте нет активной подписки. Вы можете оформить её здесь."

        await session.commit()

    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ <b>Telegram аккаунт успешно привязан!</b>\n\n"
            f"TG ID: <code>{tg_id}</code>{subscription_msg}"
        ),
        attachments=inline_keyboard([[callback_button("◀️ К подписке", "sub_info")]]),
    )
