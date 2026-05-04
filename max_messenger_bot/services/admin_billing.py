from __future__ import annotations

import html

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..keyboards import (
    admin_plan_create_duration_unit_keyboard,
    admin_plan_duration_unit_keyboard,
    admin_plan_editor_keyboard,
    admin_plan_upgrade_keyboard,
    admin_plans_list_keyboard,
    admin_promo_assign_keyboard,
    admin_promo_editor_keyboard,
    admin_promocodes_list_keyboard,
)
from ..legacy import PromoCode, SubscriptionPlan, UserSubscription, async_session_maker
from ..storage import StateStore


def _format_duration(plan: SubscriptionPlan) -> str:
    unit = "дн." if plan.duration_unit == "days" else "мес."
    return f"{plan.duration_value} {unit}"


async def list_plans(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        plans = (await session.execute(select(SubscriptionPlan).order_by(SubscriptionPlan.price.asc(), SubscriptionPlan.id.asc()))).scalars().all()
    text = "📦 <b>Тарифы</b><br/><br/>Выберите тариф для редактирования." if plans else "📦 <b>Тарифы</b><br/><br/>Тарифов пока нет."
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_plans_list_keyboard(plans))


async def show_plan_editor(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id, options=[selectinload(SubscriptionPlan.upgrades_to_plan)])
    if not plan:
        await client.send_message(chat_id=chat_id, text="Тариф не найден.")
        return
    upgrade_name = plan.upgrades_to_plan.name if plan.upgrades_to_plan else "не задан"
    text = (
        f"<b>{html.escape(plan.name)}</b><br/><br/>"
        f"<b>ID:</b> {plan.id}<br/>"
        f"<b>Цена:</b> {plan.price:.2f} руб.<br/>"
        f"<b>Длительность:</b> {_format_duration(plan)}<br/>"
        f"<b>Активен:</b> {'да' if plan.is_active else 'нет'}<br/>"
        f"<b>Только для админов:</b> {'да' if getattr(plan, 'admin_only', False) else 'нет'}<br/>"
        f"<b>Пробный тариф:</b> {'да' if plan.is_trial else 'нет'}<br/>"
        f"<b>Автопродление:</b> {'да' if getattr(plan, 'allow_auto_renewal', True) else 'нет'}<br/>"
        f"<b>Апгрейд после триала:</b> {html.escape(upgrade_name)}<br/>"
        f"<b>Кулдаун триала:</b> {plan.trial_cooldown_days} дн.<br/><br/>"
        f"<b>Описание:</b><br/><pre><code>{html.escape(plan.description or 'Не задано')}</code></pre>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_plan_editor_keyboard(plan))


async def start_create_plan(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_plan_create_name", {})
    await client.send_message(chat_id=chat_id, text="Введите название нового тарифа.")


async def save_new_plan_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    name = text.strip()
    if not name:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    await states.set(user_id, chat_id, "admin_plan_create_description", {"name": name})
    await client.send_message(chat_id=chat_id, text="Введите описание тарифа.")


async def save_new_plan_description(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    data = dict(snapshot.data if snapshot else {})
    data["description"] = text.strip()
    await states.set(user_id, chat_id, "admin_plan_create_price", data)
    await client.send_message(chat_id=chat_id, text="Введите цену тарифа в рублях. Например: 990 или 990.50")


async def save_new_plan_price(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        price = float(text.strip().replace(",", "."))
        if price < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите корректную неотрицательную цену.")
        return
    snapshot = await states.get(user_id)
    data = dict(snapshot.data if snapshot else {})
    data["price"] = price
    await states.set(user_id, chat_id, "admin_plan_create_duration_unit", data)
    await client.send_message(
        chat_id=chat_id,
        text="Выберите единицу длительности тарифа.",
        attachments=admin_plan_create_duration_unit_keyboard(),
    )


async def save_new_plan_duration_unit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, duration_unit: str) -> None:
    if duration_unit not in {"days", "months"}:
        await client.send_message(chat_id=chat_id, text="Неизвестная единица длительности.")
        return
    snapshot = await states.get(user_id)
    data = dict(snapshot.data if snapshot else {})
    data["duration_unit"] = duration_unit
    await states.set(user_id, chat_id, "admin_plan_create_duration_value", data)
    unit_label = "дней" if duration_unit == "days" else "месяцев"
    await client.send_message(chat_id=chat_id, text=f"Введите длительность тарифа в {unit_label}.")


async def save_new_plan_duration_value(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        duration_value = int(text.strip())
        if duration_value <= 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое положительное число.")
        return
    snapshot = await states.get(user_id)
    data = dict(snapshot.data if snapshot else {})
    async with async_session_maker() as session:
        plan = SubscriptionPlan(
            name=data["name"],
            description=data.get("description") or None,
            price=data["price"],
            duration_value=duration_value,
            duration_unit=data["duration_unit"],
            is_active=True,
            admin_only=False,
            is_trial=False,
            trial_cooldown_days=0,
            allow_auto_renewal=True,
        )
        session.add(plan)
        await session.commit()
        plan_id = plan.id
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Тариф «{html.escape(data['name'])}» создан.")
    await show_plan_editor(client, chat_id, plan_id)


async def start_edit_plan_field(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    plan_id: int,
    state_name: str,
    prompt: str,
) -> None:
    await states.set(user_id, chat_id, state_name, {"plan_id": plan_id})
    await client.send_message(chat_id=chat_id, text=prompt)


async def save_plan_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    name = text.strip()
    if not name:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    snapshot = await states.get(user_id)
    plan_id = snapshot.data.get("plan_id") if snapshot else None
    if not plan_id:
        await client.send_message(chat_id=chat_id, text="Состояние тарифа потеряно.")
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.name = name
        await session.commit()
    await states.clear(user_id)
    await show_plan_editor(client, chat_id, plan_id)


async def save_plan_description(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    plan_id = snapshot.data.get("plan_id") if snapshot else None
    if not plan_id:
        await client.send_message(chat_id=chat_id, text="Состояние тарифа потеряно.")
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.description = text.strip() or None
        await session.commit()
    await states.clear(user_id)
    await show_plan_editor(client, chat_id, plan_id)


async def save_plan_price(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        price = float(text.strip().replace(",", "."))
        if price < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите корректную неотрицательную цену.")
        return
    snapshot = await states.get(user_id)
    plan_id = snapshot.data.get("plan_id") if snapshot else None
    if not plan_id:
        await client.send_message(chat_id=chat_id, text="Состояние тарифа потеряно.")
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.price = price
        await session.commit()
    await states.clear(user_id)
    await show_plan_editor(client, chat_id, plan_id)


async def save_plan_duration_value(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        duration_value = int(text.strip())
        if duration_value <= 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое положительное число.")
        return
    snapshot = await states.get(user_id)
    plan_id = snapshot.data.get("plan_id") if snapshot else None
    if not plan_id:
        await client.send_message(chat_id=chat_id, text="Состояние тарифа потеряно.")
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.duration_value = duration_value
        await session.commit()
    await states.clear(user_id)
    await show_plan_editor(client, chat_id, plan_id)


async def show_duration_unit_menu(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    await client.send_message(
        chat_id=chat_id,
        text="Выберите единицу длительности.",
        attachments=admin_plan_duration_unit_keyboard(plan_id, "admin_plan_set_duration_unit"),
    )


async def save_plan_duration_unit(client: MaxApiClient, chat_id: int, plan_id: int, duration_unit: str) -> None:
    if duration_unit not in {"days", "months"}:
        await client.send_message(chat_id=chat_id, text="Неизвестная единица длительности.")
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.duration_unit = duration_unit
        await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def save_plan_trial_cooldown(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        days = int(text.strip())
        if days < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое неотрицательное число.")
        return
    snapshot = await states.get(user_id)
    plan_id = snapshot.data.get("plan_id") if snapshot else None
    if not plan_id:
        await client.send_message(chat_id=chat_id, text="Состояние тарифа потеряно.")
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.trial_cooldown_days = days
        await session.commit()
    await states.clear(user_id)
    await show_plan_editor(client, chat_id, plan_id)


async def toggle_plan_active(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.is_active = not plan.is_active
            await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def toggle_plan_admin_only(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.admin_only = not getattr(plan, "admin_only", False)
            await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def toggle_plan_renewal(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.allow_auto_renewal = not getattr(plan, "allow_auto_renewal", True)
            await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def toggle_plan_trial(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if plan:
            plan.is_trial = not plan.is_trial
            if not plan.is_trial:
                plan.upgrades_to_plan_id = None
            await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def show_plan_upgrade_menu(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id, options=[selectinload(SubscriptionPlan.upgrades_to_plan)])
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plans = (
            await session.execute(
                select(SubscriptionPlan)
                .where(SubscriptionPlan.id != plan_id, SubscriptionPlan.is_trial == False)
                .order_by(SubscriptionPlan.price.asc(), SubscriptionPlan.id.asc())
            )
        ).scalars().all()
    text = (
        f"Выберите целевой тариф для «{html.escape(plan.name)}».<br/><br/>"
        "Если апгрейд не нужен, оставьте вариант «Без апгрейда»."
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=admin_plan_upgrade_keyboard(plan_id, plans, plan.upgrades_to_plan_id),
    )


async def set_plan_upgrade_target(client: MaxApiClient, chat_id: int, plan_id: int, target_plan_id: int | None) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        if target_plan_id is not None:
            target = await session.get(SubscriptionPlan, target_plan_id)
            if not target or target.id == plan_id or target.is_trial:
                await client.send_message(chat_id=chat_id, text="Целевой тариф недоступен.")
                return
        plan.upgrades_to_plan_id = target_plan_id
        await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def delete_plan(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        active_users_count = await session.scalar(select(func.count(UserSubscription.id)).where(UserSubscription.plan_id == plan_id)) or 0
        if active_users_count > 0:
            await client.send_message(chat_id=chat_id, text=f"Нельзя удалить тариф: он назначен {active_users_count} пользователям.")
            await show_plan_editor(client, chat_id, plan_id)
            return
        trial_links_count = await session.scalar(select(func.count(SubscriptionPlan.id)).where(SubscriptionPlan.upgrades_to_plan_id == plan_id)) or 0
        if trial_links_count > 0:
            await client.send_message(chat_id=chat_id, text=f"Нельзя удалить тариф: на него ссылаются {trial_links_count} пробных тарифов.")
            await show_plan_editor(client, chat_id, plan_id)
            return
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        await session.delete(plan)
        await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Тариф удалён.")
    await list_plans(client, chat_id)


async def list_promocodes(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        promos = (await session.execute(select(PromoCode).order_by(PromoCode.id.desc()))).scalars().all()
    text = "🎁 <b>Промокоды</b><br/><br/>Выберите промокод для редактирования." if promos else "🎁 <b>Промокоды</b><br/><br/>Промокодов пока нет."
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_promocodes_list_keyboard(promos))


async def show_promo_editor(client: MaxApiClient, chat_id: int, promo_id: int) -> None:
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id, options=[selectinload(PromoCode.applicable_plans)])
    if not promo:
        await client.send_message(chat_id=chat_id, text="Промокод не найден.")
        return
    scope = "все тарифы" if promo.applies_to_all_plans else f"{len(promo.applicable_plans)} тариф(ов)"
    text = (
        f"<b>{html.escape(promo.code)}</b><br/><br/>"
        f"<b>ID:</b> {promo.id}<br/>"
        f"<b>Скидка:</b> {promo.discount_percent}%<br/>"
        f"<b>Бесплатные дни:</b> {promo.free_days}<br/>"
        f"<b>Использовано:</b> {promo.times_used} / {promo.max_uses}<br/>"
        f"<b>Активен:</b> {'да' if promo.is_active else 'нет'}<br/>"
        f"<b>Применяется к:</b> {scope}"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_promo_editor_keyboard(promo))


async def start_create_promo(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_promo_create_code", {})
    await client.send_message(chat_id=chat_id, text="Введите текст нового промокода.")


async def save_new_promo_code(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    code = text.strip().upper()
    if not code:
        await client.send_message(chat_id=chat_id, text="Код не может быть пустым.")
        return
    async with async_session_maker() as session:
        exists = await session.scalar(select(PromoCode.id).where(PromoCode.code == code))
    if exists:
        await client.send_message(chat_id=chat_id, text="Промокод с таким кодом уже существует.")
        return
    await states.set(user_id, chat_id, "admin_promo_create_discount", {"code": code})
    await client.send_message(chat_id=chat_id, text="Введите скидку в процентах от 0 до 100.")


async def save_new_promo_discount(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        discount = int(text.strip())
        if not 0 <= discount <= 100:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое число от 0 до 100.")
        return
    snapshot = await states.get(user_id)
    data = dict(snapshot.data if snapshot else {})
    data["discount_percent"] = discount
    await states.set(user_id, chat_id, "admin_promo_create_days", data)
    await client.send_message(chat_id=chat_id, text="Введите количество бесплатных дней.")


async def save_new_promo_days(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        free_days = int(text.strip())
        if free_days < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое неотрицательное число.")
        return
    snapshot = await states.get(user_id)
    data = dict(snapshot.data if snapshot else {})
    data["free_days"] = free_days
    await states.set(user_id, chat_id, "admin_promo_create_uses", data)
    await client.send_message(chat_id=chat_id, text="Введите максимальное количество использований.")


async def save_new_promo_uses(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        max_uses = int(text.strip())
        if max_uses <= 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое положительное число.")
        return
    snapshot = await states.get(user_id)
    data = dict(snapshot.data if snapshot else {})
    async with async_session_maker() as session:
        promo = PromoCode(
            code=data["code"],
            discount_percent=data["discount_percent"],
            free_days=data["free_days"],
            max_uses=max_uses,
            times_used=0,
            is_active=True,
            applies_to_all_plans=True,
        )
        session.add(promo)
        await session.commit()
        promo_id = promo.id
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Промокод «{html.escape(data['code'])}» создан.")
    await show_promo_editor(client, chat_id, promo_id)


async def start_edit_promo_field(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    promo_id: int,
    state_name: str,
    prompt: str,
) -> None:
    await states.set(user_id, chat_id, state_name, {"promo_id": promo_id})
    await client.send_message(chat_id=chat_id, text=prompt)


async def save_promo_code(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    code = text.strip().upper()
    if not code:
        await client.send_message(chat_id=chat_id, text="Код не может быть пустым.")
        return
    snapshot = await states.get(user_id)
    promo_id = snapshot.data.get("promo_id") if snapshot else None
    if not promo_id:
        await client.send_message(chat_id=chat_id, text="Состояние промокода потеряно.")
        return
    async with async_session_maker() as session:
        exists = await session.scalar(select(PromoCode.id).where(PromoCode.code == code, PromoCode.id != promo_id))
        if exists:
            await client.send_message(chat_id=chat_id, text="Промокод с таким кодом уже существует.")
            return
        promo = await session.get(PromoCode, promo_id)
        if not promo:
            await client.send_message(chat_id=chat_id, text="Промокод не найден.")
            return
        promo.code = code
        await session.commit()
    await states.clear(user_id)
    await show_promo_editor(client, chat_id, promo_id)


async def save_promo_discount(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        discount = int(text.strip())
        if not 0 <= discount <= 100:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое число от 0 до 100.")
        return
    snapshot = await states.get(user_id)
    promo_id = snapshot.data.get("promo_id") if snapshot else None
    if not promo_id:
        await client.send_message(chat_id=chat_id, text="Состояние промокода потеряно.")
        return
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if not promo:
            await client.send_message(chat_id=chat_id, text="Промокод не найден.")
            return
        promo.discount_percent = discount
        await session.commit()
    await states.clear(user_id)
    await show_promo_editor(client, chat_id, promo_id)


async def save_promo_days(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        free_days = int(text.strip())
        if free_days < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое неотрицательное число.")
        return
    snapshot = await states.get(user_id)
    promo_id = snapshot.data.get("promo_id") if snapshot else None
    if not promo_id:
        await client.send_message(chat_id=chat_id, text="Состояние промокода потеряно.")
        return
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if not promo:
            await client.send_message(chat_id=chat_id, text="Промокод не найден.")
            return
        promo.free_days = free_days
        await session.commit()
    await states.clear(user_id)
    await show_promo_editor(client, chat_id, promo_id)


async def save_promo_uses(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        max_uses = int(text.strip())
        if max_uses <= 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое положительное число.")
        return
    snapshot = await states.get(user_id)
    promo_id = snapshot.data.get("promo_id") if snapshot else None
    if not promo_id:
        await client.send_message(chat_id=chat_id, text="Состояние промокода потеряно.")
        return
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if not promo:
            await client.send_message(chat_id=chat_id, text="Промокод не найден.")
            return
        promo.max_uses = max_uses
        await session.commit()
    await states.clear(user_id)
    await show_promo_editor(client, chat_id, promo_id)


async def toggle_promo_active(client: MaxApiClient, chat_id: int, promo_id: int) -> None:
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if promo:
            promo.is_active = not promo.is_active
            await session.commit()
    await show_promo_editor(client, chat_id, promo_id)


async def toggle_promo_all_plans(client: MaxApiClient, chat_id: int, promo_id: int) -> None:
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if promo:
            promo.applies_to_all_plans = not promo.applies_to_all_plans
            await session.commit()
    await show_promo_editor(client, chat_id, promo_id)


async def toggle_promo_all_plans_from_assign_menu(client: MaxApiClient, chat_id: int, promo_id: int) -> None:
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if promo:
            promo.applies_to_all_plans = not promo.applies_to_all_plans
            await session.commit()
    await show_promo_assign_menu(client, chat_id, promo_id)


async def show_promo_assign_menu(client: MaxApiClient, chat_id: int, promo_id: int) -> None:
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id, options=[selectinload(PromoCode.applicable_plans)])
        if not promo:
            await client.send_message(chat_id=chat_id, text="Промокод не найден.")
            return
        plans = (await session.execute(select(SubscriptionPlan).order_by(SubscriptionPlan.price.asc(), SubscriptionPlan.id.asc()))).scalars().all()
        assigned_plan_ids = {plan.id for plan in promo.applicable_plans}
    text = (
        f"Привязка тарифов к промокоду «{html.escape(promo.code)}».<br/><br/>"
        "Если включён режим для всех тарифов, список ниже не ограничивает применение."
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=admin_promo_assign_keyboard(promo_id, plans, assigned_plan_ids, promo.applies_to_all_plans),
    )


async def toggle_promo_plan_assignment(client: MaxApiClient, chat_id: int, promo_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id, options=[selectinload(PromoCode.applicable_plans)])
        plan = await session.get(SubscriptionPlan, plan_id)
        if not promo or not plan:
            await client.send_message(chat_id=chat_id, text="Промокод или тариф не найден.")
            return
        assigned_ids = {item.id for item in promo.applicable_plans}
        if plan_id in assigned_ids:
            promo.applicable_plans = [item for item in promo.applicable_plans if item.id != plan_id]
        else:
            promo.applicable_plans.append(plan)
        await session.commit()
    await show_promo_assign_menu(client, chat_id, promo_id)


async def delete_promo(client: MaxApiClient, chat_id: int, promo_id: int) -> None:
    async with async_session_maker() as session:
        promo = await session.get(PromoCode, promo_id)
        if not promo:
            await client.send_message(chat_id=chat_id, text="Промокод не найден.")
            return
        await session.delete(promo)
        await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Промокод удалён.")
    await list_promocodes(client, chat_id)


async def toggle_plan_is_trial(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.is_trial = not getattr(plan, "is_trial", False)
        await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def show_upgrade_target_picker(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    from ..keyboards import callback_button, inline_keyboard
    async with async_session_maker() as session:
        plans = (await session.execute(select(SubscriptionPlan))).scalars().all()
        current_plan = await session.get(SubscriptionPlan, plan_id)
    text = f"Выберите тариф для апгрейда из «{html.escape(current_plan.name if current_plan else str(plan_id))}»:"
    rows = []
    for p in plans:
        if p.id != plan_id:
            marker = "✅ " if current_plan and current_plan.upgrades_to_plan_id == p.id else ""
            rows.append([callback_button(f"{marker}{html.escape(p.name)} ({p.price:.0f}₽)", f"admin_plan_set_upgrade_{plan_id}_{p.id}")])
    rows.append([callback_button("❌ Убрать апгрейд", f"admin_plan_clear_upgrade_{plan_id}")])
    rows.append([callback_button("◀️ Назад", f"admin_edit_plan_{plan_id}")])
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(rows))


async def set_upgrade_target(client: MaxApiClient, chat_id: int, plan_id: int, target_plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return
        plan.upgrades_to_plan_id = target_plan_id
        await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def clear_upgrade_target(client: MaxApiClient, chat_id: int, plan_id: int) -> None:
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
        if not plan:
            return
        plan.upgrades_to_plan_id = None
        await session.commit()
    await show_plan_editor(client, chat_id, plan_id)


async def reset_user_promos(client: MaxApiClient, chat_id: int, target_user_id: int) -> None:
    from ..keyboards import callback_button, inline_keyboard
    from ..legacy import User
    async with async_session_maker() as session:
        user = await session.get(User, target_user_id)
        if user:
            user.applied_promo_code = None
            await session.commit()
    await client.send_message(
        chat_id=chat_id,
        text=f"✅ Промокоды пользователя <code>{target_user_id}</code> сброшены.",
        attachments=inline_keyboard([[callback_button("◀️ Назад", f"view_client_{target_user_id}")]]),
    )
