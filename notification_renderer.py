from __future__ import annotations

from typing import Any


def render_outbox_message(
    event_type: str,
    payload: dict[str, Any],
    recipient_id: int,
) -> tuple[str, str | None, str | None]:
    """
    Renders user notification text, parse_mode, and keyboard_type from an immutable event snapshot.
    Returns: (text, parse_mode, keyboard_type)
    """
    if event_type == "purchase_success":
        provider = payload.get("provider", "Yookassa")
        if provider == "Robokassa" and "end_date_msk" in payload:
            amount = payload.get("amount", 0.0)
            plan_name = payload.get("plan_name", "")
            end_date_msk = payload.get("end_date_msk", "")
            text = (
                f"Мы получили оплату {amount:.2f} руб по вашему тарифу «{plan_name}».\n"
                f"Действие тарифа продлено до {end_date_msk} МСК.\n\n"
                f"Благодарим, что продолжаете пользоваться ботом!\n"
                f"Вы всегда можете направить нам свои пожелания, предложения по его работе."
            )
            return text, None, None
        plan_name = payload.get("plan_name", "")
        return f"✅ Ваша подписка на тариф «{plan_name}» успешно оформлена!", None, None

    if event_type == "renewal_success":
        provider = payload.get("provider", "Yookassa")
        end_date_msk = payload.get("end_date_msk", "")
        if provider == "Robokassa":
            amount = payload.get("amount", 0.0)
            plan_name = payload.get("plan_name", "")
            text = (
                f"Мы получили оплату {amount:.2f} руб по вашему тарифу «{plan_name}».\n"
                f"Действие тарифа продлено до {end_date_msk} МСК.\n\n"
                f"Благодарим, что продолжаете пользоваться ботом!\n"
                f"Вы всегда можете направить нам свои пожелания, предложения по его работе."
            )
            return text, None, None
        else:
            return f"✅ Подписка продлена до {end_date_msk} МСК.", None, None

    if event_type == "referral_bonus":
        bonus_days = payload.get("bonus_days", 0)
        return f"💰 Ваш реферал оформил подписку! Вам начислено <b>{bonus_days} бонусных дн.</b>", "HTML", None

    if event_type == "deactivate":
        provider = payload.get("provider", "Yookassa")
        if provider == "Robokassa":
            text = (
                "Ваша подписка истекла. Ошибка при автоплатеже (Robokassa) — автопродление отключено.\n\n"
                "Продлите подписку вручную в меню."
            )
        else:
            text = "Ваша подписка истекла. Сохранённый способ оплаты больше недоступен в ЮKassa (автопродление отключено).\n\nПродлите подписку вручную в меню."
        return text, None, "subscribe"

    if event_type == "unknown_cancellation":
        text = (
            "Не удалось выполнить автоматическое списание (нестандартный ответ банка). "
            "Автопродление приостановлено во избежание повторных списаний.\n\n"
            "Пожалуйста, оформите или продлите подписку вручную в меню бота."
        )
        return text, None, "subscribe"

    if event_type == "unknown_expired":
        text = (
            "Не удалось подтвердить результат списания за 24 часа. Чтобы избежать двойных списаний, автопродление приостановлено.\n\n"
            "Проверьте статус в банке или оформите подписку в меню."
        )
        return text, None, "subscribe"

    if event_type == "final_decline":
        text = (
            "Ваша подписка истекла. Не удалось списать средства после 3 попыток — автопродление отключено.\n\n"
            "Продлите подписку вручную в меню."
        )
        return text, None, "subscribe"

    if event_type.startswith("retry_limit"):
        text = (
            "Не удалось провести списание (превышен лимит по карте). Следующая попытка будет завтра. "
            "Вы также можете привязать другую карту в меню."
        )
        return text, None, "subscribe"

    if event_type in ("retry_failed_1", "retry_failed") and payload.get("attempt_count") == 1:
        next_retry = payload.get("next_retry_str", "позже")
        provider = payload.get("provider", "ЮKassa")
        return f"Не удалось списать средства ({provider}). Повторим попытку {next_retry}.", None, "subscribe"

    if event_type in ("retry_failed_2", "retry_failed") and payload.get("attempt_count") == 2:
        next_retry = payload.get("next_retry_str", "позже")
        provider = payload.get("provider", "ЮKassa")
        return f"Не удалось списать средства ({provider}). Последняя попытка — {next_retry}.", None, "subscribe"

    if event_type.startswith("retry_failed"):
        next_retry = payload.get("next_retry_str", "позже")
        provider = payload.get("provider", "ЮKassa")
        att = payload.get("attempt_count", 1)
        prefix = f"Не удалось списать средства ({provider})."
        if att == 1:
            return f"{prefix} Повторим попытку {next_retry}.", None, "subscribe"
        else:
            return f"{prefix} Последняя попытка — {next_retry}.", None, "subscribe"

    if event_type == "provider_error":
        provider = payload.get("provider", "Yookassa")
        if provider == "Robokassa":
            next_retry = payload.get("next_retry_str", "позже")
            text = (
                f"Платёжный шлюз Robokassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\n"
                f"Повторим запрос после {next_retry}."
            )
            return text, None, "subscribe"
        text = "Платёжный шлюз ЮKassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\nПовторим запрос позже."
        return text, None, "subscribe"

    if "cross_plan" in event_type:
        amount = payload.get("amount", 0.0)
        paid_name = payload.get("paid_plan_name", "")
        curr_name = payload.get("current_plan_name", "")
        text = (
            f"⚠️ Мы получили оплату ({amount:.2f} руб) по вашему предыдущему тарифу «{paid_name}». "
            f"Поскольку сейчас у вас активен тариф «{curr_name}», платёж отправлен на проверку администратору. "
            f"Срок действия текущей подписки не был изменён автоматически."
        )
        return text, None, None

    if "subscription_unresolved" in event_type:
        amount = payload.get("amount", 0.0)
        text = (
            f"⚠️ Мы получили оплату ({amount:.2f} руб), но не удалось найти вашу подписку для автоматического продления. "
            f"Платёж отправлен на проверку администратору."
        )
        return text, None, None

    if "paid_plan_unresolved" in event_type:
        amount = payload.get("amount", 0.0)
        text = (
            f"⚠️ Мы получили оплату ({amount:.2f} руб), но оплаченный тариф не найден в системе. "
            f"Платёж отправлен на проверку администратору. Срок действия текущей подписки не был изменён автоматически."
        )
        return text, None, None

    # Fallback
    raw_text = payload.get("text", f"Уведомление по платежу ({event_type})")
    return raw_text, payload.get("parse_mode"), payload.get("keyboard_type")
