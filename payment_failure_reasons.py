YOOKASSA_CANCELLATION_REASON_LABELS = {
    '3d_secure_failed': 'не пройдена проверка 3-D Secure',
    'call_issuer': 'банк отклонил операцию; нужно обратиться в банк',
    'canceled_by_merchant': 'платёж отменён магазином',
    'card_expired': 'истёк срок действия карты',
    'country_forbidden': 'карта выпущена в неподдерживаемой стране',
    'deal_expired': 'истёк срок действия сделки',
    'expired_on_capture': 'истёк срок подтверждения списания',
    'expired_on_confirmation': 'пользователь не завершил подтверждение оплаты',
    'fraud_suspected': 'операция отклонена из-за подозрения в мошенничестве',
    'general_decline': 'операция отклонена без уточнения причины',
    'identification_required': 'требуется идентификация кошелька ЮMoney',
    'insufficient_funds': 'недостаточно средств',
    'internal_timeout': 'тайм-аут на стороне ЮKassa',
    'invalid_card_number': 'неверный номер карты',
    'invalid_csc': 'неверный код безопасности карты',
    'issuer_unavailable': 'банк временно недоступен',
    'loan_application_expired': 'истёк срок заполнения заявки на кредит',
    'loan_declined': 'банк отклонил заявку на кредит',
    'loan_declined_by_payer': 'пользователь отказался от кредита',
    'payment_method_limit_exceeded': 'превышен лимит платежей',
    'payment_method_restricted': 'операции по этому платёжному средству запрещены',
    'permission_revoked': 'пользователь отозвал разрешение на автоплатежи',
    'unsupported_mobile_operator': 'оператор мобильной связи не поддерживается',
}


def get_yookassa_cancellation_reason(payment) -> str | None:
    details = getattr(payment, "cancellation_details", None)
    return getattr(details, "reason", None) if details else None


def format_yookassa_cancellation_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    return YOOKASSA_CANCELLATION_REASON_LABELS.get(reason, f"код ЮKassa: {reason}")


def format_yookassa_admin_reason_line(reason: str | None) -> str:
    text = format_yookassa_cancellation_reason(reason)
    if not text:
        return "\nПричина: ЮKassa не указала"
    if reason in YOOKASSA_CANCELLATION_REASON_LABELS:
        return f"\nПричина: {text} ({reason})"
    return f"\nПричина: {text}"
