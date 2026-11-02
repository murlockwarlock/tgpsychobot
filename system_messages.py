from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import html
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import OperationalError

from database import (
    ADDRESSING_STYLES,
    BOT_SCOPES,
    DEFAULT_BOT_SCOPE,
    DEFAULT_ADDRESSING_STYLE,
    BotAddressingConfig,
    BotGeneralConfig,
    BotSystemMessage,
)
from translation_service import normalize_locale


@dataclass(frozen=True)
class SystemMessageDefinition:
    key: str
    title: str
    defaults: dict[str, dict[str, str]]
    description: str = ""
    platforms: tuple[str, ...] = ("telegram", "max")

    def supports_platform(self, platform: str) -> bool:
        return str(platform or "").strip().lower() in self.platforms


def _styles(informal: str, formal: str | None = None) -> dict[str, str]:
    return {
        "informal": informal,
        "formal": formal if formal is not None else informal,
    }


SYSTEM_MESSAGE_CATALOG = (
    SystemMessageDefinition("ui.ai.thinking", "ИИ: ожидание", {"ru": _styles("🤖 Думаю...")}),
    SystemMessageDefinition("ui.ai.processing_default", "ИИ: сообщение ожидания", {"ru": _styles("Думаю...")}),
    SystemMessageDefinition("ui.ai.button_accepted", "ИИ: ответ на кнопку", {"ru": _styles("Ответ принят: {button_text}")}),
    SystemMessageDefinition("ui.ai.choose_action", "ИИ: выбор действия", {"ru": _styles("Выберите действие:")}),
    SystemMessageDefinition("ui.ai.image_ready", "ИИ: изображение готово", {"ru": _styles("✨ Готово!")}),
    SystemMessageDefinition("ui.ai.image_failed", "ИИ: ошибка генерации", {"ru": _styles("😔 Не удалось сгенерировать изображение.")}),
    SystemMessageDefinition("ui.ai.service_unavailable", "ИИ: сервис недоступен", {"ru": _styles("К сожалению, сервис временно недоступен из-за технической проблемы.")}),
    SystemMessageDefinition("ui.ai.service_unavailable.max", "ИИ MAX: сервис недоступен", {"ru": _styles("Сервис ИИ временно недоступен. Попробуйте позже.")}),
    SystemMessageDefinition("ui.ai.overloaded", "ИИ: перегрузка", {"ru": _styles(
        "Ой. Нейросеть сейчас перегружена и не отвечает. Загляни через несколько минут и повтори запрос. Я буду ждать!",
        "Ой. Нейросеть сейчас перегружена и не отвечает. Загляните через несколько минут и повторите запрос. Я буду ждать!",
    )}),
    SystemMessageDefinition("ui.ai.overloaded.max", "ИИ MAX: перегрузка", {"ru": _styles(
        "Ой. Нейросеть сейчас перегружена и не отвечает. Загляни через несколько минут и повтори запрос.",
        "Ой. Нейросеть сейчас перегружена и не отвечает. Загляните через несколько минут и повторите запрос.",
    )}),
    SystemMessageDefinition("ui.ai.retry_request", "ИИ: повторная отправка", {"ru": _styles("🔄 Повторить запрос")}),
    SystemMessageDefinition("ui.ai.retry_stale", "ИИ: устаревший повтор", {"ru": _styles(
        "Этот запрос уже относится к предыдущему диалогу. Отправьте сообщение заново.",
    )}),
    SystemMessageDefinition("ui.ai.retry_already_used", "ИИ: повтор уже выполнен", {"ru": _styles("Этот запрос уже был отправлен на повторную обработку.")}),
    SystemMessageDefinition("ui.ai.processing_failed", "ИИ: ошибка обработки", {"ru": _styles("Произошла ошибка при обработке сообщения.")}),
    SystemMessageDefinition("ui.ai.internal_failed", "ИИ MAX: внутренняя ошибка", {"ru": _styles("Произошла внутренняя ошибка. Попробуйте позже.")}),
    SystemMessageDefinition("ui.ai.unexpected_failed", "ИИ: непредвиденная ошибка", {"ru": _styles(
        "Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.",
    )}),
    SystemMessageDefinition("ui.ai.busy", "ИИ: запрос уже выполняется", {"ru": _styles(
        "Пожалуйста, не так быстро — я ещё разбираю твоё предыдущее сообщение. Дай мне немного времени, чтобы всё хорошенько обдумать.",
        "Пожалуйста, не так быстро — я ещё разбираю Ваше предыдущее сообщение. Дайте мне немного времени, чтобы всё хорошенько обдумать.",
    )}),
    SystemMessageDefinition("ui.ai.image_inspecting", "ИИ: анализ изображения", {"ru": _styles("👀 Тщательно изучаю изображение...")}),
    SystemMessageDefinition(
        "ui.ai.image_analyzing",
        "ИИ MAX: анализ изображения",
        {
            "ru": _styles("🤖 Анализирую изображение..."),
            "en": _styles("🤖 Analyzing the image..."),
            "pt": _styles("🤖 Analisando a imagem..."),
        },
    ),
    SystemMessageDefinition("ui.ai.editing_image", "ИИ: редактирование изображения", {"ru": _styles("🎨 Редактирую ваше фото...")}),
    SystemMessageDefinition("ui.ai.edited_image", "ИИ: результат редактирования", {"ru": _styles("✨ Результат редактирования:")}),
    SystemMessageDefinition("ui.ai.edit_image_failed", "ИИ: ошибка редактирования", {"ru": _styles("😔 К сожалению, не удалось отредактировать изображение. Возможно, сервис дал сбой или запрос был отклонен фильтрами безопасности.")}),
    SystemMessageDefinition("ui.ai.generating_image", "ИИ: генерация изображения", {"ru": _styles("🖼 Генерирую новое изображение...")}),
    SystemMessageDefinition("ui.ai.generated_image", "ИИ: новая генерация", {"ru": _styles("✨ Новая генерация:")}),
    SystemMessageDefinition("ui.ai.photo_failed", "ИИ: ошибка обработки фото", {"ru": _styles("Произошла ошибка при обработке фото.")}),
    SystemMessageDefinition("ui.card.selection_prompt", "Карты: выбор карты", {"ru": _styles("Выбери карту, которая тебе откликается:", "Выберите карту, которая Вам откликается:")}),
    SystemMessageDefinition("ui.card.next_selection_prompt", "Карты: следующая карта", {"ru": _styles("Выбери следующую карту:", "Выберите следующую карту:")}),
    SystemMessageDefinition("ui.card.stale_choice", "Карты: устаревший выбор", {"ru": _styles("Эта карта уже не участвует в текущем выборе.")}),
    SystemMessageDefinition("ui.card.invalid_back", "Карты: недоступная рубашка", {"ru": _styles("Эта техническая рубашка не должна выбираться.")}),
    SystemMessageDefinition("ui.card.selected", "Карты: выбор подтверждён", {"ru": _styles("Твой выбор подтвержден.", "Ваш выбор подтвержден.")}),
    SystemMessageDefinition("ui.card.interpretation_failed", "Карты: ошибка интерпретации", {"ru": _styles(
        "Не удалось получить интерпретацию карты. Попробуйте выбрать карту ещё раз через несколько минут.",
    )}),
    SystemMessageDefinition("ui.card.not_found", "Карты: карта не найдена", {"ru": _styles(
        "Ошибка: карта не найдена. Попробуйте начать выбор заново.",
    )}),
    SystemMessageDefinition("ui.card.spread_heading", "Карты: расклад", {"ru": _styles("Твой расклад целиком:", "Ваш расклад целиком:")}),
    SystemMessageDefinition("ui.card.next_failed", "Карты: ошибка следующего шага", {"ru": _styles(
        "Не удалось показать следующие карты. Попробуй выбрать карту ещё раз или начни расклад заново.",
        "Не удалось показать следующие карты. Попробуйте выбрать карту ещё раз или начните расклад заново.",
    )}),
    SystemMessageDefinition("ui.card.round_prompt", "Карты: раунд", {"ru": _styles(
        "Сейчас раунд {current_round} из {total_rounds}. Выбери одну из карт кнопкой ниже:",
        "Сейчас раунд {current_round} из {total_rounds}. Выберите одну из карт кнопкой ниже:",
    )}),
    SystemMessageDefinition("ui.navigation.continue_prompt", "Навигация: продолжение диалога", {"ru": _styles("Введите ваше сообщение для начала/продолжения диалога:")}),
    SystemMessageDefinition("ui.navigation.main_resume", "Навигация: возврат в диалог", {"ru": _styles("✅ Мы вернулись в основной диалог.")}),
    SystemMessageDefinition("ui.navigation.continue_ready", "Навигация: диалог готов", {"ru": _styles("Спасибо! Теперь вы можете задать свой вопрос.")}),
    SystemMessageDefinition("ui.navigation.user_error", "Навигация: ошибка пользователя", {"ru": _styles("Ошибка пользователя.")}),
    SystemMessageDefinition("ui.access.subscription_required", "Доступ: нужна подписка", {"ru": _styles("Чтобы продолжить диалог, активируйте подписку / бонусные дни.")}),
    SystemMessageDefinition("ui.access.photo_subscription_required", "Доступ: нужна подписка для фото", {"ru": _styles("Чтобы отправлять фото и получать разборы, активируйте подписку.")}),
)


SYSTEM_MESSAGE_ADMIN_METADATA = {
    "ui.ai.thinking": ("Нейросеть готовит ответ", "Показывается, пока бот готовит ответ на сообщение."),
    "ui.ai.processing_default": ("Сообщение во время ожидания ответа", "Временно показывается до ответа нейросети, а затем удаляется."),
    "ui.ai.button_accepted": ("Подтверждение кнопки", "Показывается после нажатия кнопки с вариантом ответа."),
    "ui.ai.choose_action": ("Выбор действия", "Показывается, когда пользователю нужно выбрать следующий шаг."),
    "ui.ai.image_ready": ("Изображение готово", "Показывается после успешного создания изображения."),
    "ui.ai.image_failed": ("Не удалось создать изображение", "Показывается, если создать изображение не удалось."),
    "ui.ai.service_unavailable": ("Сервис ИИ временно недоступен", "Показывается при временной недоступности сервиса ИИ."),
    "ui.ai.service_unavailable.max": ("Сервис ИИ временно недоступен", "Показывается пользователю MAX при временной недоступности сервиса ИИ."),
    "ui.ai.overloaded": ("Нейросеть временно перегружена", "Показывается, когда нейросеть не успевает обработать запрос."),
    "ui.ai.overloaded.max": ("Нейросеть временно перегружена", "Показывается пользователю MAX при перегрузке нейросети."),
    "ui.ai.retry_request": ("Повторить запрос", "Текст кнопки для повторной отправки запроса."),
    "ui.ai.retry_stale": ("Повтор больше не доступен", "Показывается, если кнопка повтора относится к предыдущему диалогу."),
    "ui.ai.retry_already_used": ("Повтор уже выполнен", "Показывается после повторного нажатия уже использованной кнопки."),
    "ui.ai.processing_failed": ("Не удалось обработать сообщение", "Показывается, если обработка сообщения завершилась ошибкой."),
    "ui.ai.internal_failed": ("Не удалось обработать запрос", "Показывается пользователю MAX при неожиданной внутренней ошибке."),
    "ui.ai.unexpected_failed": ("Произошла непредвиденная ошибка", "Показывается при ошибке, для которой нет отдельного сообщения."),
    "ui.ai.busy": ("Предыдущий запрос ещё выполняется", "Показывается, если пользователь отправляет новый запрос слишком быстро."),
    "ui.ai.image_inspecting": ("Изучение изображения", "Показывается во время подготовки изображения к анализу в Telegram."),
    "ui.ai.image_analyzing": ("Анализ изображения", "Показывается во время анализа изображения в MAX."),
    "ui.ai.editing_image": ("Редактирование изображения", "Показывается во время изменения фотографии."),
    "ui.ai.edited_image": ("Результат редактирования", "Показывается перед результатом изменения фотографии."),
    "ui.ai.edit_image_failed": ("Не удалось отредактировать изображение", "Показывается, если изменить фотографию не удалось."),
    "ui.ai.generating_image": ("Создание изображения", "Показывается во время создания нового изображения."),
    "ui.ai.generated_image": ("Результат создания изображения", "Показывается перед созданным изображением."),
    "ui.ai.photo_failed": ("Не удалось обработать фотографию", "Показывается, если обработка фотографии завершилась ошибкой."),
    "ui.card.selection_prompt": ("Выбор карты", "Показывается, когда пользователю нужно выбрать карту."),
    "ui.card.next_selection_prompt": ("Выбор следующей карты", "Показывается перед выбором следующей карты."),
    "ui.card.stale_choice": ("Выбор карты больше недоступен", "Показывается, если карта уже не относится к текущему раскладу."),
    "ui.card.invalid_back": ("Недоступная карта", "Показывается при попытке выбрать недоступную карту."),
    "ui.card.selected": ("Выбор карты подтверждён", "Показывается после подтверждения выбранной карты."),
    "ui.card.interpretation_failed": ("Не удалось истолковать карту", "Показывается, если получить толкование карты не удалось."),
    "ui.card.not_found": ("Карта не найдена", "Показывается, если выбранная карта больше недоступна."),
    "ui.card.spread_heading": ("Готовый расклад", "Показывается перед итогом расклада."),
    "ui.card.next_failed": ("Не удалось показать следующие карты", "Показывается, если следующий шаг расклада не загрузился."),
    "ui.card.round_prompt": ("Выбор карты в раунде", "Показывается перед выбором карты в текущем раунде."),
    "ui.navigation.continue_prompt": ("Начало или продолжение диалога", "Показывается, когда бот ждёт первое или следующее сообщение пользователя."),
    "ui.navigation.main_resume": ("Возврат в основной диалог", "Показывается после возврата пользователя в основной диалог."),
    "ui.navigation.continue_ready": ("Диалог готов к продолжению", "Показывается, когда пользователь может задать следующий вопрос."),
    "ui.navigation.user_error": ("Ошибка в данных пользователя", "Показывается, если данные пользователя нельзя обработать."),
    "ui.access.subscription_required": ("Нужна подписка", "Показывается, когда для продолжения нужна активная подписка."),
    "ui.access.photo_subscription_required": ("Нужна подписка для фотографий", "Показывается, когда для работы с фотографией нужна активная подписка."),
}

SYSTEM_MESSAGE_PLATFORM_SCOPES = {
    "ui.ai.processing_default": ("telegram",),
    "ui.ai.button_accepted": ("telegram",),
    "ui.ai.image_ready": ("telegram",),
    "ui.ai.image_failed": ("telegram",),
    "ui.ai.overloaded": ("telegram",),
    "ui.ai.processing_failed": ("telegram",),
    "ui.ai.unexpected_failed": ("telegram",),
    "ui.ai.image_inspecting": ("telegram",),
    "ui.ai.edited_image": ("telegram",),
    "ui.ai.edit_image_failed": ("telegram",),
    "ui.ai.generated_image": ("telegram",),
    "ui.ai.photo_failed": ("telegram",),
    "ui.card.selection_prompt": ("telegram",),
    "ui.card.next_selection_prompt": ("telegram",),
    "ui.card.stale_choice": ("telegram",),
    "ui.card.invalid_back": ("telegram",),
    "ui.card.selected": ("telegram",),
    "ui.card.interpretation_failed": ("telegram",),
    "ui.card.not_found": ("telegram",),
    "ui.card.spread_heading": ("telegram",),
    "ui.card.next_failed": ("telegram",),
    "ui.card.round_prompt": ("telegram",),
    "ui.navigation.main_resume": ("telegram",),
    "ui.navigation.continue_ready": ("telegram",),
    "ui.navigation.user_error": ("telegram",),
    "ui.access.subscription_required": ("telegram",),
    "ui.access.photo_subscription_required": ("telegram",),
    "ui.ai.service_unavailable.max": ("max",),
    "ui.ai.overloaded.max": ("max",),
    "ui.ai.internal_failed": ("max",),
    "ui.ai.image_analyzing": ("max",),
}

SYSTEM_MESSAGE_CATALOG = tuple(
    replace(
        definition,
        title=SYSTEM_MESSAGE_ADMIN_METADATA[definition.key][0],
        description=SYSTEM_MESSAGE_ADMIN_METADATA[definition.key][1],
        platforms=SYSTEM_MESSAGE_PLATFORM_SCOPES.get(definition.key, ("telegram", "max")),
    )
    for definition in SYSTEM_MESSAGE_CATALOG
)

SYSTEM_MESSAGE_AUTHORING_LOCALES = ("ru", "en", "pt")


def system_message_authoring_locales(
    platform: str,
    *,
    multilingual_enabled: bool = False,
) -> tuple[str, ...]:
    if str(platform or "").strip().lower() == "telegram" and multilingual_enabled:
        return SYSTEM_MESSAGE_AUTHORING_LOCALES
    return ("ru",)


def system_message_locale_is_available(
    platform: str,
    locale: str,
    *,
    multilingual_enabled: bool = False,
) -> bool:
    return normalize_locale(locale) in system_message_authoring_locales(
        platform,
        multilingual_enabled=multilingual_enabled,
    )


def system_message_catalog_for_platform(platform: str) -> tuple[SystemMessageDefinition, ...]:
    normalized_platform = str(platform or "").strip().lower()
    return tuple(
        definition
        for definition in SYSTEM_MESSAGE_CATALOG
        if definition.supports_platform(normalized_platform)
    )


def system_message_page_count(platform: str) -> int:
    definitions = system_message_catalog_for_platform(platform)
    return max(1, (len(definitions) + SYSTEM_MESSAGE_PAGE_SIZE - 1) // SYSTEM_MESSAGE_PAGE_SIZE)


def system_message_page_definitions(platform: str, page: int) -> tuple[SystemMessageDefinition, ...]:
    definitions = system_message_catalog_for_platform(platform)
    total_pages = system_message_page_count(platform)
    current_page = max(0, min(int(page), total_pages - 1))
    start = current_page * SYSTEM_MESSAGE_PAGE_SIZE
    return definitions[start:start + SYSTEM_MESSAGE_PAGE_SIZE]


SYSTEM_MESSAGE_DEFINITIONS = {item.key: item for item in SYSTEM_MESSAGE_CATALOG}
SYSTEM_MESSAGE_PAGE_SIZE = 8
TELEGRAM_TEXT_LIMIT = 4096
CALLBACK_ANSWER_LIMIT = 200
SYSTEM_MESSAGE_CALLBACK_TOKENS = {
    key: hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]
    for key in SYSTEM_MESSAGE_DEFINITIONS
}
SYSTEM_MESSAGE_KEYS_BY_CALLBACK_TOKEN = {
    token: key for key, token in SYSTEM_MESSAGE_CALLBACK_TOKENS.items()
}


def system_message_callback_data(
    action: str,
    key: str | None = None,
    locale: str = "ru",
    style: str | None = None,
    page: int = 0,
) -> str:
    if action in {"d", "m"}:
        parts = ("sm", action, system_message_callback_token(key or ""), locale, str(page))
    elif action in {"v", "e", "r", "s", "c"}:
        parts = (
            "sm",
            action,
            system_message_callback_token(key or ""),
            locale,
            style or "informal",
            str(page),
        )
    elif action == "l":
        parts = ("sm", action, locale, str(page))
    elif action == "p":
        parts = ("sm", action, str(page), locale)
    else:
        raise ValueError("Неизвестное действие системного сообщения.")
    return ":".join(parts)


def system_message_callback_token(key: str) -> str:
    try:
        return SYSTEM_MESSAGE_CALLBACK_TOKENS[key]
    except KeyError as exc:
        raise ValueError("Неизвестный ключ системного сообщения.") from exc


def parse_system_message_callback(data: str) -> dict[str, Any] | None:
    parts = (data or "").split(":")
    if len(parts) < 3 or parts[0] != "sm":
        return None
    action = parts[1]
    if action == "l" and len(parts) == 4:
        return {"action": action, "locale": parts[2], "page": parts[3]}
    if action == "p" and len(parts) == 4:
        return {"action": action, "page": parts[2], "locale": parts[3]}
    if action in {"d", "m"} and len(parts) == 5:
        key = SYSTEM_MESSAGE_KEYS_BY_CALLBACK_TOKEN.get(parts[2])
        return {"action": action, "key": key, "locale": parts[3], "page": parts[4]} if key else None
    if action in {"v", "e", "r", "s", "c"} and len(parts) == 6:
        key = SYSTEM_MESSAGE_KEYS_BY_CALLBACK_TOKEN.get(parts[2])
        return (
            {"action": action, "key": key, "locale": parts[3], "style": parts[4], "page": parts[5]}
            if key
            else None
        )
    return None


def limit_callback_answer(value: Any) -> str:
    text = str(value or "")
    return text[:CALLBACK_ANSWER_LIMIT]


def escape_text_with_limit(value: Any, limit: int) -> str:
    source = str(value or "")
    escaped = []
    used = 0
    truncated = False
    for character in source:
        escaped_character = html.escape(character)
        if used + len(escaped_character) > limit:
            truncated = True
            break
        escaped.append(escaped_character)
        used += len(escaped_character)
    if truncated and used < limit:
        escaped.append("…")
    return "".join(escaped)


def render_system_message_detail(
    definition: SystemMessageDefinition,
    key: str,
    locale: str,
    variants: dict[str, str],
    mode_note: str = "",
) -> str:
    header = (
        "💬 <b>Сообщение бота</b>\n\n"
        f"<b>{html.escape(definition.title)}</b>\n"
        f"Когда появляется: {html.escape(definition.description)}\n"
        f"Язык: <b>{html.escape(locale)}</b>"
        + (f"\nРежим: <b>{html.escape(mode_note)}</b>" if mode_note else "")
    )
    informal_prefix = f"{header}\n\n<b>Ты:</b>\n"
    formal_prefix = "\n\n<b>Вы:</b>\n"
    available = max(0, TELEGRAM_TEXT_LIMIT - len(informal_prefix) - len(formal_prefix))
    informal_limit = available // 2
    formal_limit = available - informal_limit
    return (
        informal_prefix
        + escape_text_with_limit(variants.get("informal", ""), informal_limit)
        + formal_prefix
        + escape_text_with_limit(variants.get("formal", ""), formal_limit)
    )


def render_system_message_preview(
    key: str,
    style: Any,
    text: str,
) -> str:
    style_label = "Вы" if normalize_addressing_style(style) == "formal" else "Ты"
    definition = SYSTEM_MESSAGE_DEFINITIONS.get(key)
    title = definition.title if definition else "Сообщение бота"
    header = f"👁 <b>Предпросмотр · {html.escape(title)} · {style_label}</b>\n\n"
    return header + escape_text_with_limit(text, max(0, TELEGRAM_TEXT_LIMIT - len(header)))


def render_system_message_editor(
    key: str,
    style: Any,
    text: str,
    locale: str = "Русский",
    mode_note: str = "",
) -> str:
    definition = SYSTEM_MESSAGE_DEFINITIONS.get(key)
    title = definition.title if definition else "Сообщение бота"
    note = f"\n\n{html.escape(mode_note)}" if mode_note else ""
    prefix = (
        f"✏️ <b>{html.escape(title)}</b>\n"
        f"Язык: <b>{html.escape(locale)}</b>\n"
        f"Обращение: <b>{addressing_style_label(style)}</b>\n\n"
        "Текущий текст:\n"
    )
    suffix = note + "\n\nОтправьте новый текст одним сообщением. Затем нажмите «Сохранить» или «Отмена»."
    return prefix + escape_text_with_limit(text, max(0, TELEGRAM_TEXT_LIMIT - len(prefix) - len(suffix))) + suffix


def render_system_message_draft(text: str) -> str:
    prefix = "Черновик сохранён в форме. Проверьте текст и нажмите «Сохранить» или «Отмена».\n\n"
    return prefix + escape_text_with_limit(text, max(0, TELEGRAM_TEXT_LIMIT - len(prefix)))


def normalize_addressing_style(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ADDRESSING_STYLES else DEFAULT_ADDRESSING_STYLE


def normalize_bot_scope(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in BOT_SCOPES else DEFAULT_BOT_SCOPE


def addressing_style_label(style: Any) -> str:
    return "Вы" if normalize_addressing_style(style) == "formal" else "Ты"


def addressing_style_instruction(style: Any, locale: Any = "ru") -> str:
    normalized_style = normalize_addressing_style(style)
    normalized_locale = normalize_locale(locale) or "ru"
    instructions = {
        "ru": {
            "informal": "ОБРАЩЕНИЕ: обращайся к пользователю на «ты», используя неформальный, тёплый стиль.",
            "formal": "ОБРАЩЕНИЕ: обращайся к пользователю на «Вы», используя вежливый, уважительный стиль.",
        },
        "en": {
            "informal": "ADDRESSING STYLE: use a friendly, informal tone when addressing the user.",
            "formal": "ADDRESSING STYLE: use a polite, formal tone when addressing the user.",
        },
        "pt": {
            "informal": "FORMA DE TRATAMENTO: use um tom informal e próximo ao falar com a pessoa usuária.",
            "formal": "FORMA DE TRATAMENTO: use um tom educado e formal ao falar com a pessoa usuária.",
        },
    }
    return instructions[normalized_locale][normalized_style]


async def resolve_addressing_style(session, bot_scope: Any = DEFAULT_BOT_SCOPE) -> str:
    normalized_scope = normalize_bot_scope(bot_scope)
    try:
        scoped_config = await session.get(BotAddressingConfig, normalized_scope)
    except (AttributeError, OperationalError):
        scoped_config = None
    if scoped_config is not None:
        return normalize_addressing_style(getattr(scoped_config, "addressing_style", DEFAULT_ADDRESSING_STYLE))
    try:
        general_config = await session.get(BotGeneralConfig, 1)
    except (AttributeError, OperationalError):
        general_config = None
    return normalize_addressing_style(
        getattr(general_config, "addressing_style", DEFAULT_ADDRESSING_STYLE)
    )


async def save_addressing_style(session, style: Any, bot_scope: Any = DEFAULT_BOT_SCOPE) -> str:
    normalized_scope = normalize_bot_scope(bot_scope)
    normalized_style = normalize_addressing_style(style)
    scoped_config = await session.get(BotAddressingConfig, normalized_scope)
    if scoped_config is None:
        session.add(BotAddressingConfig(
            bot_scope=normalized_scope,
            addressing_style=normalized_style,
        ))
    else:
        scoped_config.addressing_style = normalized_style
    if normalized_scope == DEFAULT_BOT_SCOPE:
        general_config = await session.get(BotGeneralConfig, 1)
        if general_config is not None:
            general_config.addressing_style = normalized_style
    return normalized_style


def system_message_definition(key: str) -> SystemMessageDefinition:
    try:
        return SYSTEM_MESSAGE_DEFINITIONS[key]
    except KeyError as exc:
        raise ValueError("Неизвестный ключ системного сообщения.") from exc


def _default_text(key: str, locale: str, style: str) -> str | None:
    definition = SYSTEM_MESSAGE_DEFINITIONS.get(key)
    if definition is None:
        return None
    locale_defaults = definition.defaults.get(locale) or definition.defaults.get("ru") or {}
    return locale_defaults.get(style) or locale_defaults.get(DEFAULT_ADDRESSING_STYLE)


async def _legacy_translation(session, key: str, locale: str) -> str | None:
    if locale == "ru":
        return None
    from translation_registry import STATIC_TRANSLATION_SOURCES
    from translation_service import source_hash

    source_item = STATIC_TRANSLATION_SOURCES.get(key)
    from database import BotTranslation

    try:
        translation = await session.scalar(
            select(BotTranslation).where(
                BotTranslation.locale == locale,
                BotTranslation.translation_key == key,
            )
        )
    except (AttributeError, OperationalError):
        return None
    translation_text = getattr(translation, "text", None) if translation is not None else None
    if not isinstance(translation_text, str) or not translation_text.strip():
        return None
    if source_item is None or getattr(translation, "source_hash", None) == source_hash(source_item[0]):
        return translation_text
    return None


async def resolve_system_text(
    session,
    key: str,
    locale: Any = "ru",
    addressing_style: Any = None,
    *,
    bot_scope: Any = DEFAULT_BOT_SCOPE,
    fallback: str | None = None,
    **variables: Any,
) -> str:
    definition = system_message_definition(key)
    normalized_locale = normalize_locale(locale) or "ru"
    normalized_scope = normalize_bot_scope(bot_scope)
    if addressing_style is None:
        addressing_style = await resolve_addressing_style(session, normalized_scope)
    normalized_style = normalize_addressing_style(addressing_style)
    try:
        row = await session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.bot_scope == normalized_scope,
                BotSystemMessage.translation_key == definition.key,
                BotSystemMessage.locale == normalized_locale,
                BotSystemMessage.addressing_style == normalized_style,
            )
        )
    except (AttributeError, OperationalError):
        row = None
    text = getattr(row, "text", None) if row is not None else None
    if not isinstance(text, str):
        text = None
    if not text:
        text = await _legacy_translation(session, definition.key, normalized_locale)
    if not text:
        text = _default_text(definition.key, normalized_locale, normalized_style)
    if not text:
        text = fallback or ""
    if variables:
        try:
            text = text.format(**variables)
        except (KeyError, IndexError, ValueError):
            pass
    return text


async def save_system_message(
    session,
    key: str,
    locale: Any,
    addressing_style: Any,
    text: str,
    *,
    bot_scope: Any = DEFAULT_BOT_SCOPE,
) -> None:
    definition = system_message_definition(key)
    normalized_locale = normalize_locale(locale) or "ru"
    normalized_style = normalize_addressing_style(addressing_style)
    normalized_scope = normalize_bot_scope(bot_scope)
    value = str(text or "").strip()
    if not value:
        raise ValueError("Текст не может быть пустым.")
    if len(value) > 4096:
        raise ValueError("Текст не должен превышать 4096 символов.")
    row = await session.scalar(
        select(BotSystemMessage).where(
            BotSystemMessage.bot_scope == normalized_scope,
            BotSystemMessage.translation_key == definition.key,
            BotSystemMessage.locale == normalized_locale,
            BotSystemMessage.addressing_style == normalized_style,
        )
    )
    if row is None:
        session.add(BotSystemMessage(
            bot_scope=normalized_scope,
            translation_key=definition.key,
            locale=normalized_locale,
            addressing_style=normalized_style,
            text=value,
        ))
    else:
        row.text = value


async def reset_system_message(
    session,
    key: str,
    locale: Any,
    addressing_style: Any,
    *,
    bot_scope: Any = DEFAULT_BOT_SCOPE,
) -> None:
    definition = system_message_definition(key)
    normalized_locale = normalize_locale(locale) or "ru"
    normalized_style = normalize_addressing_style(addressing_style)
    normalized_scope = normalize_bot_scope(bot_scope)
    await session.execute(
        delete(BotSystemMessage).where(
            BotSystemMessage.bot_scope == normalized_scope,
            BotSystemMessage.translation_key == definition.key,
            BotSystemMessage.locale == normalized_locale,
            BotSystemMessage.addressing_style == normalized_style,
        )
    )


async def get_system_message_variants(
    session,
    key: str,
    locale: Any,
    *,
    bot_scope: Any = DEFAULT_BOT_SCOPE,
) -> dict[str, str]:
    normalized_locale = normalize_locale(locale) or "ru"
    return {
        style: await resolve_system_text(
            session,
            key,
            normalized_locale,
            style,
            bot_scope=bot_scope,
        )
        for style in ADDRESSING_STYLES
    }
