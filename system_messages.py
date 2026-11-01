from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import OperationalError

from database import (
    ADDRESSING_STYLES,
    DEFAULT_ADDRESSING_STYLE,
    BotGeneralConfig,
    BotSystemMessage,
)
from translation_service import normalize_locale


@dataclass(frozen=True)
class SystemMessageDefinition:
    key: str
    title: str
    defaults: dict[str, dict[str, str]]


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


SYSTEM_MESSAGE_DEFINITIONS = {item.key: item for item in SYSTEM_MESSAGE_CATALOG}
SYSTEM_MESSAGE_PAGE_SIZE = 8


def normalize_addressing_style(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ADDRESSING_STYLES else DEFAULT_ADDRESSING_STYLE


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
    fallback: str | None = None,
    **variables: Any,
) -> str:
    definition = system_message_definition(key)
    normalized_locale = normalize_locale(locale) or "ru"
    if addressing_style is None:
        try:
            config = await session.get(BotGeneralConfig, 1)
        except (AttributeError, OperationalError):
            config = None
        addressing_style = getattr(config, "addressing_style", DEFAULT_ADDRESSING_STYLE)
    normalized_style = normalize_addressing_style(addressing_style)
    try:
        row = await session.scalar(
            select(BotSystemMessage).where(
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


async def save_system_message(session, key: str, locale: Any, addressing_style: Any, text: str) -> None:
    definition = system_message_definition(key)
    normalized_locale = normalize_locale(locale) or "ru"
    normalized_style = normalize_addressing_style(addressing_style)
    value = str(text or "").strip()
    if not value:
        raise ValueError("Текст не может быть пустым.")
    if len(value) > 4096:
        raise ValueError("Текст не должен превышать 4096 символов.")
    row = await session.scalar(
        select(BotSystemMessage).where(
            BotSystemMessage.translation_key == definition.key,
            BotSystemMessage.locale == normalized_locale,
            BotSystemMessage.addressing_style == normalized_style,
        )
    )
    if row is None:
        session.add(BotSystemMessage(
            translation_key=definition.key,
            locale=normalized_locale,
            addressing_style=normalized_style,
            text=value,
        ))
    else:
        row.text = value


async def reset_system_message(session, key: str, locale: Any, addressing_style: Any) -> None:
    definition = system_message_definition(key)
    normalized_locale = normalize_locale(locale) or "ru"
    normalized_style = normalize_addressing_style(addressing_style)
    await session.execute(
        delete(BotSystemMessage).where(
            BotSystemMessage.translation_key == definition.key,
            BotSystemMessage.locale == normalized_locale,
            BotSystemMessage.addressing_style == normalized_style,
        )
    )


async def get_system_message_variants(session, key: str, locale: Any) -> dict[str, str]:
    normalized_locale = normalize_locale(locale) or "ru"
    return {
        style: await resolve_system_text(session, key, normalized_locale, style)
        for style in ADDRESSING_STYLES
    }
