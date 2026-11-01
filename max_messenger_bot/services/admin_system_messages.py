from __future__ import annotations

from database import BotGeneralConfig
from system_messages import (
    ADDRESSING_STYLES,
    SYSTEM_MESSAGE_CATALOG,
    SYSTEM_MESSAGE_DEFINITIONS,
    SYSTEM_MESSAGE_PAGE_SIZE,
    addressing_style_label,
    get_system_message_variants,
    render_system_message_detail,
    render_system_message_draft,
    render_system_message_editor,
    render_system_message_preview,
    resolve_addressing_style,
    reset_system_message,
    save_addressing_style,
    save_system_message,
    resolve_system_text,
    system_message_callback_data,
)

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..legacy import async_session_maker
from ..storage import StateStore


LOCALES = ("ru", "en", "pt")


def _locale_label(locale: str) -> str:
    return {"ru": "Русский", "en": "English", "pt": "Português"}.get(locale, "Русский")


def _valid_page(value: str | int | None) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _valid_locale(value: str | None) -> str:
    return value if value in LOCALES else "ru"


def _valid_style(value: str | None) -> str:
    return value if value in ADDRESSING_STYLES else "informal"


def _definition(key: str):
    return SYSTEM_MESSAGE_DEFINITIONS.get(key)


def _edit_context_matches(snapshot, key: str, locale: str, style: str, page: int) -> bool:
    if snapshot is None or snapshot.state != "max_system_message_edit":
        return False
    data = snapshot.data
    try:
        stored_page = int(data.get("page"))
    except (TypeError, ValueError):
        return False
    return (
        data.get("key") == key
        and data.get("locale") == locale
        and data.get("style") == style
        and stored_page == page
    )


async def show_general_settings(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(BotGeneralConfig, 1)
        if config is None:
            config = BotGeneralConfig(id=1)
            session.add(config)
            await session.commit()
        style = await resolve_addressing_style(session, bot_scope="max")
    await client.send_message(
        chat_id=chat_id,
        text=(
            "⚙️ <b>Общие настройки</b>\n\n"
            f"Обращение: <b>{addressing_style_label(style)}</b>\n"
            "Стиль настраивается только для этого бота."
        ),
        attachments=inline_keyboard([
            [
                callback_button("✅ Ты" if style == "informal" else "Ты", "admin_general_set_addressing_informal"),
                callback_button("✅ Вы" if style == "formal" else "Вы", "admin_general_set_addressing_formal"),
            ],
            [callback_button("💬 Системные сообщения", "admin_system_messages")],
            [callback_button("◀️ Назад", "admin_panel")],
        ]),
    )


async def set_addressing_style(client: MaxApiClient, chat_id: int, style: str) -> None:
    normalized_style = _valid_style(style)
    async with async_session_maker() as session:
        config = await session.get(BotGeneralConfig, 1)
        if config is None:
            config = BotGeneralConfig(id=1)
            session.add(config)
        await save_addressing_style(session, normalized_style, bot_scope="max")
        await session.commit()
    await show_general_settings(client, chat_id)


async def show_system_messages(
    client: MaxApiClient,
    chat_id: int,
    locale: str = "ru",
    page: int = 0,
) -> None:
    locale = _valid_locale(locale)
    total_pages = max(1, (len(SYSTEM_MESSAGE_CATALOG) + SYSTEM_MESSAGE_PAGE_SIZE - 1) // SYSTEM_MESSAGE_PAGE_SIZE)
    page = min(_valid_page(page), total_pages - 1)
    start = page * SYSTEM_MESSAGE_PAGE_SIZE
    definitions = SYSTEM_MESSAGE_CATALOG[start:start + SYSTEM_MESSAGE_PAGE_SIZE]
    rows = [
        [callback_button(
            f"{item.title} · {item.key}",
            system_message_callback_data("d", item.key, locale, page=page),
        )]
        for item in definitions
    ]
    rows.append([callback_button(
        f"✅ {_locale_label(locale)}",
        system_message_callback_data("l", locale=locale, page=page),
    )])
    rows.append([
        callback_button("Русский", system_message_callback_data("l", locale="ru", page=page)),
        callback_button("English", system_message_callback_data("l", locale="en", page=page)),
        callback_button("Português", system_message_callback_data("l", locale="pt", page=page)),
    ])
    navigation = []
    if page > 0:
        navigation.append(callback_button("◀️", system_message_callback_data("p", locale=locale, page=page - 1)))
    if page < total_pages - 1:
        navigation.append(callback_button("▶️", system_message_callback_data("p", locale=locale, page=page + 1)))
    if navigation:
        rows.append(navigation)
    rows.append([callback_button("⚙️ Общие настройки", "admin_general_settings")])
    await client.send_message(
        chat_id=chat_id,
        text=(
            "💬 <b>Системные сообщения</b>\n\n"
            f"Язык: <b>{_locale_label(locale)}</b> · страница {page + 1}/{total_pages}\n"
            "Варианты «Ты» и «Вы» хранятся независимо."
        ),
        attachments=inline_keyboard(rows),
    )


async def show_system_message(
    client: MaxApiClient,
    chat_id: int,
    key: str,
    locale: str = "ru",
    page: int = 0,
) -> None:
    locale = _valid_locale(locale)
    definition = _definition(key)
    if definition is None:
        await client.send_message(chat_id=chat_id, text="Системное сообщение не найдено.")
        return
    async with async_session_maker() as session:
        variants = await get_system_message_variants(session, key, locale, bot_scope="max")
    rows = []
    for style in ADDRESSING_STYLES:
        rows.append([
            callback_button(
                f"👁 Просмотр · {addressing_style_label(style)}",
                system_message_callback_data("v", key, locale, style, _valid_page(page)),
            ),
            callback_button(
                f"✏️ Изменить · {addressing_style_label(style)}",
                system_message_callback_data("e", key, locale, style, _valid_page(page)),
            ),
        ])
        rows.append([
            callback_button(
                f"↩️ Сбросить · {addressing_style_label(style)}",
                system_message_callback_data("r", key, locale, style, _valid_page(page)),
            )
        ])
    rows.append([callback_button(
        "◀️ К списку",
        system_message_callback_data("p", locale=locale, page=_valid_page(page)),
    )])
    message_text = render_system_message_detail(definition, key, _locale_label(locale), variants)
    await client.send_message(
        chat_id=chat_id,
        text=message_text,
        attachments=inline_keyboard(rows),
    )


async def preview_system_message(
    client: MaxApiClient,
    chat_id: int,
    key: str,
    locale: str,
    style: str,
    page: int,
) -> None:
    if _definition(key) is None:
        await client.send_message(chat_id=chat_id, text="Системное сообщение не найдено.")
        return
    locale = _valid_locale(locale)
    style = _valid_style(style)
    async with async_session_maker() as session:
        text = await resolve_system_text(session, key, locale, style, bot_scope="max")
    await client.send_message(
        chat_id=chat_id,
        text=render_system_message_preview(key, style, text),
        attachments=inline_keyboard([
            [callback_button(
                "◀️ К сообщению",
                system_message_callback_data("d", key, locale, page=_valid_page(page)),
            )],
        ]),
    )


async def start_edit_system_message(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    key: str,
    locale: str,
    style: str,
    page: int,
) -> None:
    if _definition(key) is None:
        await client.send_message(chat_id=chat_id, text="Системное сообщение не найдено.")
        return
    locale = _valid_locale(locale)
    style = _valid_style(style)
    await states.set(
        user_id,
        chat_id,
        "max_system_message_edit",
        {"key": key, "locale": locale, "style": style, "page": _valid_page(page)},
    )
    async with async_session_maker() as session:
        current_text = await resolve_system_text(session, key, locale, style, bot_scope="max")
    await client.send_message(
        chat_id=chat_id,
        text=render_system_message_editor(key, style, current_text),
        attachments=inline_keyboard([
            [callback_button(
                "Отмена",
                system_message_callback_data("c", key, locale, style, _valid_page(page)),
            )],
        ]),
    )


async def receive_system_message(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    value = (text or "").strip()
    if not value:
        await client.send_message(chat_id=chat_id, text="Текст не может быть пустым.")
        return
    if len(value) > 4096:
        await client.send_message(chat_id=chat_id, text="Текст не должен превышать 4096 символов.")
        return
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    await states.update(user_id, draft=value)
    key = data.get("key", "")
    locale = _valid_locale(data.get("locale"))
    style = _valid_style(data.get("style"))
    page = _valid_page(data.get("page"))
    await client.send_message(
        chat_id=chat_id,
        text=render_system_message_draft(value),
        attachments=inline_keyboard([
            [callback_button("Сохранить", system_message_callback_data("s", key, locale, style, page))],
            [callback_button("Отмена", system_message_callback_data("c", key, locale, style, page))],
        ]),
    )


async def save_system_message_draft(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    key: str,
    locale: str,
    style: str,
    page: int,
) -> None:
    snapshot = await states.get(user_id)
    normalized_locale = _valid_locale(locale)
    normalized_style = _valid_style(style)
    normalized_page = _valid_page(page)
    if not _edit_context_matches(snapshot, key, normalized_locale, normalized_style, normalized_page):
        await client.send_message(chat_id=chat_id, text="Изменение устарело. Откройте сообщение заново.")
        return
    draft = snapshot.data.get("draft") if snapshot else None
    if not isinstance(draft, str) or not draft.strip():
        await client.send_message(chat_id=chat_id, text="Сначала отправьте новый текст.")
        return
    try:
        async with async_session_maker() as session:
            await save_system_message(
                session,
                key,
                normalized_locale,
                normalized_style,
                draft,
                bot_scope="max",
            )
            await session.commit()
    except ValueError as exc:
        await client.send_message(chat_id=chat_id, text=str(exc))
        return
    await states.clear(user_id)
    await show_system_message(client, chat_id, key, normalized_locale, normalized_page)


async def cancel_system_message_edit(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    key: str,
    locale: str,
    style: str,
    page: int,
) -> None:
    snapshot = await states.get(user_id)
    normalized_locale = _valid_locale(locale)
    normalized_style = _valid_style(style)
    normalized_page = _valid_page(page)
    if not _edit_context_matches(snapshot, key, normalized_locale, normalized_style, normalized_page):
        await client.send_message(chat_id=chat_id, text="Изменение устарело. Откройте сообщение заново.")
        return
    await states.clear(user_id)
    await show_system_message(client, chat_id, key, normalized_locale, normalized_page)


async def reset_system_message_text(
    client: MaxApiClient,
    chat_id: int,
    key: str,
    locale: str,
    style: str,
    page: int,
) -> None:
    try:
        async with async_session_maker() as session:
            await reset_system_message(session, key, locale, style, bot_scope="max")
            await session.commit()
    except ValueError as exc:
        await client.send_message(chat_id=chat_id, text=str(exc))
        return
    await show_system_message(client, chat_id, key, locale, page)
