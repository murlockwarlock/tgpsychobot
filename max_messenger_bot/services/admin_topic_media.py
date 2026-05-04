from __future__ import annotations

from sqlalchemy import func, select

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..legacy import (
    MediaCollection,
    MediaLibrary,
    Topic,
    TopicMediaDeck,
    async_session_maker,
    media_collection_items,
)
from ..storage import StateStore

MEDIA_PAGE_SIZE = 10


# ── Internal keyboard builders ────────────────────────────────────────────────

def _topic_media_list_keyboard(
    topic_id: int,
    media_list: list,
    page: int,
    total_pages: int,
) -> list[dict]:
    rows: list[list[dict]] = []
    for media in media_list:
        label = media.file_name or f"#{media.id}"
        type_icon = "🎵" if media.media_type == "audio" else "🖼️"
        rows.append([callback_button(f"{type_icon} {label}", f"admin_media_view_{media.id}")])
    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"admin_topic_media_{topic_id}_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"admin_topic_media_{topic_id}_{page + 1}"))
    if nav_row:
        rows.append(nav_row)
    rows.append([callback_button("➕ Добавить файл", f"admin_media_add_{topic_id}")])
    rows.append([callback_button("⬅️ К теме", f"admin_topic_editor_{topic_id}")])
    return inline_keyboard(rows)


def _media_detail_keyboard(media_id: int, topic_id: int) -> list[dict]:
    return inline_keyboard(
        [
            [
                callback_button("✏️ Имя", f"admin_media_editname_{media_id}"),
                callback_button("✏️ Категорию", f"admin_media_editcat_{media_id}"),
            ],
            [
                callback_button("✏️ Описание", f"admin_media_editdesc_{media_id}"),
                callback_button("🔄 Файл", f"admin_media_editfile_{media_id}"),
            ],
            [callback_button("🗑️ Удалить", f"admin_media_delete_{media_id}")],
            [callback_button("⬅️ Назад к списку", f"admin_topic_media_{topic_id}_0")],
        ]
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _build_list_text(topic: Topic, total_count: int, topic_id: int) -> str:
    async with async_session_maker() as session:
        cat_res = await session.execute(
            select(MediaLibrary.category, MediaLibrary.file_name).where(
                MediaLibrary.topic_id == topic_id,
                MediaLibrary.category.isnot(None),
                MediaLibrary.category != "",
            )
        )
        cat_rows = cat_res.all()

    categories: set[str] = set()
    back_categories: set[str] = set()
    for cat, fname in cat_rows:
        categories.add(cat)
        if fname == "_back":
            back_categories.add(cat)

    cats_info = ""
    if categories:
        lines = []
        for cat in sorted(categories):
            marker = "🃏" if cat in back_categories else "⚠️ нет рубашки"
            lines.append(f"  <code>{cat}</code> — {marker}")
        cats_info = "\n<b>Категории:</b>\n" + "\n".join(lines) + "\n"

    return (
        f"📁 Медиа-библиотека темы: <b>{topic.name}</b>\n"
        f"Файлов: {total_count}"
        f"{cats_info}\n"
        f"<b>Теги для AI:</b>\n"
        f"<code>[RANDOM_IMG: категория]</code> — случайная карта\n"
        f"<code>[RANDOM_IMG: категория | N]</code> — N случайных карт сразу\n"
        f"<code>[CHOICE_IMG: категория | N]</code> — выбор из N (лицом)\n"
        f"<code>[CHOICE_IMG: категория | N | R]</code> — расклад из R карт, выбор из N\n"
        f"<code>[CHOICE_IMG_HIDDEN: категория | N]</code> — выбор из N (рубашкой)\n"
        f"<code>[CHOICE_IMG_HIDDEN: категория | N | R]</code> — расклад из R карт вслепую\n"
        f"<code>[SHOW_IMG: имя_файла]</code> — конкретная карта\n"
        f"<code>[SEND_AUDIO: имя]</code> — аудиофайл\n\n"
        f"🃏 Для скрытого выбора добавьте файл с именем <code>_back</code> в нужную категорию."
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def show_list(
    client: MaxApiClient,
    chat_id: int,
    topic_id: int,
    page: int = 0,
) -> None:
    async with async_session_maker() as session:
        total_count = await session.scalar(
            select(func.count()).select_from(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
        )
        stmt = (
            select(MediaLibrary)
            .where(MediaLibrary.topic_id == topic_id)
            .order_by(MediaLibrary.id)
            .offset(page * MEDIA_PAGE_SIZE)
            .limit(MEDIA_PAGE_SIZE)
        )
        media_list = (await session.execute(stmt)).scalars().all()
        topic = await session.get(Topic, topic_id)

    if not topic:
        await client.send_message(chat_id=chat_id, text="Тема не найдена.")
        return

    total_pages = max(1, (total_count + MEDIA_PAGE_SIZE - 1) // MEDIA_PAGE_SIZE)
    text = await _build_list_text(topic, total_count, topic_id)
    kb = _topic_media_list_keyboard(topic_id, media_list, page, total_pages)
    await client.send_message(chat_id=chat_id, text=text, attachments=kb)


async def show_media_detail(
    client: MaxApiClient,
    chat_id: int,
    media_id: int,
) -> None:
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if not media:
            await client.send_message(chat_id=chat_id, text="Файл не найден.")
            return
        coll_res = await session.execute(
            select(MediaCollection.name)
            .join(media_collection_items, media_collection_items.c.collection_id == MediaCollection.id)
            .where(media_collection_items.c.media_id == media_id)
        )
        coll_names = [r[0] for r in coll_res.all()]

    role_hint = ""
    if media.file_name == "_back":
        role_hint = f"\n🃏 <b>Рубашка</b> для категории <code>{media.category}</code>"

    colls_text = ", ".join(coll_names) if coll_names else "нет"
    text = (
        f"<b>📄 Данные файла:</b>\n"
        f"ID: <code>{media.id}</code>\n"
        f"Имя для AI: <code>{media.file_name}</code>\n"
        f"Тип: {media.media_type}\n"
        f"Категория: {media.category or 'Не задана'}\n"
        f"Коллекции: {colls_text}\n"
        f"Описание: {media.description or 'Нет'}"
        f"{role_hint}"
    )
    kb = _media_detail_keyboard(media.id, media.topic_id)
    await client.send_message(chat_id=chat_id, text=text, attachments=kb)


# ── Edit name ─────────────────────────────────────────────────────────────────

async def start_edit_name(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_media_edit_name", {"media_id": media_id})
    await client.send_message(
        chat_id=chat_id,
        text="Введи новое <b>техническое имя</b> для файла (на английском, без пробелов):",
    )


async def save_edit_name(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    media_id = snapshot.data.get("media_id") if snapshot else None
    if not media_id:
        await client.send_message(chat_id=chat_id, text="Состояние потеряно.")
        return
    new_name = text.strip().lower().replace(" ", "_")
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.file_name = new_name
            await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Имя изменено на <code>{new_name}</code>.")
    await show_media_detail(client, chat_id, media_id)


# ── Edit category ─────────────────────────────────────────────────────────────

async def start_edit_category(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_media_edit_category", {"media_id": media_id})
    await client.send_message(
        chat_id=chat_id,
        text="Введи новую <b>категорию</b> (например: <code>tarot</code>, <code>mak</code>, <code>oracle</code>):",
    )


async def save_edit_category(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    media_id = snapshot.data.get("media_id") if snapshot else None
    if not media_id:
        await client.send_message(chat_id=chat_id, text="Состояние потеряно.")
        return
    new_category = text.strip().lower().replace(" ", "_")
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.category = new_category
            await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Категория изменена на <code>{new_category}</code>.")
    await show_media_detail(client, chat_id, media_id)


# ── Edit description ──────────────────────────────────────────────────────────

async def start_edit_description(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_media_edit_desc", {"media_id": media_id})
    await client.send_message(chat_id=chat_id, text="Введи новое <b>описание</b> для файла:")


async def save_edit_description(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    media_id = snapshot.data.get("media_id") if snapshot else None
    if not media_id:
        await client.send_message(chat_id=chat_id, text="Состояние потеряно.")
        return
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.description = text.strip()
            await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Описание обновлено.")
    await show_media_detail(client, chat_id, media_id)


# ── Replace file token ────────────────────────────────────────────────────────

async def start_edit_file(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_media_edit_file", {"media_id": media_id})
    await client.send_message(
        chat_id=chat_id,
        text=(
            "Отправь новый файл (фото или аудио) для замены, "
            "либо вставь токен файла напрямую."
        ),
    )


async def save_edit_file(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    token: str,
    media_type: str,
) -> None:
    snapshot = await states.get(user_id)
    media_id = snapshot.data.get("media_id") if snapshot else None
    if not media_id:
        await client.send_message(chat_id=chat_id, text="Состояние потеряно.")
        return
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if media:
            media.file_id = token
            media.media_type = media_type
            await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Файл заменён.")
    await show_media_detail(client, chat_id, media_id)


# ── Add new media — multi-step flow ───────────────────────────────────────────

async def start_add_media(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    topic_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_media_add_file", {"topic_id": topic_id})
    await client.send_message(
        chat_id=chat_id,
        text=(
            "Отправьте файл (фото, аудио) или вставьте токен файла для добавления в медиатеку.\n"
            "Если вставляете токен текстом — просто отправьте его следующим сообщением."
        ),
    )


_TYPE_MAP = {
    "photo": "photo",
    "image": "photo",
    "фото": "photo",
    "audio": "audio",
    "аудио": "audio",
}


async def receive_add_file(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str | None = None,
    media_token: str | None = None,
    media_type: str | None = None,
) -> None:
    """Step 1 — receive token + type, then ask for a technical name."""
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    topic_id = data.get("topic_id")

    if media_token and media_type:
        # Received as an attachment with known type
        token = media_token
        m_type = media_type
    elif text:
        # Admin pasted a raw token — need to ask for type
        await states.set(
            user_id, chat_id,
            "admin_media_add_type",
            {"topic_id": topic_id, "token": text.strip()},
        )
        await client.send_message(
            chat_id=chat_id,
            text=(
                "Токен принят.\n"
                "Укажи тип файла: <b>photo</b> (фото/изображение) или <b>audio</b> (аудио)."
            ),
        )
        return
    else:
        await client.send_message(chat_id=chat_id, text="Пожалуйста, отправь файл или вставь токен.")
        return

    await states.set(
        user_id, chat_id,
        "admin_media_add_name",
        {"topic_id": topic_id, "token": token, "media_type": m_type},
    )
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ <b>Файл получен как {m_type}!</b>\n\n"
            "Придумай короткое <b>техническое имя</b> на английском "
            "(например: <code>morning_meditation</code>, <code>card_death</code>).\n\n"
            "⚠️ <b>ВАЖНО:</b> Это имя используется в системном промпте, "
            "например <code>[SEND_AUDIO: имя]</code>."
        ),
    )


async def resolve_add_type(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    """Step 1b — admin specified the type for a text-token upload."""
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    m_type = _TYPE_MAP.get(text.strip().lower())
    if not m_type:
        await client.send_message(
            chat_id=chat_id,
            text="Введи <b>photo</b> или <b>audio</b>.",
        )
        return
    token = data.get("token", "")
    topic_id = data.get("topic_id")
    await states.set(
        user_id, chat_id,
        "admin_media_add_name",
        {"topic_id": topic_id, "token": token, "media_type": m_type},
    )
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Тип <b>{m_type}</b> принят.\n\n"
            "Придумай короткое <b>техническое имя</b> на английском "
            "(например: <code>morning_meditation</code>, <code>card_death</code>).\n\n"
            "⚠️ <b>ВАЖНО:</b> Это имя используется в системном промпте."
        ),
    )


async def save_add_name(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    """Step 2 — save technical name; branch by type."""
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    tech_name = text.strip().lower().replace(" ", "_")
    await states.set(
        user_id, chat_id,
        "admin_media_add_name",  # keep current state momentarily while updating
        {**data, "file_name": tech_name},
    )
    m_type = data.get("media_type", "")
    if m_type == "photo":
        await states.set(
            user_id, chat_id,
            "admin_media_add_category",
            {**data, "file_name": tech_name},
        )
        await client.send_message(
            chat_id=chat_id,
            text=(
                f"👌 Имя <code>{tech_name}</code> принято.\n\n"
                "Введи <b>категорию</b> для изображения "
                "(например: <code>tarot</code>, <code>mak</code>, <code>oracle</code>).\n"
                "Категория используется для группировки карт одной колоды."
            ),
        )
    else:
        await states.set(
            user_id, chat_id,
            "admin_media_add_desc",
            {**data, "file_name": tech_name},
        )
        await client.send_message(
            chat_id=chat_id,
            text=(
                f"👌 Имя <code>{tech_name}</code> принято.\n\n"
                "Введи описание файла.\n"
                "Для аудио — опиши, в какой момент AI должен предложить эту практику."
            ),
        )


async def save_add_category(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    """Step 3 (photos only) — save category, then ask for description."""
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    category = text.strip().lower().replace(" ", "_")
    await states.set(
        user_id, chat_id,
        "admin_media_add_desc",
        {**data, "category": category},
    )
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"👌 Категория <code>{category}</code> принята.\n\n"
            "Введи описание карты — трактовку, которую AI учтёт при интерпретации."
        ),
    )


async def save_add_description(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    """Final step — persist MediaLibrary entry and show success + updated list."""
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}

    topic_id = data.get("topic_id")
    m_type = data.get("media_type", "photo")
    m_name = data.get("file_name", "")
    token = data.get("token", "")
    category = data.get("category", "")
    description = text.strip()

    async with async_session_maker() as session:
        new_media = MediaLibrary(
            topic_id=topic_id,
            media_type=m_type,
            file_id=token,
            file_name=m_name,
            category=category,
            description=description,
        )
        session.add(new_media)

        if category and topic_id:
            existing = (
                await session.execute(
                    select(TopicMediaDeck).where(
                        TopicMediaDeck.topic_id == topic_id,
                        TopicMediaDeck.deck_name == category,
                    )
                )
            ).scalar_one_or_none()
            if not existing:
                session.add(TopicMediaDeck(topic_id=topic_id, deck_name=category))

        await session.commit()

        total_count = await session.scalar(
            select(func.count()).select_from(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
        )
        total_pages = max(1, (total_count + MEDIA_PAGE_SIZE - 1) // MEDIA_PAGE_SIZE)
        last_page = total_pages - 1
        media_list = (
            await session.execute(
                select(MediaLibrary)
                .where(MediaLibrary.topic_id == topic_id)
                .order_by(MediaLibrary.id)
                .offset(last_page * MEDIA_PAGE_SIZE)
                .limit(MEDIA_PAGE_SIZE)
            )
        ).scalars().all()
        topic = await session.get(Topic, topic_id)

    await states.clear(user_id)

    if m_name == "_back":
        usage_hint = (
            f"🃏 Это рубашка для категории <code>{category}</code>.\n"
            f"Теперь AI может использовать скрытый выбор:\n"
            f"<code>[CHOICE_IMG_HIDDEN: {category} | 3]</code>"
        )
    elif m_type == "audio":
        usage_hint = f"<code>[SEND_AUDIO: {m_name}]</code>"
    else:
        usage_hint = (
            f"<code>[RANDOM_IMG: {category}]</code> — одна случайная карта\n"
            f"<code>[RANDOM_IMG: {category} | 5]</code> — 5 случайных карт сразу\n"
            f"<code>[CHOICE_IMG: {category} | 3]</code> — выбор из 3 карт (лицом)\n"
            f"<code>[CHOICE_IMG: {category} | 3 | 5]</code> — расклад из 5 карт, выбор из 3\n"
            f"<code>[CHOICE_IMG_HIDDEN: {category} | 3]</code> — выбор вслепую (рубашкой)\n"
            f"<code>[CHOICE_IMG_HIDDEN: {category} | 3 | 5]</code> — расклад из 5 вслепую\n"
            f"<code>[SHOW_IMG: {m_name}]</code> — показать именно эту карту"
        )

    topic_name = topic.name if topic else str(topic_id)
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ <b>Файл успешно добавлен!</b>\n\n"
            f"AI может использовать его через теги:\n"
            f"{usage_hint}\n\n"
            f"Файл привязан к теме: <b>{topic_name}</b>"
        ),
    )

    text_list = await _build_list_text(topic, total_count, topic_id)
    kb = _topic_media_list_keyboard(topic_id, media_list, last_page, total_pages)
    await client.send_message(chat_id=chat_id, text=text_list, attachments=kb)


# ── Delete ────────────────────────────────────────────────────────────────────

async def delete_media(
    client: MaxApiClient,
    chat_id: int,
    media_id: int,
) -> None:
    async with async_session_maker() as session:
        media = await session.get(MediaLibrary, media_id)
        if not media:
            await client.send_message(chat_id=chat_id, text="Файл не найден.")
            return
        topic_id = media.topic_id
        await session.delete(media)
        await session.commit()

        total_count = await session.scalar(
            select(func.count()).select_from(MediaLibrary).where(MediaLibrary.topic_id == topic_id)
        )
        total_pages = max(1, (total_count + MEDIA_PAGE_SIZE - 1) // MEDIA_PAGE_SIZE)
        media_list = (
            await session.execute(
                select(MediaLibrary)
                .where(MediaLibrary.topic_id == topic_id)
                .order_by(MediaLibrary.id)
                .limit(MEDIA_PAGE_SIZE)
            )
        ).scalars().all()
        topic = await session.get(Topic, topic_id)

    await client.send_message(chat_id=chat_id, text="🗑️ Файл удалён.")
    text = await _build_list_text(topic, total_count, topic_id)
    kb = _topic_media_list_keyboard(topic_id, media_list, 0, total_pages)
    await client.send_message(chat_id=chat_id, text=text, attachments=kb)
