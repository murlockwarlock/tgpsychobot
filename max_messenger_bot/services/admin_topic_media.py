from __future__ import annotations

from sqlalchemy import case, func, select

from media_scope import ensure_topic_collection, load_media_scope, media_scope_predicate

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..models import canonical_media_type
from ..legacy import (
    MediaCollection,
    MediaLibrary,
    Topic,
    async_session_maker,
    media_collection_items,
)
from ..storage import StateStore

MEDIA_PAGE_SIZE = 10


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
        rows.append([
            callback_button(
                f"{type_icon} {label}",
                f"admin_media_view_{topic_id}_{page}_{media.id}",
            )
        ])
    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"admin_topic_media_{topic_id}_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"admin_topic_media_{topic_id}_{page + 1}"))
    if nav_row:
        rows.append(nav_row)
    rows.append([callback_button("➕ Добавить файл", f"admin_media_add_{topic_id}_{page}")])
    rows.append([callback_button("⬅️ К теме", f"admin_edit_topic_{topic_id}")])
    return inline_keyboard(rows)


def _add_collection_keyboard(topic_id: int, page: int, collections: list[tuple[int, str]]) -> list[dict]:
    rows = [
        [callback_button(name, f"admin_media_add_collection_{topic_id}_{page}_{coll_id}")]
        for coll_id, name in collections
    ]
    rows.append([callback_button("⬅️ Назад к списку", f"admin_topic_media_{topic_id}_{page}")])
    return inline_keyboard(rows)


def _media_detail_keyboard(media_id: int, topic_id: int, page: int) -> list[dict]:
    return inline_keyboard(
        [
            [
                callback_button("✏️ Имя", f"admin_media_editname_{topic_id}_{page}_{media_id}"),
                callback_button("✏️ Категорию", f"admin_media_editcat_{topic_id}_{page}_{media_id}"),
            ],
            [
                callback_button("✏️ Описание", f"admin_media_editdesc_{topic_id}_{page}_{media_id}"),
                callback_button("🔄 Файл", f"admin_media_editfile_{topic_id}_{page}_{media_id}"),
            ],
            [callback_button("⬅️ Назад к списку", f"admin_topic_media_{topic_id}_{page}")],
        ]
    )


async def _scoped_media(session, topic_id: int, media_id: int):
    return await session.scalar(
        select(MediaLibrary).where(media_scope_predicate(topic_id), MediaLibrary.id == media_id)
    )


def _build_list_text(topic: Topic, total_count: int, category_rows: list[tuple[str, int]]) -> str:
    categories: set[str] = set()
    back_categories: set[str] = set()
    for category, has_back in category_rows:
        categories.add(category)
        if has_back:
            back_categories.add(category)

    cats_info = ""
    if categories:
        lines = []
        for category in sorted(categories):
            marker = "🃏" if category in back_categories else "⚠️ нет рубашки"
            lines.append(f"  <code>{category}</code> — {marker}")
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


async def show_list(
    client: MaxApiClient,
    chat_id: int,
    topic_id: int,
    page: int = 0,
) -> None:
    async with async_session_maker() as session:
        predicate = media_scope_predicate(topic_id)
        total_count = await session.scalar(
            select(func.count()).select_from(MediaLibrary).where(predicate)
        ) or 0
        total_pages = max(1, (total_count + MEDIA_PAGE_SIZE - 1) // MEDIA_PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))
        media_list = (
            await session.execute(
                select(MediaLibrary)
                .where(predicate)
                .order_by(MediaLibrary.id)
                .offset(page * MEDIA_PAGE_SIZE)
                .limit(MEDIA_PAGE_SIZE)
            )
        ).scalars().all()
        category_rows = (
            await session.execute(
                select(
                    MediaLibrary.category,
                    func.max(case((MediaLibrary.file_name == "_back", 1), else_=0)),
                )
                .where(
                    predicate,
                    MediaLibrary.category.is_not(None),
                    MediaLibrary.category != "",
                )
                .group_by(MediaLibrary.category)
                .order_by(MediaLibrary.category)
            )
        ).all()
        topic = await session.get(Topic, topic_id)

    if not topic:
        await client.send_message(chat_id=chat_id, text="Тема не найдена.")
        return

    text = _build_list_text(topic, total_count, category_rows)
    kb = _topic_media_list_keyboard(topic_id, media_list, page, total_pages)
    await client.send_message(chat_id=chat_id, text=text, attachments=kb)


async def show_media_detail(
    client: MaxApiClient,
    chat_id: int,
    media_id: int,
    topic_id: int | None = None,
    page: int = 0,
) -> None:
    if topic_id is None:
        await client.send_message(chat_id=chat_id, text="Контекст темы потерян. Откройте файл из медиатеки темы.")
        return

    async with async_session_maker() as session:
        media = await _scoped_media(session, topic_id, media_id)
        if not media:
            await client.send_message(chat_id=chat_id, text="Файл не найден в медиатеке этой темы.")
            return
        coll_res = await session.execute(
            select(MediaCollection.name)
            .join(media_collection_items, media_collection_items.c.collection_id == MediaCollection.id)
            .where(media_collection_items.c.media_id == media_id)
            .order_by(MediaCollection.name)
        )
        coll_names = [row[0] for row in coll_res.all()]

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
    kb = _media_detail_keyboard(media.id, topic_id, page)
    await client.send_message(chat_id=chat_id, text=text, attachments=kb)


async def _start_edit(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
    topic_id: int | None,
    page: int,
    state_name: str,
    prompt: str,
) -> None:
    if topic_id is None:
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="Контекст темы потерян.")
        return
    async with async_session_maker() as session:
        media = await _scoped_media(session, topic_id, media_id)
    if not media:
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="Файл не найден в медиатеке этой темы.")
        return
    await states.set(
        user_id,
        chat_id,
        state_name,
        {"media_id": media_id, "topic_id": topic_id, "page": page},
    )
    await client.send_message(
        chat_id=chat_id,
        text=prompt,
        attachments=inline_keyboard([
            [callback_button("❌ Отмена", f"admin_topic_media_{topic_id}_{page}")]
        ]),
    )


async def _save_edit(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    updates: dict[str, str],
    success_text: str,
) -> None:
    snapshot = await states.get(user_id)
    data = (snapshot.data if snapshot else {}) or {}
    media_id = data.get("media_id")
    topic_id = data.get("topic_id")
    page = data.get("page", 0)
    if not media_id or topic_id is None:
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="Состояние потеряно.")
        return

    async with async_session_maker() as session:
        media = await _scoped_media(session, topic_id, media_id)
        if not media:
            await states.clear(user_id)
            await client.send_message(chat_id=chat_id, text="Файл больше недоступен в этой теме.")
            return
        for field, value in updates.items():
            setattr(media, field, value)
        await session.commit()

    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=success_text)
    await show_media_detail(client, chat_id, media_id, topic_id, page)


async def start_edit_name(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
    topic_id: int | None = None,
    page: int = 0,
) -> None:
    await _start_edit(
        client, states, chat_id, user_id, media_id, topic_id, page,
        "admin_media_edit_name",
        "Введи новое <b>техническое имя</b> для файла (на английском, без пробелов):",
    )


async def save_edit_name(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    new_name = text.strip().lower().replace(" ", "_")
    await _save_edit(
        client, states, chat_id, user_id, {"file_name": new_name},
        f"✅ Имя изменено на <code>{new_name}</code>.",
    )


async def start_edit_category(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
    topic_id: int | None = None,
    page: int = 0,
) -> None:
    await _start_edit(
        client, states, chat_id, user_id, media_id, topic_id, page,
        "admin_media_edit_category",
        "Введи новую <b>категорию</b> (например: <code>tarot</code>, <code>mak</code>, <code>oracle</code>):",
    )


async def save_edit_category(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    new_category = text.strip().lower().replace(" ", "_")
    await _save_edit(
        client, states, chat_id, user_id, {"category": new_category},
        f"✅ Категория изменена на <code>{new_category}</code>.",
    )


async def start_edit_description(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
    topic_id: int | None = None,
    page: int = 0,
) -> None:
    await _start_edit(
        client, states, chat_id, user_id, media_id, topic_id, page,
        "admin_media_edit_desc",
        "Введи новое <b>описание</b> для файла:",
    )


async def save_edit_description(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    await _save_edit(
        client, states, chat_id, user_id, {"description": text.strip()},
        "✅ Описание обновлено.",
    )


async def start_edit_file(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    media_id: int,
    topic_id: int | None = None,
    page: int = 0,
) -> None:
    await _start_edit(
        client, states, chat_id, user_id, media_id, topic_id, page,
        "admin_media_edit_file",
        "Отправь новый файл (фото или аудио) для замены, либо вставь токен файла напрямую.",
    )


async def save_edit_file(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    token: str | None,
    media_type: str | None,
) -> None:
    media_type = canonical_media_type(media_type)
    if not token:
        await client.send_message(chat_id=chat_id, text="Отправь файл или вставь его токен.")
        return
    updates = {"file_id": token}
    if media_type:
        updates["media_type"] = media_type
    await _save_edit(
        client, states, chat_id, user_id, updates,
        "✅ Файл заменён.",
    )


_TYPE_MAP = {
    "photo": "photo",
    "image": "photo",
    "фото": "photo",
    "audio": "audio",
    "аудио": "audio",
}


async def start_add_media(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    topic_id: int,
    page: int = 0,
    collection_id: int | None = None,
) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if not topic:
            await states.clear(user_id)
            await client.send_message(chat_id=chat_id, text="Тема не найдена.")
            return
        scope = await load_media_scope(session, topic_id, include_media_ids=False)
        if collection_id is not None:
            if collection_id not in scope.collection_ids:
                await states.clear(user_id)
                await client.send_message(chat_id=chat_id, text="Коллекция не привязана к этой теме.")
                return
            target_collection_id = collection_id
            collection_name = await session.scalar(
                select(MediaCollection.name).where(MediaCollection.id == target_collection_id)
            )
        elif len(scope.collection_ids) == 1:
            target_collection_id = scope.collection_ids[0]
            collection_name = await session.scalar(
                select(MediaCollection.name).where(MediaCollection.id == target_collection_id)
            )
        elif len(scope.collection_ids) > 1:
            collection_rows = (
                await session.execute(
                    select(MediaCollection.id, MediaCollection.name)
                    .where(MediaCollection.id.in_(scope.collection_ids))
                    .order_by(MediaCollection.name)
                )
            ).all()
            await states.clear(user_id)
            await client.send_message(
                chat_id=chat_id,
                text="Выберите коллекцию для нового файла:",
                attachments=_add_collection_keyboard(topic_id, page, collection_rows),
            )
            return
        else:
            target_collection_id = await ensure_topic_collection(session, topic_id)
            collection_name = await session.scalar(
                select(MediaCollection.name).where(MediaCollection.id == target_collection_id)
            ) if target_collection_id else None
            await session.commit()

    if not target_collection_id:
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="Не удалось выбрать коллекцию.")
        return

    await states.set(
        user_id,
        chat_id,
        "admin_media_add_file",
        {"topic_id": topic_id, "page": page, "collection_id": target_collection_id},
    )
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"Коллекция: <b>{collection_name or target_collection_id}</b>\n"
            "Отправьте файл (фото, аудио) или вставьте токен файла для добавления в медиатеку.\n"
            "Если вставляете токен текстом — просто отправьте его следующим сообщением."
        ),
        attachments=inline_keyboard([
            [callback_button("❌ Отмена", f"admin_topic_media_{topic_id}_{page}")]
        ]),
    )


async def receive_add_file(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str | None = None,
    media_token: str | None = None,
    media_type: str | None = None,
) -> None:
    media_type = canonical_media_type(media_type)
    snapshot = await states.get(user_id)
    data = (snapshot.data if snapshot else {}) or {}
    topic_id = data.get("topic_id")

    if media_token and media_type:
        token = media_token
        m_type = media_type
    elif text:
        await states.set(
            user_id,
            chat_id,
            "admin_media_add_type",
            {**data, "token": text.strip()},
        )
        await client.send_message(
            chat_id=chat_id,
            text="Токен принят. Укажи тип файла: <b>photo</b> (фото/изображение) или <b>audio</b> (аудио).",
        )
        return
    else:
        await client.send_message(chat_id=chat_id, text="Пожалуйста, отправь файл или вставь токен.")
        return

    await states.set(
        user_id,
        chat_id,
        "admin_media_add_name",
        {**data, "topic_id": topic_id, "token": token, "media_type": m_type},
    )
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ <b>Файл получен как {m_type}!</b>\n\n"
            "Придумай короткое <b>техническое имя</b> на английском (например: "
            "<code>morning_meditation</code>, <code>card_death</code>)."
        ),
    )


async def resolve_add_type(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    data = (snapshot.data if snapshot else {}) or {}
    m_type = _TYPE_MAP.get(text.strip().lower())
    if not m_type:
        await client.send_message(chat_id=chat_id, text="Введи <b>photo</b> или <b>audio</b>.")
        return
    await states.set(
        user_id,
        chat_id,
        "admin_media_add_name",
        {**data, "media_type": m_type},
    )
    await client.send_message(
        chat_id=chat_id,
        text=f"✅ Тип <b>{m_type}</b> принят.\n\nПридумай короткое <b>техническое имя</b> на английском.",
    )


async def save_add_name(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    data = (snapshot.data if snapshot else {}) or {}
    tech_name = text.strip().lower().replace(" ", "_")
    next_state = "admin_media_add_category" if data.get("media_type") == "photo" else "admin_media_add_desc"
    await states.set(user_id, chat_id, next_state, {**data, "file_name": tech_name})
    if next_state == "admin_media_add_category":
        await client.send_message(
            chat_id=chat_id,
            text=(
                f"👌 Имя <code>{tech_name}</code> принято.\n\n"
                "Введи <b>категорию</b> для изображения (например: <code>tarot</code>, <code>mak</code>, <code>oracle</code>)."
            ),
        )
    else:
        await client.send_message(
            chat_id=chat_id,
            text=f"👌 Имя <code>{tech_name}</code> принято.\n\nВведи описание файла.",
        )


async def save_add_category(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    data = (snapshot.data if snapshot else {}) or {}
    category = text.strip().lower().replace(" ", "_")
    await states.set(user_id, chat_id, "admin_media_add_desc", {**data, "category": category})
    await client.send_message(
        chat_id=chat_id,
        text=f"👌 Категория <code>{category}</code> принята.\n\nВведи описание карты.",
    )


async def save_add_description(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    data = (snapshot.data if snapshot else {}) or {}
    topic_id = data.get("topic_id")
    collection_id = data.get("collection_id")
    m_type = data.get("media_type", "photo")
    m_name = data.get("file_name", "")
    token = data.get("token", "")
    category = data.get("category", "")
    page = data.get("page", 0)
    if not topic_id or not collection_id or not token or not m_name:
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="Состояние добавления потеряно.")
        return

    async with async_session_maker() as session:
        scope = await load_media_scope(session, topic_id, include_media_ids=False)
        if collection_id not in scope.collection_ids:
            await states.clear(user_id)
            await client.send_message(chat_id=chat_id, text="Коллекция больше не привязана к этой теме.")
            return
        new_media = MediaLibrary(
            media_type=m_type,
            file_id=token,
            file_name=m_name,
            category=category,
            description=text.strip(),
        )
        session.add(new_media)
        await session.flush()
        await session.execute(
            media_collection_items.insert().values(
                collection_id=collection_id,
                media_id=new_media.id,
            )
        )
        topic = await session.get(Topic, topic_id)
        await session.commit()

    await states.clear(user_id)

    if m_name == "_back":
        usage_hint = f"🃏 Рубашка категории <code>{category}</code>: <code>[CHOICE_IMG_HIDDEN: {category} | 3]</code>"
    elif m_type == "audio":
        usage_hint = f"<code>[SEND_AUDIO: {m_name}]</code>"
    else:
        usage_hint = (
            f"<code>[RANDOM_IMG: {category}]</code>\n"
            f"<code>[CHOICE_IMG: {category} | 3]</code>\n"
            f"<code>[SHOW_IMG: {m_name}]</code>"
        )

    topic_name = topic.name if topic else str(topic_id)
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ <b>Файл успешно добавлен!</b>\n\n"
            f"AI может использовать его через теги:\n{usage_hint}\n\n"
            f"Файл привязан к теме: <b>{topic_name}</b>"
        ),
    )
    await show_list(client, chat_id, topic_id, page)


async def delete_media(
    client: MaxApiClient,
    chat_id: int,
    media_id: int,
    topic_id: int | None = None,
    page: int = 0,
) -> None:
    if topic_id is None:
        await client.send_message(chat_id=chat_id, text="Контекст темы потерян. Файл не удалён.")
        return
    async with async_session_maker() as session:
        media = await _scoped_media(session, topic_id, media_id)
        if not media:
            await client.send_message(chat_id=chat_id, text="Файл не найден в медиатеке этой темы.")
            return
    await client.send_message(
        chat_id=chat_id,
        text="Общий файл нельзя удалить из просмотра темы. Управляйте его членством в конкретной коллекции.",
    )
    await show_media_detail(client, chat_id, media_id, topic_id, page)
