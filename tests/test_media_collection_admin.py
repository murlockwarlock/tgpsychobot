from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import database
import handlers
from database import (
    Base,
    MediaCollection,
    MediaLibrary,
    Topic,
    TopicMediaDeck,
    main_dialogue_collection_association,
    media_collection_items,
    topic_collection_association,
)
from media_scope import load_available_media, load_media_scope


@pytest_asyncio.fixture
async def admin_store(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'media-admin.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    try:
        yield sessions
    finally:
        await engine.dispose()


class FakeState:
    def __init__(self):
        self.state = None
        self.data = {}

    async def set_state(self, state):
        self.state = state

    async def update_data(self, **values):
        self.data.update(values)

    async def get_data(self):
        return dict(self.data)

    async def clear(self):
        self.state = None
        self.data.clear()


class FakeMessage:
    def __init__(self, text=None, *, message_id=100, photo=None, video=None, audio=None, document=None):
        self.text = text
        self.message_id = message_id
        self.chat = SimpleNamespace(id=900)
        self.from_user = SimpleNamespace(id=1)
        self.photo = photo
        self.video = video
        self.audio = audio
        self.voice = None
        self.document = document
        self.events = []
        self.deleted = False

    async def edit_text(self, text, **kwargs):
        self.text = text
        self.events.append(("edit_text", {"text": text, **kwargs}))

    async def answer(self, text=None, **kwargs):
        self.events.append(("answer", {"text": text, **kwargs}))
        return FakeMessage(text=text, message_id=self.message_id + len(self.events))

    async def answer_photo(self, **kwargs):
        self.events.append(("answer_photo", kwargs))
        return FakeMessage(message_id=self.message_id + len(self.events))

    async def answer_video(self, **kwargs):
        self.events.append(("answer_video", kwargs))
        return FakeMessage(message_id=self.message_id + len(self.events))

    async def answer_audio(self, **kwargs):
        self.events.append(("answer_audio", kwargs))
        return FakeMessage(message_id=self.message_id + len(self.events))

    async def answer_document(self, **kwargs):
        self.events.append(("answer_document", kwargs))
        return FakeMessage(message_id=self.message_id + len(self.events))

    async def delete(self):
        self.deleted = True


class FakeBot:
    def __init__(self):
        self.events = []

    async def edit_message_text(self, **kwargs):
        self.events.append(("edit_text", kwargs))

    async def send_message(self, **kwargs):
        self.events.append(("message", kwargs))

    async def send_photo(self, **kwargs):
        self.events.append(("photo", kwargs))

    async def send_video(self, **kwargs):
        self.events.append(("video", kwargs))

    async def send_audio(self, **kwargs):
        self.events.append(("audio", kwargs))

    async def send_document(self, **kwargs):
        self.events.append(("document", kwargs))

    async def get_me(self):
        return SimpleNamespace(username="test_bot")


class FakeCallback:
    def __init__(self, data, message, bot):
        self.data = data
        self.message = message
        self.bot = bot
        self.from_user = SimpleNamespace(id=1)
        self.answers = []

    async def answer(self, *args, **kwargs):
        self.answers.append((args, kwargs))


def _buttons(markup):
    return [button for row in markup.inline_keyboard for button in row]


async def _seed_collection(sessions, name="Cards"):
    async with sessions() as session:
        collection = MediaCollection(name=name)
        session.add(collection)
        await session.commit()
        return collection.id


@pytest.mark.asyncio
async def test_collection_upload_cancel_and_successful_save_return_to_collection_files(admin_store):
    coll_id = await _seed_collection(admin_store)
    bot = FakeBot()
    state = FakeState()
    detail_message = FakeMessage("collection")

    await handlers.admin_coll_upload(
        FakeCallback(f"admin_coll_upload_{coll_id}_3", detail_message, bot),
        state,
    )
    assert state.state == handlers.AdminCollectionState.waiting_for_upload_file
    cancel_data = detail_message.events[-1][1]["reply_markup"].inline_keyboard[0][0].callback_data
    await handlers.admin_coll_view(
        FakeCallback(cancel_data, detail_message, bot),
        state,
    )
    assert state.state is None
    assert "Медиа-коллекции" not in detail_message.text
    assert "Файлов:" in detail_message.text

    state = FakeState()
    await handlers.admin_coll_upload(
        FakeCallback(f"admin_coll_upload_{coll_id}_3", detail_message, bot),
        state,
    )
    uploaded_photo = SimpleNamespace(file_id="photo-token", file_unique_id="unique-photo")
    upload_message = FakeMessage(photo=[uploaded_photo], message_id=200)
    await handlers.admin_coll_upload_file(upload_message, state)
    await handlers.admin_coll_upload_name(FakeMessage("card_one"), state)
    await handlers.admin_coll_upload_category(FakeMessage("tarot"), state)
    description_message = FakeMessage("Card description")
    await handlers.admin_coll_upload_description(description_message, state)

    assert state.state is None
    assert "Файлы коллекции" in description_message.events[-1][1]["text"]
    async with admin_store() as session:
        media = await session.scalar(select_media_by_file_name("card_one"))
        assert media is not None
        assert media.media_type == "photo"
        assert media.category == "tarot"
        assert media.description == "Card description"
        assert await session.scalar(
            select_association_count(coll_id, media.id)
        ) == 1

    files_message = FakeMessage("detail")
    await handlers.admin_coll_files(
        FakeCallback(f"admin_coll_files_{coll_id}_0_3", files_message, bot),
        FakeState(),
    )
    markup = files_message.events[-1][1]["reply_markup"]
    assert any(f"admin_coll_file_view_{coll_id}_" in button.callback_data for button in _buttons(markup))

    upload_callback = next(
        button.callback_data
        for button in _buttons(markup)
        if button.callback_data.startswith("admin_coll_upload_files_")
    )
    await handlers.admin_coll_upload_from_files(FakeCallback(upload_callback, files_message, bot), state)
    assert state.state == handlers.AdminCollectionState.waiting_for_upload_file
    cancel_data = files_message.events[-1][1]["reply_markup"].inline_keyboard[0][0].callback_data
    await handlers.admin_coll_files(FakeCallback(cancel_data, files_message, bot), state)
    assert state.state is None
    assert "Файлы коллекции" in files_message.events[-1][1]["text"]


@pytest.mark.asyncio
async def test_collection_upload_can_skip_description_and_omits_empty_rendering(admin_store):
    coll_id = await _seed_collection(admin_store)
    bot = FakeBot()
    state = FakeState()
    detail_message = FakeMessage("collection")

    await handlers.admin_coll_upload(
        FakeCallback(f"admin_coll_upload_{coll_id}_0", detail_message, bot),
        state,
    )
    upload_message = FakeMessage(
        photo=[SimpleNamespace(file_id="photo-token", file_unique_id="unique-photo")],
    )
    await handlers.admin_coll_upload_file(upload_message, state)
    await handlers.admin_coll_upload_name(FakeMessage("card_without_description"), state)
    category_message = FakeMessage("tarot")
    await handlers.admin_coll_upload_category(category_message, state)
    description_buttons = _buttons(category_message.events[-1][1]["reply_markup"])
    assert any(button.text == "Пропустить" for button in description_buttons)

    description_prompt = FakeMessage("description")
    await handlers.admin_coll_upload_skip_description(
        FakeCallback("admin_coll_upload_skip_description", description_prompt, bot),
        state,
    )

    assert state.state is None
    async with admin_store() as session:
        media = await session.scalar(select_media_by_file_name("card_without_description"))
        assert media is not None
        assert media.description is None
        detail = handlers._collection_media_text(media, ["Cards"])

    assert "Описание:" not in detail
    assert "None" not in detail


@pytest.mark.asyncio
async def test_collection_description_clear_button_is_explicit_and_idempotent(admin_store):
    coll_id = await _seed_collection(admin_store)
    async with admin_store() as session:
        media = MediaLibrary(
            file_id="photo-token",
            file_name="described-card",
            media_type="photo",
            description="Existing description",
        )
        session.add(media)
        await session.flush()
        await session.execute(media_collection_items.insert().values(collection_id=coll_id, media_id=media.id))
        await session.commit()
        media_id = media.id

    bot = FakeBot()
    state = FakeState()
    prompt_message = FakeMessage("settings")
    await handlers.admin_coll_media_edit_desc_start(
        FakeCallback(f"admin_coll_media_editdesc_{media_id}_{coll_id}_0_0", prompt_message, bot),
        state,
    )
    clear_data = next(
        button.callback_data
        for button in _buttons(prompt_message.events[-1][1]["reply_markup"])
        if button.text == "Очистить"
    )
    await handlers.admin_coll_media_clear_desc(
        FakeCallback(clear_data, prompt_message, bot),
        state,
    )

    async with admin_store() as session:
        assert (await session.get(MediaLibrary, media_id)).description is None

    await handlers.admin_coll_media_clear_desc(
        FakeCallback(clear_data, FakeMessage("settings"), bot),
        FakeState(),
    )
    async with admin_store() as session:
        assert (await session.get(MediaLibrary, media_id)).description is None


def select_media_by_file_name(file_name):
    from sqlalchemy import select

    return select(MediaLibrary).where(MediaLibrary.file_name == file_name)


def select_association_count(coll_id, media_id):
    from sqlalchemy import select, func

    return select(func.count()).select_from(media_collection_items).where(
        media_collection_items.c.collection_id == coll_id,
        media_collection_items.c.media_id == media_id,
    )


@pytest.mark.asyncio
async def test_collection_list_detail_crud_navigation(admin_store):
    coll_id = await _seed_collection(admin_store)
    bot = FakeBot()
    message = FakeMessage("admin")

    await handlers.admin_collections_page(
        FakeCallback("admin_collections_page_2", message, bot),
        FakeState(),
    )
    list_markup = bot.events[-1][1]["reply_markup"]
    view_callback = next(
        button.callback_data for button in _buttons(list_markup) if button.callback_data.startswith("admin_coll_view_")
    )
    await handlers.admin_coll_view(FakeCallback(view_callback, message, bot), FakeState())
    detail_markup = message.events[-1][1]["reply_markup"]
    assert "admin_coll_files_" in [button.callback_data for button in _buttons(detail_markup)][1]

    rename_callback = next(
        button.callback_data for button in _buttons(detail_markup) if button.callback_data.startswith("admin_coll_rename_")
    )
    rename_state = FakeState()
    await handlers.admin_coll_rename(FakeCallback(rename_callback, message, bot), rename_state)
    cancel_callback = message.events[-1][1]["reply_markup"].inline_keyboard[0][0].callback_data
    await handlers.admin_coll_view(FakeCallback(cancel_callback, message, bot), rename_state)
    assert rename_state.state is None
    assert "Файлов:" in message.text

    detail_markup = message.events[-1][1]["reply_markup"]
    rename_callback = next(
        button.callback_data for button in _buttons(detail_markup) if button.callback_data.startswith("admin_coll_rename_")
    )
    rename_state = FakeState()
    await handlers.admin_coll_rename(FakeCallback(rename_callback, message, bot), rename_state)
    await handlers.admin_coll_rename_done(FakeMessage("Renamed"), rename_state)
    assert rename_state.state is None
    async with admin_store() as session:
        assert (await session.get(MediaCollection, coll_id)).name == "Renamed"

    await handlers.admin_collections_page(
        FakeCallback("admin_collections_page_2", message, bot),
        FakeState(),
    )
    list_markup = bot.events[-1][1]["reply_markup"]
    create_callback = next(
        button.callback_data for button in _buttons(list_markup) if button.callback_data.startswith("admin_coll_create_")
    )
    create_state = FakeState()
    await handlers.admin_coll_create(FakeCallback(create_callback, message, bot), create_state)
    cancel_callback = message.events[-1][1]["reply_markup"].inline_keyboard[0][0].callback_data
    await handlers.admin_collections_page(FakeCallback(cancel_callback, message, bot), create_state)
    assert create_state.state is None

    list_markup = bot.events[-1][1]["reply_markup"]
    create_callback = next(
        button.callback_data for button in _buttons(list_markup) if button.callback_data.startswith("admin_coll_create_")
    )
    create_state = FakeState()
    await handlers.admin_coll_create(FakeCallback(create_callback, message, bot), create_state)
    created_input = FakeMessage("name")
    await handlers.admin_coll_create_name(created_input, create_state)
    assert create_state.state is None
    created_detail_markup = created_input.events[-1][1]["reply_markup"]
    created_delete_callback = next(
        button.callback_data
        for button in _buttons(created_detail_markup)
        if button.callback_data.startswith("admin_coll_delete_")
    )
    created_id = int(created_delete_callback.split("_")[3])
    await handlers.admin_coll_delete(
        FakeCallback(
            created_delete_callback,
            created_input,
            bot,
        ),
        FakeState(),
    )
    assert "Медиа-коллекции" in bot.events[-1][1]["text"]
    async with admin_store() as session:
        assert await session.get(MediaCollection, created_id) is None


@pytest.mark.asyncio
async def test_collection_attach_detach_and_back_preserve_collection_page(admin_store):
    coll_id = await _seed_collection(admin_store)
    async with admin_store() as session:
        media = MediaLibrary(file_id="unassigned", file_name="unassigned", media_type="audio")
        session.add(media)
        await session.commit()
        media_id = media.id

    bot = FakeBot()
    message = FakeMessage("files")
    state = FakeState()
    await handlers.admin_coll_attach(
        FakeCallback(f"admin_coll_attach_{coll_id}_3_7", message, bot),
        state,
    )
    add_callback = next(
        button.callback_data
        for button in _buttons(message.events[-1][1]["reply_markup"])
        if button.callback_data == f"coll_file_add_{coll_id}_{media_id}_0_7"
    )
    await handlers.admin_coll_toggle_file(FakeCallback(add_callback, message, bot), state)
    async with admin_store() as session:
        assert await session.scalar(select_association_count(coll_id, media_id)) == 1

    remove_callback = next(
        button.callback_data
        for button in _buttons(message.events[-1][1]["reply_markup"])
        if button.callback_data == f"coll_file_remove_{coll_id}_{media_id}_0_7"
    )
    await handlers.admin_coll_toggle_file(FakeCallback(remove_callback, message, bot), state)
    async with admin_store() as session:
        assert await session.scalar(select_association_count(coll_id, media_id)) == 0

    back_callback = next(
        button.callback_data
        for button in _buttons(message.events[-1][1]["reply_markup"])
        if button.callback_data.startswith("admin_coll_files_")
    )
    await handlers.admin_coll_files(FakeCallback(back_callback, message, bot), state)
    assert message.events[-1][1]["reply_markup"].inline_keyboard[-1][0].callback_data == f"admin_coll_view_{coll_id}_7"


@pytest.mark.asyncio
async def test_collection_file_edit_cancel_save_delete_and_back_preserve_context(admin_store):
    coll_id = await _seed_collection(admin_store)
    async with admin_store() as session:
        media = MediaLibrary(file_id="photo-token", file_name="card_one", category="tarot", description="old", media_type="photo")
        session.add(media)
        fillers = [
            MediaLibrary(file_id=f"filler-{index}", file_name=f"filler_{index}", media_type="photo")
            for index in range(10)
        ]
        session.add_all(fillers)
        await session.flush()
        await session.execute(media_collection_items.insert().values(collection_id=coll_id, media_id=media.id))
        await session.execute(
            media_collection_items.insert(),
            [{"collection_id": coll_id, "media_id": filler.id} for filler in fillers],
        )
        await session.commit()
        media_id = media.id

    bot = FakeBot()
    files_message = FakeMessage("files")
    await handlers.admin_coll_file_view(
        FakeCallback(f"admin_coll_file_view_{coll_id}_{media_id}_1_4", files_message, bot),
        FakeState(),
    )
    assert bot.events[-1][0] == "photo"
    settings_markup = bot.events[-1][1]["reply_markup"]
    file_back_callback = next(
        button.callback_data
        for button in _buttons(settings_markup)
        if button.callback_data.startswith("admin_coll_files_")
    )

    edit_state = FakeState()
    edit_message = FakeMessage("settings")
    await handlers.admin_coll_media_edit_name_start(
        FakeCallback(f"admin_coll_media_editname_{media_id}_{coll_id}_1_4", edit_message, bot),
        edit_state,
    )
    cancel_data = edit_message.events[-1][1]["reply_markup"].inline_keyboard[0][0].callback_data
    await handlers.admin_coll_media_cancel(FakeCallback(cancel_data, edit_message, bot), edit_state)
    assert edit_state.state is None
    assert bot.events[-1][0] == "photo"

    files_back_message = FakeMessage("settings")
    await handlers.admin_coll_files(
        FakeCallback(file_back_callback, files_back_message, bot),
        FakeState(),
    )
    assert "Файлы коллекции" in files_back_message.events[-1][1]["text"]
    assert any(
        button.callback_data.startswith(f"admin_coll_file_view_{coll_id}_")
        and button.callback_data.endswith("_1_4")
        for button in _buttons(files_back_message.events[-1][1]["reply_markup"])
    )

    collection_back_callback = next(
        button.callback_data
        for button in _buttons(files_back_message.events[-1][1]["reply_markup"])
        if button.callback_data.startswith("admin_coll_view_")
    )
    await handlers.admin_coll_view(
        FakeCallback(collection_back_callback, files_back_message, bot),
        FakeState(),
    )
    assert "Файлов:" in files_back_message.text
    detail_markup = handlers.kb.admin_collection_detail_keyboard(coll_id, 4)
    list_back_callback = next(
        button.callback_data
        for button in _buttons(detail_markup)
        if button.callback_data.startswith("admin_collections_page_")
    )
    await handlers.admin_collections_page(
        FakeCallback(list_back_callback, files_back_message, bot),
        FakeState(),
    )
    assert "Медиа-коллекции" in bot.events[-1][1]["text"]

    await handlers.admin_coll_file_view(
        FakeCallback(f"admin_coll_file_view_{coll_id}_{media_id}_1_4", files_back_message, bot),
        FakeState(),
    )

    edit_state = FakeState()
    await handlers.admin_coll_media_edit_name_start(
        FakeCallback(f"admin_coll_media_editname_{media_id}_{coll_id}_1_4", FakeMessage("settings"), bot),
        edit_state,
    )
    await handlers.admin_media_edit_name_finish(FakeMessage("card_renamed"), edit_state)
    assert edit_state.state is None
    async with admin_store() as session:
        assert await session.scalar(select_media_by_file_name("card_renamed")) is not None

    edit_state = FakeState()
    await handlers.admin_coll_media_edit_category_start(
        FakeCallback(f"admin_coll_media_editcat_{media_id}_{coll_id}_1_4", FakeMessage("settings"), bot),
        edit_state,
    )
    await handlers.admin_media_edit_category_finish(FakeMessage("oracle"), edit_state)

    edit_state = FakeState()
    await handlers.admin_coll_media_edit_desc_start(
        FakeCallback(f"admin_coll_media_editdesc_{media_id}_{coll_id}_1_4", FakeMessage("settings"), bot),
        edit_state,
    )
    await handlers.admin_media_edit_desc_finish(FakeMessage("new description"), edit_state)

    edit_state = FakeState()
    await handlers.admin_coll_media_edit_file_start(
        FakeCallback(f"admin_coll_media_editfile_{media_id}_{coll_id}_1_4", FakeMessage("settings"), bot),
        edit_state,
    )
    replacement = SimpleNamespace(file_id="replacement-photo", file_unique_id="replacement-unique")
    await handlers.admin_media_edit_file_finish(FakeMessage(photo=[replacement]), edit_state)

    async with admin_store() as session:
        updated = await session.get(MediaLibrary, media_id)
        assert updated.file_name == "card_renamed"
        assert updated.category == "oracle"
        assert updated.description == "new description"
        assert updated.file_id == "replacement-photo"
        assert updated.media_type == "photo"

    delete_message = FakeMessage("settings")
    await handlers.admin_coll_media_delete_start(
        FakeCallback(f"admin_coll_media_delete_{media_id}_{coll_id}_1_4", delete_message, bot),
        FakeState(),
    )
    cancel_data = delete_message.events[-1][1]["reply_markup"].inline_keyboard[1][0].callback_data
    await handlers.admin_coll_media_delete_cancel(
        FakeCallback(cancel_data, FakeMessage("confirmation"), bot),
        FakeState(),
    )
    async with admin_store() as session:
        assert await session.get(MediaLibrary, media_id) is not None

    delete_message = FakeMessage("settings")
    await handlers.admin_coll_media_delete_start(
        FakeCallback(f"admin_coll_media_delete_{media_id}_{coll_id}_1_4", delete_message, bot),
        FakeState(),
    )
    confirm_data = delete_message.events[-1][1]["reply_markup"].inline_keyboard[0][0].callback_data
    list_message = FakeMessage("confirmation")
    await handlers.admin_coll_media_delete_confirm(
        FakeCallback(confirm_data, list_message, bot),
        FakeState(),
    )
    async with admin_store() as session:
        assert await session.get(MediaLibrary, media_id) is not None
        assert await session.scalar(select_association_count(coll_id, media_id)) == 0
    assert "Файлы коллекции" in list_message.text


@pytest.mark.asyncio
async def test_collection_media_delete_removes_only_selected_membership(admin_store):
    async with admin_store() as session:
        first_topic = Topic(id=1, name="Topic one")
        second_topic = Topic(id=2, name="Topic two")
        first_collection = MediaCollection(name="First")
        second_collection = MediaCollection(name="Second")
        media = MediaLibrary(file_id="shared", file_name="shared", media_type="photo")
        session.add_all([first_topic, second_topic, first_collection, second_collection, media])
        await session.flush()
        await session.execute(
            topic_collection_association.insert(),
            [
                {"topic_id": first_topic.id, "collection_id": first_collection.id},
                {"topic_id": second_topic.id, "collection_id": second_collection.id},
            ],
        )
        await session.execute(
            media_collection_items.insert(),
            [
                {"collection_id": first_collection.id, "media_id": media.id},
                {"collection_id": second_collection.id, "media_id": media.id},
            ],
        )
        await session.commit()
        first_collection_id = first_collection.id
        second_collection_id = second_collection.id
        media_id = media.id

    callback = FakeCallback(
        f"admin_coll_media_delete_confirm_{media_id}_{first_collection_id}_0_0",
        FakeMessage("confirmation"),
        FakeBot(),
    )
    await handlers.admin_coll_media_delete_confirm(callback, FakeState())

    async with admin_store() as session:
        assert await session.get(MediaLibrary, media_id) is not None
        assert (await session.execute(
            select(media_collection_items.c.collection_id)
            .where(media_collection_items.c.media_id == media_id)
            .order_by(media_collection_items.c.collection_id)
        )).scalars().all() == [second_collection_id]
        scope, available = await load_available_media(session, 2)
        assert scope.collection_ids == (second_collection_id,)
        assert [item.id for item in available] == [media_id]


@pytest.mark.asyncio
async def test_collection_media_mutation_denied_for_non_admin(admin_store, monkeypatch):
    coll_id = await _seed_collection(admin_store)
    async with admin_store() as session:
        media = MediaLibrary(file_id="shared", file_name="shared", media_type="photo")
        session.add(media)
        await session.flush()
        await session.execute(media_collection_items.insert().values(collection_id=coll_id, media_id=media.id))
        await session.commit()
        media_id = media.id

    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=False))
    state = FakeState()
    state.state = "stale"
    state.data = {"pending": True}
    callback = FakeCallback(
        f"coll_file_remove_{coll_id}_{media_id}_0_0",
        FakeMessage("files"),
        FakeBot(),
    )
    await handlers.admin_coll_toggle_file(callback, state)

    assert state.state is None
    assert callback.answers == [(('Недостаточно прав администратора.',), {'show_alert': True})]
    async with admin_store() as session:
        assert await session.scalar(select_association_count(coll_id, media_id)) == 1


@pytest.mark.asyncio
async def test_collection_media_fsm_rechecks_demoted_admin_before_save(admin_store, monkeypatch):
    coll_id = await _seed_collection(admin_store)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=False))
    state = FakeState()
    state.state = handlers.AdminCollectionState.waiting_for_upload_description
    state.data = {
        "upload_coll_id": coll_id,
        "upload_file_id": "file-token",
        "upload_media_type": "photo",
        "upload_file_name": "new-card",
    }
    message = FakeMessage("description")

    await handlers.admin_coll_upload_description(message, state)

    assert state.state is None
    assert message.events[0] == ("answer", {"text": "Недостаточно прав администратора."})
    async with admin_store() as session:
        assert await session.scalar(select_media_by_file_name("new-card")) is None


@pytest.mark.asyncio
async def test_topic_and_main_collection_selectors_attach_detach_and_back(admin_store):
    async with admin_store() as session:
        topic = Topic(id=1, name="Topic one")
        collection = MediaCollection(name="Shared")
        session.add_all([topic, collection])
        await session.commit()
        coll_id = collection.id

    bot = FakeBot()
    topic_keyboard = handlers.kb.edit_topic_keyboard(1, True)
    topic_buttons = _buttons(topic_keyboard)
    assert all(button.text != "📁 Медиа-файлы темы" for button in topic_buttons)
    assert all(not button.callback_data.startswith("admin_topic_media_") for button in topic_buttons)
    legacy_state = FakeState()
    legacy_state.state = "legacy"
    await handlers.admin_topic_media_list(
        FakeCallback("admin_topic_media_1_4", FakeMessage("legacy"), bot),
        legacy_state,
    )
    assert legacy_state.state is None
    assert "Привязка коллекций к теме" in bot.events[-1][1]["text"]
    topic_state = FakeState()
    topic_message = FakeMessage("topic")
    await handlers.admin_assign_coll_to_topic(
        FakeCallback("assign_coll_topic_1_page_0", topic_message, bot),
        topic_state,
    )
    topic_markup = bot.events[-1][1]["reply_markup"]
    toggle_data = next(
        button.callback_data for button in _buttons(topic_markup) if f"_{coll_id}_" in button.callback_data
    )
    await handlers.admin_toggle_coll_for_topic(FakeCallback(toggle_data, topic_message, bot), topic_state)
    async with admin_store() as session:
        assert await session.scalar(
            select(topic_collection_association.c.collection_id).where(
                topic_collection_association.c.topic_id == 1,
                topic_collection_association.c.collection_id == coll_id,
            )
        ) == coll_id

    toggle_data = next(
        button.callback_data for button in _buttons(bot.events[-1][1]["reply_markup"]) if f"_{coll_id}_" in button.callback_data
    )
    await handlers.admin_toggle_coll_for_topic(FakeCallback(toggle_data, topic_message, bot), topic_state)
    async with admin_store() as session:
        assert await session.scalar(
            select(topic_collection_association.c.collection_id).where(
                topic_collection_association.c.topic_id == 1,
                topic_collection_association.c.collection_id == coll_id,
            )
        ) is None

    await handlers.cancel_handler(
        FakeCallback("cancel_state_edit_topic_1", topic_message, bot),
        topic_state,
    )
    assert topic_state.state is None
    assert "Редактирование темы" in bot.events[-1][1]["text"]
    assert any(button.callback_data.startswith("edit_topic_") for button in _buttons(bot.events[-1][1]["reply_markup"]))

    main_state = FakeState()
    main_message = FakeMessage("general")
    await handlers.admin_main_collections_page(
        FakeCallback("admin_main_collections_page_0", main_message, bot),
        main_state,
    )
    main_markup = bot.events[-1][1]["reply_markup"]
    toggle_data = next(
        button.callback_data for button in _buttons(main_markup) if f"_{coll_id}_" in button.callback_data
    )
    await handlers.admin_toggle_coll_for_main(
        FakeCallback(toggle_data, main_message, bot),
        main_state,
    )
    async with admin_store() as session:
        assert await session.scalar(
            select(main_dialogue_collection_association.c.collection_id).where(
                main_dialogue_collection_association.c.collection_id == coll_id
            )
        ) == coll_id

    await handlers.admin_general_settings(
        FakeCallback("admin_general_settings", main_message, bot),
        main_state,
    )
    assert main_state.state is None
    general_markup = main_message.events[-1][1]["reply_markup"]
    assert "admin_main_collections_page_0" in [button.callback_data for button in _buttons(general_markup)]

    await handlers.admin_toggle_coll_for_main(
        FakeCallback(f"maincoll_remove_{coll_id}_0", main_message, bot),
        main_state,
    )
    async with admin_store() as session:
        assert await session.scalar(
            select(main_dialogue_collection_association.c.collection_id).where(
                main_dialogue_collection_association.c.collection_id == coll_id
            )
        ) is None


@pytest.mark.asyncio
async def test_shared_collection_isolated_by_scope_and_ai_list_uses_same_resolver(admin_store):
    async with admin_store() as session:
        topic = Topic(id=1, name="Topic one")
        other_topic = Topic(id=2, name="Topic two")
        collection = MediaCollection(name="Shared")
        other_collection = MediaCollection(name="Other")
        first = MediaLibrary(file_id="first", file_name="first", media_type="photo")
        second = MediaLibrary(file_id="second", file_name="second", media_type="audio")
        foreign = MediaLibrary(file_id="foreign", file_name="foreign", media_type="photo")
        session.add_all([topic, other_topic, collection, other_collection, first, second, foreign])
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=1, collection_id=collection.id))
        await session.execute(topic_collection_association.insert().values(topic_id=2, collection_id=other_collection.id))
        await session.execute(main_dialogue_collection_association.insert().values(collection_id=collection.id))
        await session.execute(media_collection_items.insert().values(collection_id=collection.id, media_id=first.id))
        await session.execute(media_collection_items.insert().values(collection_id=collection.id, media_id=second.id))
        await session.execute(media_collection_items.insert().values(collection_id=other_collection.id, media_id=foreign.id))
        await session.commit()

        topic_scope, topic_media = await load_available_media(session, 1)
        main_scope, main_media = await load_available_media(session, None)
        other_scope = await load_media_scope(session, 2)

    assert topic_scope.collection_media_ids == main_scope.collection_media_ids
    assert {media.file_name for media in topic_media} == {"first", "second"}
    assert {media.file_name for media in main_media} == {"first", "second"}
    assert "foreign" not in {media.file_name for media in topic_media}
    assert foreign.id in other_scope.collection_media_ids
    assert first.id not in other_scope.collection_media_ids


@pytest.mark.asyncio
async def test_legacy_migration_keeps_existing_collection_scope_and_ignores_decks(admin_store):
    async with admin_store() as session:
        existing_media = MediaLibrary(file_id="existing", file_name="existing", media_type="photo")
        direct_media = MediaLibrary(
            file_id="direct",
            file_name="direct",
            media_type="photo",
            topic_id=1,
        )
        deck_media = MediaLibrary(
            file_id="deck",
            file_name="deck",
            media_type="photo",
            category="tarot",
        )
        session.add_all([
            Topic(id=1, name="Topic one"),
            MediaCollection(name="Existing"),
            TopicMediaDeck(topic_id=1, deck_name="tarot"),
            existing_media,
            direct_media,
            deck_media,
        ])
        await session.flush()
        collection = await session.scalar(select(MediaCollection).where(MediaCollection.name == "Existing"))
        await session.execute(
            topic_collection_association.insert().values(topic_id=1, collection_id=collection.id)
        )
        await session.execute(
            media_collection_items.insert().values(collection_id=collection.id, media_id=existing_media.id)
        )
        await session.commit()

    async with admin_store() as session:
        await session.run_sync(database._migrate_legacy_media_ownership)
        await session.commit()
        scope = await load_media_scope(session, 1)
        media_names = set((await session.execute(
            select_media_by_ids(scope.collection_media_ids)
        )).scalars().all())
        direct_topic_id = await session.scalar(
            select(MediaLibrary.topic_id).where(MediaLibrary.file_name == "direct")
        )
        legacy_decks = (await session.execute(select(TopicMediaDeck))).scalars().all()

    assert media_names == {"existing", "direct"}
    assert direct_topic_id is None
    assert legacy_decks == []


@pytest.mark.asyncio
async def test_legacy_migration_keeps_direct_and_deck_media_without_existing_collection(admin_store):
    async with admin_store() as session:
        session.add_all([
            Topic(id=2, name="Topic two"),
            TopicMediaDeck(topic_id=2, deck_name="tarot"),
            MediaLibrary(file_id="direct", file_name="direct", media_type="photo", topic_id=2),
            MediaLibrary(file_id="deck", file_name="deck", media_type="photo", category="tarot"),
        ])
        await session.commit()

    async with admin_store() as session:
        await session.run_sync(database._migrate_legacy_media_ownership)
        await session.commit()
        scope = await load_media_scope(session, 2)
        media_names = set((await session.execute(
            select_media_by_ids(scope.collection_media_ids)
        )).scalars().all())

    assert media_names == {"direct", "deck"}


@pytest.mark.asyncio
async def test_legacy_migration_is_idempotent(admin_store):
    async with admin_store() as session:
        session.add_all([
            Topic(id=3, name="Topic three"),
            TopicMediaDeck(topic_id=3, deck_name="tarot"),
            MediaLibrary(file_id="direct", file_name="direct", media_type="photo", topic_id=3),
            MediaLibrary(file_id="deck", file_name="deck", media_type="photo", category="tarot"),
        ])
        await session.commit()

    async with admin_store() as session:
        await session.run_sync(database._migrate_legacy_media_ownership)
        await session.commit()
        first_collections = set((await session.execute(select(MediaCollection.name))).scalars().all())
        first_bindings = set((await session.execute(
            select(topic_collection_association.c.topic_id, topic_collection_association.c.collection_id)
        )).all())
        first_items = set((await session.execute(
            select(media_collection_items.c.collection_id, media_collection_items.c.media_id)
        )).all())

        await session.run_sync(database._migrate_legacy_media_ownership)
        await session.commit()
        second_collections = set((await session.execute(select(MediaCollection.name))).scalars().all())
        second_bindings = set((await session.execute(
            select(topic_collection_association.c.topic_id, topic_collection_association.c.collection_id)
        )).all())
        second_items = set((await session.execute(
            select(media_collection_items.c.collection_id, media_collection_items.c.media_id)
        )).all())

    assert second_collections == first_collections
    assert second_bindings == first_bindings
    assert second_items == first_items


@pytest.mark.asyncio
async def test_legacy_migration_does_not_resurrect_deleted_binding(admin_store):
    async with admin_store() as session:
        session.add_all([
            Topic(id=4, name="Topic four"),
            TopicMediaDeck(topic_id=4, deck_name="tarot"),
            MediaLibrary(file_id="direct", file_name="direct", media_type="photo", topic_id=4),
            MediaLibrary(file_id="deck", file_name="deck", media_type="photo", category="tarot"),
        ])
        await session.commit()

    async with admin_store() as session:
        await session.run_sync(database._migrate_legacy_media_ownership)
        await session.commit()
        await session.execute(
            topic_collection_association.delete().where(topic_collection_association.c.topic_id == 4)
        )
        await session.commit()

        await session.run_sync(database._migrate_legacy_media_ownership)
        await session.commit()
        binding_ids = (await session.execute(
            select(topic_collection_association.c.collection_id).where(
                topic_collection_association.c.topic_id == 4
            )
        )).scalars().all()
        media_names = set((await session.execute(
            select(MediaLibrary.file_name).where(MediaLibrary.file_name.in_(["direct", "deck"]))
        )).scalars().all())

    assert binding_ids == []
    assert media_names == {"direct", "deck"}


def select_media_by_ids(media_ids):
    from sqlalchemy import select

    return select(MediaLibrary.file_name).where(MediaLibrary.id.in_(media_ids))
