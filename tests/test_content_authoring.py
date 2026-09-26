import os

os.environ.setdefault("BOT_TOKEN", "123456:test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from content_authoring import (
    RESOURCES,
    create_resource, delete_answer, editing_locale, ensure_answer_identities,
    insert_answer, read_content_value, reorder_answers, save_content_value,
    set_editing_locale,
)
from database import Base, BotGeneralConfig, BotTranslation, TestQuestion as Question, Topic
from translation_pack_manager import audit_translation_readiness, export_translation_pack, import_translation_pack
from translation_registry import TranslationRegistry, TranslationSource, build_translation_registry
from translation_service import TranslationSnapshot, source_hash
from universal_tests import get_answer_options, make_option_answer_record


RESOURCE_FIELDS = [(kind, field) for kind, spec in RESOURCES.items() for field, _, _ in spec.fields]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,field", RESOURCE_FIELDS)
async def test_all_authorable_fields_keep_one_object_and_one_translation_source(factory, kind, field):
    spec = RESOURCES[kind]
    defaults = {
        "topic": {"name": "Тема"}, "content": {"key": "example"},
        "plan": {"name": "Тариф", "price": 100, "duration_value": 1, "duration_unit": "months"},
        "test_question": {"text": "Вопрос", "category": "custom"},
        "secret_test_question": {"text": "Вопрос"},
        "mailing": {"target_audience": "all", "status": "draft"},
        "automation_action": {"handler_id": 1, "action_type": "send_message", "recipient_type": "event_user"},
        "followup_step": {"campaign_id": 1, "delay_minutes": 10, "message_type": "static"},
        "referral_template": {"text": "Приглашение"},
        "case_study": {"text": "История"},
        "subscription_config": {"id": 1},
        "media_library": {"file_id": "isolated-file", "media_type": "photo"},
    }
    async with factory() as session:
        resource = await session.get(BotGeneralConfig, 1) if kind == "bot_general_config" else spec.model(**defaults[kind])
        session.add(resource)
        setattr(resource, field, "Исходный текст")
        await session.flush()
        identity = getattr(resource, spec.identity_field)
        for locale, value in (("en", "English value"), ("pt", "Valor português")):
            await save_content_value(session, kind, resource, field, locale, value)
            assert getattr(resource, field) == "Исходный текст"
            assert (await read_content_value(session, kind, resource, field, locale)).text == value
        await save_content_value(session, kind, resource, field, "ru", "Новый русский текст")
        assert getattr(resource, spec.identity_field) == identity
        for locale in ("en", "pt"):
            value = await read_content_value(session, kind, resource, field, locale)
            assert value.needs_review
            await save_content_value(session, kind, resource, field, locale, value.text)
            assert not (await read_content_value(session, kind, resource, field, locale)).needs_review
        translations = (await session.scalars(select(BotTranslation).where(BotTranslation.translation_key == f"{kind}.{identity}.{field}"))).all()
        assert len(translations) == 2
        await session.commit()
    from database import AsyncSession
    from admin_authoring_context import content_editing_locale
    token = content_editing_locale.set("pt")
    try:
        async with AsyncSession(factory.kw["bind"], expire_on_commit=False) as session:
            resource = await session.get(spec.model, identity)
            setattr(resource, field, "Edição normal")
            await session.flush()
            await session.commit()
            assert getattr(resource, field) == "Новый русский текст"
            assert (await read_content_value(session, kind, resource, field, "pt")).text == "Edição normal"
    finally:
        content_editing_locale.reset(token)


@pytest.mark.asyncio
async def test_locale_first_preserves_embedded_machine_targets(factory):
    async with factory() as session:
        resource = await create_resource(session, "content", "pt", {"text_content": "<a href=\"https://example.invalid/a\">Abrir</a>"})
        with pytest.raises(ValueError):
            await save_content_value(session, "content", resource, "text_content", "en", "<a href=\"https://example.invalid/b\">Open</a>")
        await save_content_value(session, "content", resource, "text_content", "en", "<a href=\"https://example.invalid/a\">Open</a>")
        assert not resource.text_content


@pytest.mark.asyncio
async def test_existing_keyboard_needs_refresh_without_guessing_identity(factory):
    from content_menu import needs_menu_refresh, remember_menu
    from translation_service import translation_cache
    prior = translation_cache.snapshot
    try:
        translation_cache.install(1, {"topic.17.name": "Тема"}, {("pt", "topic.17.name"): "Tema"})
        assert await needs_menu_refresh(1, "Tema", session_maker=factory)
        assert not await needs_menu_refresh(1, "Произвольное сообщение", session_maker=factory)
        await remember_menu(1, [("Tema", "topic", 17)], session_maker=factory)
        assert not await needs_menu_refresh(1, "Tema", session_maker=factory)
    finally:
        translation_cache.install(prior.revision, prior.sources, prior.translations)


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(BotGeneralConfig(id=1, telegram_enabled_languages='["ru","en","pt"]', telegram_language_selection_enabled=True))
        await session.commit()
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_workspace_is_per_admin_and_bot(factory):
    async with factory() as session:
        from database import User
        session.add(User(id=1, telegram_language_code="en"))
        await session.commit()
        assert await editing_locale(session, 1, 1) == "ru"
        await set_editing_locale(session, 1, 1, "pt")
        await session.commit()
    async with factory() as session:
        assert await editing_locale(session, 1, 1) == "pt"
        assert await editing_locale(session, 1, 2) == "ru"
        assert await editing_locale(session, 2, 1) == "ru"
        config = await session.get(BotGeneralConfig, 1)
        assert config.telegram_default_language == "ru"
        assert config.telegram_enabled_languages == '["ru","en","pt"]'
        assert config.telegram_language_selection_enabled is True
        assert (await session.get(User, 1)).telegram_language_code == "en"


def test_active_snapshot_keeps_translations_after_live_question_removal():
    from test_content_identity import question_snapshot, questions_for_session, translate_question
    from translation_service import translation_cache
    prior = translation_cache.snapshot
    question = Question(id=17, text="Вопрос", category="custom")
    try:
        translation_cache.install(1, {"test_question.17.text": "Вопрос"}, {("en", "test_question.17.text"): "Question", ("pt", "test_question.17.text"): "Pergunta"})
        session = SimpleNamespace(question_snapshot=question_snapshot([question]))
        translation_cache.install(2, {}, {})
        preserved = questions_for_session(session, [])[0]
        assert translate_question(preserved, "test_question.17.text", "pt", source="Вопрос") == "Pergunta"
        assert translate_question(preserved, "test_question.17.text", "en", source="Вопрос") == "Question"
        assert translate_question(preserved, "test_question.17.text", "ru", source="Вопрос") == "Вопрос"
    finally:
        translation_cache.install(prior.revision, prior.sources, prior.translations)


@pytest.mark.asyncio
async def test_legacy_implicit_scoring_variables_do_not_follow_question_positions(factory):
    from test_content_identity import move_question
    from universal_tests import get_question_variable
    async with factory() as session:
        first = Question(id=1, text="Первый", category="general", sort_order=0)
        second = Question(id=2, text="Второй", category="general", sort_order=1)
        session.add_all([first, second])
        await session.commit()
        assert get_question_variable(first, 0) == "answer_01"
        assert get_question_variable(second, 1) == "answer_02"
        await move_question(session, second.id, -1)
        assert get_question_variable(first, 1) == "answer_01"
        assert get_question_variable(second, 0) == "answer_02"
        await create_resource(session, "test_question", "pt", {"text": "Novo"}, {"sort_order": 0})
        assert get_question_variable(first, 2) == "answer_01"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,field", [("topic", "name"), ("content", "button_title"), ("plan", "name"), ("test_question", "text"), ("secret_test_question", "text"), ("referral_template", "text"), ("mailing", "text")])
async def test_portuguese_first_and_russian_added_to_same_resource(factory, kind, field):
    async with factory() as session:
        resource = await create_resource(session, kind, "pt", {field: "Autoestima"})
        assert getattr(resource, field) in {None, ""}
        identity = getattr(resource, "id", getattr(resource, "key", None))
        assert (await read_content_value(session, kind, resource, field, "ru")).missing
        assert (await read_content_value(session, kind, resource, field, "pt")).text == "Autoestima"
        await save_content_value(session, kind, resource, field, "ru", "Самооценка")
        assert getattr(resource, "id", getattr(resource, "key", None)) == identity
        assert (await read_content_value(session, kind, resource, field, "pt")).needs_review
        await save_content_value(session, kind, resource, field, "pt", "Autoestima")
        assert not (await read_content_value(session, kind, resource, field, "pt")).needs_review
        assert getattr(resource, field) == "Самооценка"
        await session.commit()


@pytest.mark.asyncio
async def test_imported_values_are_read_without_rewriting(factory):
    async with factory() as session:
        session.add(Topic(id=17, name="Отношения"))
        for locale, value in (("en", "Relationships"), ("pt", "Relacionamentos")):
            session.add(BotTranslation(locale=locale, translation_key="topic.17.name", text=value, source_hash=source_hash("Отношения")))
        await session.commit()
        topic = await session.get(Topic, 17)
        assert (await read_content_value(session, "topic", topic, "name", "en")).text == "Relationships"
        await save_content_value(session, "topic", topic, "name", "pt", "Relacionamentos e intimidade")
        assert topic.name == "Отношения"
        assert (await read_content_value(session, "topic", topic, "name", "en")).text == "Relationships"


def test_dynamic_fallback_never_uses_a_third_language():
    snapshot = TranslationSnapshot(1, {"topic.17.name": ""}, {("pt", "topic.17.name"): "Autoestima"})
    assert snapshot.get("topic.17.name", "ru", "pt") is None
    assert snapshot.get("topic.17.name", "en", "pt") is None
    assert snapshot.get("topic.17.name", "pt") == "Autoestima"


@pytest.mark.asyncio
async def test_answer_aliases_survive_insert_delete_and_reorder(factory):
    async with factory() as session:
        question = Question(id=17, text="Вопрос", category="custom", answer_options_json=json.dumps([{"text": "Первый", "value": 3}, {"text": "Второй", "value": 7}]))
        session.add(question)
        for locale in ("en", "pt"):
            for index in (0, 1):
                session.add(BotTranslation(locale=locale, translation_key=f"test_question.17.option.{index}.text", text=f"{locale}-{index}", source_hash=source_hash(("Первый", "Второй")[index])))
        await session.commit()
        original = await ensure_answer_identities(session, question)
        added = await insert_answer(session, question, 1, value=9)
        await reorder_answers(session, question, [original[1]["identity"], added["identity"], original[0]["identity"]])
        options = get_answer_options(question)
        assert [item.callback_id for item in options] == [1, 2, 0]
        assert make_option_answer_record(question, 0, "test_opt_0_0")["numeric_value"] == 3
        assert make_option_answer_record(question, 0, "test_opt_0_1")["numeric_value"] == 7
        for locale in ("en", "pt"):
            assert (await read_content_value(session, "test_question", question, "option.1.text", locale)).text == f"{locale}-1"
            assert (await read_content_value(session, "test_question", question, "option.0.text", locale)).text == f"{locale}-0"
            assert (await read_content_value(session, "test_question", question, f"option.{added['translation_slot']}.text", locale)).missing
        await delete_answer(session, question, original[0]["identity"])
        with pytest.raises(ValueError):
            make_option_answer_record(question, 0, "test_opt_0_0")
        next_added = await insert_answer(session, question, 0)
        assert next_added["callback_id"] > added["callback_id"]
        assert next_added["translation_slot"] not in {"0", "1", added["translation_slot"]}
        assert (await session.scalars(select(BotTranslation))).all().__len__() == 4


@pytest.mark.asyncio
async def test_question_identity_survives_order_insert_and_delete(factory):
    async with factory() as session:
        first = await create_resource(session, "test_question", "ru", {"text": "Первый"}, {"sort_order": 0})
        last = await create_resource(session, "test_question", "ru", {"text": "Последний"}, {"sort_order": 2})
        for locale in ("en", "pt"):
            await save_content_value(session, "test_question", first, "text", locale, locale + " first")
            await save_content_value(session, "test_question", last, "text", locale, locale + " last")
        middle = await create_resource(session, "test_question", "pt", {"text": "Meio"}, {"sort_order": 1})
        first.sort_order, last.sort_order = 3, 0
        for locale in ("en", "pt"):
            assert (await read_content_value(session, "test_question", first, "text", locale)).text == locale + " first"
            assert (await read_content_value(session, "test_question", last, "text", locale)).text == locale + " last"
        removed_id = middle.id
        await session.delete(middle)
        await session.flush()
        new = await create_resource(session, "test_question", "pt", {"text": "Novo"})
        assert new.id != removed_id


@pytest.mark.asyncio
async def test_system_pack_skips_legacy_dynamic_entries(factory):
    registry = TranslationRegistry([TranslationSource("ui.example", "Пример"), TranslationSource("topic.17.name", "Тема")])
    async with factory() as session:
        session.add(BotTranslation(locale="pt", translation_key="topic.17.name", text="Novo", source_hash=source_hash("Тема")))
        await session.commit()
        pack = await export_translation_pack(session, registry, locale="pt")
        assert [row["translation_key"] for row in pack["translations"]] == ["ui.example"]
    pack["translations"][0]["text"] = "Exemplo"
    pack["translations"].append({"locale": "pt", "translation_key": "topic.17.name", "text": "Antigo", "source_hash": "obsolete"})
    assert await import_translation_pack(factory, pack, registry=registry) == 1
    async with factory() as session:
        assert await session.scalar(select(BotTranslation.text).where(BotTranslation.translation_key == "topic.17.name")) == "Novo"
        assert (await audit_translation_readiness(session, registry, locales=("pt",)))["ready"]


@pytest.mark.asyncio
async def test_existing_master_flush_and_commit_do_not_write_portuguese_into_russian(factory):
    from admin_authoring_context import content_editing_locale
    from database import AsyncSession, Mailing
    custom_factory = async_sessionmaker(factory.kw["bind"], class_=AsyncSession, expire_on_commit=False)
    async with custom_factory() as session:
        session.add(Topic(id=8, name="Русское имя"))
        await session.commit()
    token = content_editing_locale.set("pt")
    try:
        async with custom_factory() as session:
            topic = await session.get(Topic, 8)
            topic.name = "Nome"
            await session.flush()
            topic.sort_order = 9
            await session.commit()
        async with custom_factory() as session:
            mailing = Mailing(text="Mensagem", target_audience="all", status="draft")
            session.add(mailing)
            await session.commit()
            assert mailing.text == ""
            assert (await read_content_value(session, "mailing", mailing, "text", "pt")).text == "Mensagem"
    finally:
        content_editing_locale.reset(token)
    async with custom_factory() as session:
        topic = await session.get(Topic, 8)
        assert topic.name == "Русское имя"
        assert topic.sort_order == 9
        assert (await read_content_value(session, "topic", topic, "name", "pt")).text == "Nome"


@pytest.mark.asyncio
async def test_active_session_keeps_question_and_answer_callbacks_after_structural_changes(factory):
    from database import TestSession as Session, User
    from test_content_identity import move_question, preserve_active_test_definitions, questions_for_session
    async with factory() as session:
        session.add(User(id=19))
        first = await create_resource(session, "test_question", "ru", {"text": "Первый"}, {"sort_order": 0, "answer_options_json": '[{"text":"Да","value":4},{"text":"Нет","value":0}]'})
        second = await create_resource(session, "test_question", "ru", {"text": "Второй"}, {"sort_order": 1})
        active = Session(user_id=19, current_question_index=0)
        session.add(active)
        await session.commit()
        await preserve_active_test_definitions(session)
        await move_question(session, second.id, -1)
        old_options = await ensure_answer_identities(session, first)
        await insert_answer(session, first, 0, value=99)
        await delete_answer(session, first, old_options[0]["identity"])
        await session.delete(second)
        await session.commit()
        snapshot = questions_for_session(active, [first])
        assert [item.id for item in snapshot] == [first.id, second.id]
        assert make_option_answer_record(snapshot[0], 0, "test_opt_0_0")["numeric_value"] == 4
        assert make_option_answer_record(snapshot[0], 0, "test_opt_0_1")["numeric_value"] == 0


@pytest.mark.asyncio
async def test_menu_labels_cannot_rebind_a_deleted_resource(factory):
    from content_menu import remember_menu, resolve_menu
    await remember_menu(19, [("Tema", "topic", 17)], session_maker=factory)
    assert await resolve_menu(19, "Tema", "topic", session_maker=factory) == "17"
    await remember_menu(19, [("Tema", "topic", 18)], session_maker=factory)
    assert await resolve_menu(19, "Tema", "topic", session_maker=factory) is None


@pytest.mark.asyncio
async def test_question_import_never_replaces_existing_ids_without_explicit_identity(factory):
    from question_authoring_import import apply_question_import
    async with factory() as session:
        question = await create_resource(session, "test_question", "ru", {"text": "Вопрос"})
        await session.commit()
        with pytest.raises(ValueError, match="ID"):
            await apply_question_import(session, [{"text": "Pergunta"}], "pt")
        await apply_question_import(session, [{"id": question.id, "text": "Pergunta"}], "pt")
        assert question.text == "Вопрос"
        assert (await read_content_value(session, "test_question", question, "text", "pt")).text == "Pergunta"


@pytest.mark.asyncio
async def test_runtime_retains_dynamic_review_values_and_refreshes_revision(factory):
    from translation_service import refresh_translation_cache, translation_cache
    async with factory() as session:
        topic = await create_resource(session, "topic", "ru", {"name": "Тема"})
        await save_content_value(session, "topic", topic, "name", "en", "Topic")
        await session.commit()
        key = f"topic.{topic.id}.name"
    first = await refresh_translation_cache(factory, force=True)
    async with factory() as session:
        topic = await session.get(Topic, topic.id)
        await save_content_value(session, "topic", topic, "name", "ru", "Новая тема")
        await session.commit()
    second = await refresh_translation_cache(factory, force=True)
    assert second.revision > first.revision
    assert second.get(key, "en") == "Topic"
    assert second.get(key, "ru") == "Новая тема"
    translation_cache.install(0, {}, {})


@pytest.mark.asyncio
async def test_telegram_authoring_workflow_uses_object_local_values(factory, monkeypatch):
    import importlib
    import sys
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock
    from aiogram import Bot, Dispatcher
    from aiogram.client.session.base import BaseSession
    from aiogram.types import CallbackQuery, Chat, Message, Update, User
    from aiogram.fsm.storage.memory import MemoryStorage
    module = importlib.import_module("admin_content_authoring")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setitem(sys.modules, "handlers", SimpleNamespace(is_admin=AsyncMock(return_value=True)))

    class RecordingSession(BaseSession):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def close(self):
            pass

        async def make_request(self, bot, method, timeout=None):
            self.calls.append(method)
            if hasattr(method, "text"):
                return Message(message_id=len(self.calls), date=datetime.now(timezone.utc), chat=Chat(id=11, type="private"), text=method.text)
            return True

        async def stream_content(self, *args, **kwargs):
            if False:
                yield b""

    session = RecordingSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(module.router)
    user = User(id=11, is_bot=False, first_name="Администратор")
    incoming = Message(message_id=100, date=datetime.now(timezone.utc), chat=Chat(id=11, type="private"), from_user=user, text="Экран")
    update_id = 0

    async def click(data):
        nonlocal update_id
        update_id += 1
        await dispatcher.feed_update(bot, Update(update_id=update_id, callback_query=CallbackQuery(id=str(update_id), from_user=user, chat_instance="test", message=incoming, data=data)))

    try:
        async with factory() as db:
            config = await db.get(BotGeneralConfig, 1)
            config.multilingual_authoring_enabled = True
            await db.commit()
        await click("ca:new:topic")
        update_id += 1
        await dispatcher.feed_update(bot, Update(update_id=update_id, message=incoming.model_copy(update={"text": "Autoestima"})))
        async with factory() as db:
            topic = await db.scalar(select(Topic))
            assert topic.name == "Autoestima"
        await click(f"ca:view:topic:{topic.id}")
        card = next(call for call in reversed(session.calls) if getattr(call, "text", None))
        assert "Autoestima" in card.text and "Название" in card.text
        callback_data = [button.callback_data for row in card.reply_markup.inline_keyboard for button in row]
        assert f"ca:locale:topic:{topic.id}:pt" in callback_data
        for row in card.reply_markup.inline_keyboard:
            for button in row:
                assert len(button.callback_data.encode()) <= 64
        await click(f"ca:locale:topic:{topic.id}:pt")
        card = next(call for call in reversed(session.calls) if getattr(call, "text", None))
        assert "Перевод не задан" in card.text
        await click(f"ca:edit:topic:{topic.id}:pt:0")
        update_id += 1
        await dispatcher.feed_update(bot, Update(update_id=update_id, message=incoming.model_copy(update={"text": "Autoestima PT"})))
        async with factory() as db:
            same = await db.get(Topic, topic.id)
            assert same.name == "Autoestima"
            assert (await read_content_value(db, "topic", same, "name", "pt")).text == "Autoestima PT"
        await click(f"ca:locale:topic:{topic.id}:en")
        await click(f"ca:edit:topic:{topic.id}:en:0")
        update_id += 1
        await dispatcher.feed_update(bot, Update(update_id=update_id, message=incoming.model_copy(update={"text": "Self-esteem"})))
        async with factory() as db:
            same = await db.get(Topic, topic.id)
            assert same.name == "Autoestima"
            assert (await read_content_value(db, "topic", same, "name", "en")).text == "Self-esteem"
        workspace = dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11)
        assert await workspace.get_state() is None
    finally:
        await dispatcher.storage.close()
        await bot.session.close()
