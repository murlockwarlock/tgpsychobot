from __future__ import annotations

import html
import re
import math

from aiogram import BaseMiddleware, F, Router
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import select

from content_authoring import (
    RESOURCES, create_resource, editing_locale, read_content_value, field_source,
    save_content_value, set_editing_locale,
)
from database import async_session_maker
from translation_pack_manager import translation_coordination_lock
from translation_service import LOCALE_LABELS, SUPPORTED_TELEGRAM_LOCALES, refresh_translation_cache
from universal_tests import get_answer_options


router = Router(name="content_authoring")


class ContentAuthoringStates(StatesGroup):
    value = State()
    answer_value = State()


def keyboard(rows):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=text, callback_data=callback) for text, callback in row]
        for row in rows
    ])


def locale_heading(locale):
    return f"✍️ Язык контента: {LOCALE_LABELS.get(locale, locale)}"


async def locale_for(event):
    async with async_session_maker() as session:
        return await editing_locale(session, event.bot.id, event.from_user.id)


async def allowed(event):
    from handlers import is_admin
    if await is_admin(event.from_user.id):
        return True
    if isinstance(event, CallbackQuery):
        await event.answer("Недостаточно прав.", show_alert=True)
    return False


async def show_root(event, state=None):
    if not await allowed(event):
        return
    if state is not None:
        await state.clear()
    import keyboards as kb
    markup = kb.admin_panel_keyboard()
    markup.inline_keyboard.insert(0, [InlineKeyboardButton(text="🌐 Сменить язык контента", callback_data="ca:language")])
    text = "Добро пожаловать в админ-панель!\n\n" + locale_heading(await locale_for(event))
    if isinstance(event, CallbackQuery):
        await event.message.edit_text(text, reply_markup=markup)
        await event.answer()
    else:
        await event.answer(text, reply_markup=markup)


@router.message(Command("admin"))
async def root_message(message: Message, state):
    await show_root(message, state)


@router.callback_query(F.data == "admin_panel")
async def root_callback(callback: CallbackQuery, state):
    await show_root(callback, state)


async def resource_list(event, kind, page=0):
    spec = RESOURCES[kind]
    locale = await locale_for(event)
    async with async_session_maker() as session:
        query = select(spec.model).order_by(getattr(spec.model, spec.identity_field))
        if hasattr(spec.model, "sort_order"):
            query = select(spec.model).order_by(spec.model.sort_order, getattr(spec.model, spec.identity_field))
        if kind == "automation_action":
            query = query.where(spec.model.action_type == "send_message", ~spec.model.recipient_type.contains("admin"))
        if kind == "followup_step":
            query = query.where(spec.model.message_type == "static")
        resources = (await session.scalars(query.offset(page * 8).limit(9))).all()
        rows = []
        for resource in resources[:8]:
            identity = getattr(resource, spec.identity_field)
            value = await read_content_value(session, kind, resource, spec.fields[0][0], locale)
            label = value.admin_label().replace("\n", " ")
            row = [(f"{identity}: {label}"[:64], f"ca:view:{kind}:{identity}")]
            if kind == "topic":
                row.extend([("Выше", f"move_topic_up_{identity}_{page}"), ("Ниже", f"move_topic_down_{identity}_{page}")])
            rows.append(row)
        if kind == "topic":
            from database import SubscriptionConfig
            config = await session.get(SubscriptionConfig, 1)
            if config:
                value = await read_content_value(session, "subscription_config", config, "topics_btn_name", locale)
                rows.extend([
                    [("Темы: " + ("включены" if config.topics_enabled else "выключены"), "admin_toggle_topics")],
                    [("Кнопка тем: " + ("сверху" if config.topics_btn_on_top else "в списке"), "admin_toggle_topics_on_top")],
                    [("Название кнопки: " + value.admin_label()[:40], "admin_rename_topics_btn")],
                ])
    nav = []
    if page:
        nav.append(("Назад", f"ca:list:{kind}:{page - 1}"))
    if len(resources) > 8:
        nav.append(("Далее", f"ca:list:{kind}:{page + 1}"))
    if nav:
        rows.append(nav)
    if kind in {"topic", "content", "plan", "test_question", "secret_test_question", "referral_template", "case_study"}:
        rows.append([("Добавить", f"ca:new:{kind}")])
    rows.extend([[('🌐 Сменить язык контента', 'ca:language')], [('В админ-панель', 'admin_panel')]])
    await event.message.edit_text(f"{locale_heading(locale)}\n\n<b>{spec.title}</b>", reply_markup=keyboard(rows))


async def resource_card(event, kind, identity):
    spec = RESOURCES[kind]
    identity = identity if spec.identity_field == "key" else int(identity)
    locale = await locale_for(event)
    async with async_session_maker() as session:
        resource = await session.get(spec.model, identity)
        if resource is None:
            await event.answer("Материал не найден.", show_alert=True)
            return
        fields = list(spec.fields)
        text = [locale_heading(locale), f"<b>{spec.title} #{html.escape(str(identity))}</b>"]
        rows = []
        for field, title, _ in fields:
            value = await read_content_value(session, kind, resource, field, locale)
            preview = value.admin_label()
            if len(preview) > 250:
                preview = preview[:247] + "…"
            text.append(f"\n<b>{title}:</b>\n{html.escape(preview)}")
            if value.needs_review:
                text.append("⚠️ Требует проверки после изменения русского текста")
            if field.startswith("option."):
                _, slot, part = field.split(".")
                option = next(option for option in get_answer_options(resource) if option.translation_slot == slot)
                route = f"ca:opt:{identity}:{option.callback_id}:{part}"
            else:
                route = f"ca:edit:{kind}:{identity}:{fields.index((field, title, _))}"
            rows.append([(f"Изменить: {title}", route)])
        globals_markup = None
        import keyboards as kb
        if kind == "topic":
            globals_markup = kb.edit_topic_keyboard(identity, resource.is_active, in_menu=resource.show_in_main_menu, in_list=resource.show_in_list, admin_only=resource.admin_only, auto_start=resource.auto_start_dialogue)
        elif kind == "plan":
            rows.append([("Общие настройки тарифа", f"ca:settings:plan:{identity}")])
        elif kind == "content":
            rows.append([("Медиа и общие настройки", f"ca:settings:content:{identity}")])
        elif kind == "test_question":
            rows.append([("Варианты и порядок", f"ca:answers:{identity}")])
        elif kind == "referral_template":
            globals_markup = kb.admin_referral_template_detail_keyboard(identity, resource.is_enabled)
        elif kind == "secret_test_question":
            rows.append([("Удалить для всех языков", f"delete_secret_q_{identity}")])
        elif kind == "case_study":
            rows.append([("Удалить для всех языков", f"delete_case_{identity}")])
        if globals_markup:
            for row in globals_markup.inline_keyboard:
                filtered = [(button.text, button.callback_data) for button in row if not button.callback_data.startswith(("edit_topic_name_", "edit_topic_intro_", "edit_topic_btn_text_", "admin_ref_tpl_edit_"))]
                if filtered:
                    rows.append(filtered)
        rows.extend([[('🌐 Сменить язык контента', 'ca:language')], [('К списку', f'ca:list:{kind}:0')]])
    await event.message.edit_text("\n".join(text), reply_markup=keyboard(rows))


def resource_fields(kind, resource):
    fields = list(RESOURCES[kind].fields)
    if kind == "test_question":
        for index, option in enumerate(get_answer_options(resource)):
            fields.extend([
                (f"option.{option.translation_slot}.text", f"Ответ {index + 1}", "text"),
                (f"option.{option.translation_slot}.button_text", f"Кнопка ответа {index + 1}", "inline_button"),
            ])
    return fields


async def begin_edit(event, state, kind, identity, field_index):
    locale = await locale_for(event)
    spec = RESOURCES[kind]
    async with async_session_maker() as session:
        resource = await session.get(spec.model, identity if spec.identity_field == "key" else int(identity))
        if resource is None:
            await event.answer("Материал не найден.", show_alert=True)
            return
        fields = resource_fields(kind, resource)
        if isinstance(field_index, str):
            selected = next((item for item in fields if item[0] == field_index), None)
            if selected is None:
                await event.answer("Поле удалено. Откройте материал заново.", show_alert=True)
                return
        else:
            selected = fields[field_index]
        field, title, _ = selected
        value = await read_content_value(session, kind, resource, field, locale)
    await state.set_state(ContentAuthoringStates.value)
    await state.set_data({"kind": kind, "identity": identity, "field": field, "locale": locale, "source_hash": field_source(kind, resource, field).source_hash})
    rows = []
    if value.needs_review:
        rows.append([("Подтвердить перевод", "ca:confirm")])
    rows.append([("Отмена", f"ca:view:{kind}:{identity}")])
    reference = f"\n\nРусский исходник:\n{html.escape((value.russian or 'Не задан')[:1000])}" if locale != "ru" else ""
    await event.message.edit_text(f"{locale_heading(locale)}\n\n{title}:\n{html.escape(value.admin_label()[:1500])}{reference}\n\nВведите новое значение.", reply_markup=keyboard(rows))


@router.callback_query(F.data.startswith("ca:"))
async def content_callback(callback: CallbackQuery, state):
    if not await allowed(callback):
        return
    parts = callback.data.split(":")
    action = parts[1]
    if action == "language":
        await state.clear()
        await callback.message.edit_text(locale_heading(await locale_for(callback)), reply_markup=keyboard(
            [[(LOCALE_LABELS.get(locale, locale), f"ca:locale:{locale}")] for locale in SUPPORTED_TELEGRAM_LOCALES]
            + [[("Назад", "admin_panel")]],
        ))
    elif action == "locale":
        async with async_session_maker() as session:
            await set_editing_locale(session, callback.bot.id, callback.from_user.id, parts[2])
            await session.commit()
        await show_root(callback, state)
        return
    elif action == "list":
        await state.clear()
        await resource_list(callback, parts[2], int(parts[3]))
    elif action == "view":
        await state.clear()
        await resource_card(callback, parts[2], parts[3])
    elif action == "edit":
        await begin_edit(callback, state, parts[2], parts[3], int(parts[4]))
    elif action == "opt":
        async with async_session_maker() as session:
            question = await session.get(RESOURCES["test_question"].model, int(parts[2]))
            option = next((option for option in get_answer_options(question) if option.callback_id == int(parts[3])), None)
            if option is None:
                await callback.answer("Вариант удалён.", show_alert=True)
                return
            field = f"option.{option.translation_slot}.{parts[4]}"
        await begin_edit(callback, state, "test_question", parts[2], field)
    elif action == "answers":
        await state.clear()
        await show_answers(callback, int(parts[2]), int(parts[3]) if len(parts) > 3 else 0)
    elif action in {"answer_add", "answer_up", "answer_delete", "question_up", "question_down", "question_delete"}:
        await change_structure(callback, state, parts)
        return
    elif action == "new":
        kind = parts[2]
        locale = await locale_for(callback)
        await state.set_state(ContentAuthoringStates.value)
        await state.set_data({"kind": kind, "identity": None, "field": RESOURCES[kind].fields[0][0], "locale": locale})
        await callback.message.edit_text(f"{locale_heading(locale)}\n\n{RESOURCES[kind].fields[0][1]}:", reply_markup=keyboard([[('Отмена', f'ca:list:{kind}:0')]]))
    elif action == "settings":
        from handlers import _show_admin_edit_plan_menu, start_content_edit
        from admin_authoring_context import content_editing_locale
        locale = await locale_for(callback)
        token = content_editing_locale.set(locale)
        try:
            if parts[2] == "plan":
                await _show_admin_edit_plan_menu(callback.bot, callback.message.chat.id, callback.message.message_id, int(parts[3]))
            elif parts[2] == "content":
                await start_content_edit(callback.model_copy(update={"data": "edit_content_" + parts[3]}), state)
            if await state.get_state():
                await state.update_data(authoring_locale=locale)
        finally:
            content_editing_locale.reset(token)
    elif action == "confirm":
        data = await state.get_data()
        if not data.get("identity"):
            await callback.answer("Откройте материал заново.")
            return
        spec = RESOURCES[data["kind"]]
        async with async_session_maker() as session:
            async with translation_coordination_lock(session):
                resource = await session.get(spec.model, data["identity"] if spec.identity_field == "key" else int(data["identity"]))
                if resource is None or field_source(data["kind"], resource, data["field"]).source_hash != data.get("source_hash"):
                    await callback.answer("Русский текст изменился. Откройте материал заново для проверки.", show_alert=True)
                    return
                value = await read_content_value(session, data["kind"], resource, data["field"], data["locale"])
                if value.text:
                    await save_content_value(session, data["kind"], resource, data["field"], data["locale"], value.text)
                    await session.commit()
        await state.clear()
        await resource_card(callback, data["kind"], data["identity"])
    await callback.answer()


@router.message(ContentAuthoringStates.value, F.text)
async def content_value_received(message: Message, state):
    if not await allowed(message):
        return
    data = await state.get_data()
    field_kind = next((kind for field, _, kind in RESOURCES[data["kind"]].fields if field == data["field"]), "text")
    value = message.html_text if field_kind in {"html", "caption"} and data["kind"] != "test_question" else message.text
    if data["kind"] == "bot_general_config":
        if data["locale"] == "ru":
            from handlers import serialize_ai_processing_message_text
            try:
                value = serialize_ai_processing_message_text(message.text, message.entities)
            except ValueError as exc:
                await message.answer(str(exc))
                return
        else:
            value = message.html_text
    if not value.strip():
        await message.answer("Значение не может быть пустым.")
        return
    try:
        async with async_session_maker() as session:
            async with translation_coordination_lock(session):
                spec = RESOURCES[data["kind"]]
                identity = data["identity"]
                if identity is None:
                    resource = await create_resource(session, data["kind"], data["locale"], {data["field"]: value})
                    identity = getattr(resource, spec.identity_field)
                else:
                    resource = await session.get(spec.model, identity if spec.identity_field == "key" else int(identity))
                    if resource is None:
                        raise ValueError("Материал удалён. Вернитесь к списку.")
                    if data["locale"] != "ru" and data.get("source_hash") != field_source(data["kind"], resource, data["field"]).source_hash:
                        raise ValueError("Русский текст изменился. Откройте материал заново для проверки.")
                    await save_content_value(session, data["kind"], resource, data["field"], data["locale"], value)
                await session.commit()
    except ValueError as exc:
        await message.answer(html.escape(str(exc)))
        return
    if data["kind"] == "case_study" and data["locale"] == "ru":
        from vector_store import update_case_study_index
        await update_case_study_index(int(identity), value)
    await state.clear()
    await refresh_translation_cache(async_session_maker, force=True)
    await message.answer(f"{locale_heading(data['locale'])}\n\nСохранено.", reply_markup=keyboard([[('Открыть материал', f"ca:view:{data['kind']}:{identity}")]]))


ENTRY_LISTS = {"admin_content": "content", "admin_plans": "plan", "admin_secret_questions": "secret_test_question", "admin_test_questions": "test_question", "admin_referral_templates": "referral_template"}


@router.callback_query(lambda event: event.data in ENTRY_LISTS or bool(re.fullmatch(r"admin_(?:topics|case_studies)_page_\d+", event.data or "")))
async def list_entry(callback: CallbackQuery, state):
    if not await allowed(callback):
        return
    await state.clear()
    paginated = callback.data.startswith(("admin_topics_page_", "admin_case_studies_page_"))
    kind = "case_study" if callback.data.startswith("admin_case_studies_page_") else ENTRY_LISTS.get(callback.data, "topic")
    await resource_list(callback, kind, int(callback.data.rsplit("_", 1)[1]) if paginated else 0)
    await callback.answer()


DETAIL_ROUTES = ((r"edit_topic_(\d+)", "topic"), (r"admin_edit_plan_(\d+)", "plan"), (r"edit_content_(?!btn_)(.+)", "content"))


@router.callback_query(lambda event: any(re.fullmatch(pattern, event.data or "") for pattern, _ in DETAIL_ROUTES))
async def detail_entry(callback: CallbackQuery, state):
    if not await allowed(callback):
        return
    await state.clear()
    for pattern, kind in DETAIL_ROUTES:
        match = re.fullmatch(pattern, callback.data)
        if match:
            await resource_card(callback, kind, match[1])
            break
    await callback.answer()


FIELD_ROUTES = (
    (r"edit_topic_name_(\d+)", "topic", 0),
    (r"edit_topic_intro_(\d+)", "topic", 2),
    (r"edit_topic_btn_text_(\d+)", "topic", 3),
    (r"edit_plan_field_name_(\d+)", "plan", 0),
    (r"edit_plan_field_description_(\d+)", "plan", 1),
    (r"admin_ref_tpl_edit_(\d+)", "referral_template", 0),
    (r"admin_media_edit_desc_(\d+)", "media_library", 0),
    (r"admin_coll_media_editdesc_(\d+)_\d+_\d+_\d+", "media_library", 0),
)
SINGLE_FIELDS = {
    "admin_general_edit_ai_processing_message_text": ("bot_general_config", 0),
    "admin_rename_topics_btn": ("subscription_config", 0),
    "admin_referral_set_btn_name": ("subscription_config", 1),
    "admin_referral_set_sub_btn_name": ("subscription_config", 2),
}


@router.callback_query(lambda event: event.data in SINGLE_FIELDS or any(re.fullmatch(pattern, event.data or "") for pattern, _, _ in FIELD_ROUTES))
async def field_entry(callback: CallbackQuery, state):
    if not await allowed(callback):
        return
    if callback.data in SINGLE_FIELDS:
        kind, index = SINGLE_FIELDS[callback.data]
        await begin_edit(callback, state, kind, "1", index)
    else:
        for pattern, kind, index in FIELD_ROUTES:
            match = re.fullmatch(pattern, callback.data)
            if match:
                await begin_edit(callback, state, kind, match[1], index)
                break
    await callback.answer()


async def show_answers(callback, question_id, page=0):
    async with async_session_maker() as session:
        question = await session.get(RESOURCES["test_question"].model, question_id)
        if question is None:
            await callback.answer("Вопрос удалён. Обновите список.", show_alert=True)
            return
        locale = await locale_for(callback)
        rows = []
        options = get_answer_options(question)
        for option in options[page * 8:(page + 1) * 8]:
            value = await read_content_value(session, "test_question", question, f"option.{option.translation_slot}.text", locale)
            rows.append([(value.admin_label()[:64], f"ca:opt:{question_id}:{option.callback_id}:text")])
            rows.append([("Подпись кнопки", f"ca:opt:{question_id}:{option.callback_id}:button_text")])
            rows.append([("Выше", f"ca:answer_up:{question_id}:{option.callback_id}"), ("Удалить", f"ca:answer_delete:{question_id}:{option.callback_id}")])
        navigation = []
        if page:
            navigation.append(("Назад", f"ca:answers:{question_id}:{page - 1}"))
        if len(options) > (page + 1) * 8:
            navigation.append(("Далее", f"ca:answers:{question_id}:{page + 1}"))
        if navigation:
            rows.append(navigation)
        rows.extend([
            [("Добавить ответ", f"ca:answer_add:{question_id}")],
            [("Вопрос выше", f"ca:question_up:{question_id}"), ("Вопрос ниже", f"ca:question_down:{question_id}")],
            [("Удалить вопрос", f"ca:question_delete:{question_id}")],
            [("Назад", f"ca:view:test_question:{question_id}")],
        ])
    await callback.message.edit_text(f"{locale_heading(locale)}\n\nВопрос #{question_id}\nПорядок, значения и удаление общие для всех языков.", reply_markup=keyboard(rows))


async def change_structure(callback, state, parts):
    from content_authoring import delete_answer, ensure_answer_identities, reorder_answers, bump_revision
    from test_content_identity import move_question, preserve_active_test_definitions
    action, question_id = parts[1], int(parts[2])
    if action == "answer_add":
        await state.set_state(ContentAuthoringStates.answer_value)
        await state.set_data({"question_id": question_id, "locale": await locale_for(callback)})
        await callback.message.edit_text("Введите подпись ответа в первой строке, числовое значение во второй (или — для ответа без балла).", reply_markup=keyboard([[('Отмена', f'ca:answers:{question_id}')]]))
        await callback.answer()
        return
    if action.endswith("delete") and (len(parts) < 4 or parts[-1] != "confirmed"):
        await callback.message.edit_text("Удалить для всех языков? Существующие прохождения сохранят свои вопросы и ответы.", reply_markup=keyboard([[('Удалить', callback.data + ':confirmed')], [('Отмена', f'ca:answers:{question_id}')]]))
        await callback.answer()
        return
    async with async_session_maker() as session:
        async with translation_coordination_lock(session):
            question = await session.get(RESOURCES["test_question"].model, question_id)
            if question is None:
                await callback.answer("Вопрос удалён. Обновите список.", show_alert=True)
                return
            if action == "question_delete":
                await preserve_active_test_definitions(session)
                await session.delete(question)
            elif action.startswith("question_"):
                await move_question(session, question_id, -1 if action == "question_up" else 1)
            else:
                items = await ensure_answer_identities(session, question)
                index = next((index for index, item in enumerate(items) if item["callback_id"] == int(parts[3])), None)
                if index is None:
                    await callback.answer("Вариант удалён. Обновите вопрос.", show_alert=True)
                    return
                if action == "answer_delete":
                    await delete_answer(session, question, items[index]["identity"])
                elif index:
                    items[index - 1], items[index] = items[index], items[index - 1]
                    await reorder_answers(session, question, [item["identity"] for item in items])
            await bump_revision(session)
            await session.commit()
    if action == "question_delete":
        await resource_list(callback, "test_question")
    else:
        await show_answers(callback, question_id)
    await callback.answer()


@router.message(ContentAuthoringStates.answer_value, F.text)
async def new_answer_received(message: Message, state):
    if not await allowed(message):
        return
    from content_authoring import insert_answer
    label, separator, score = message.text.partition("\n")
    try:
        if not separator or not label.strip():
            raise ValueError()
        value = None if score.strip() == "—" else float(score.strip())
        if value is not None and not math.isfinite(value):
            raise ValueError()
    except ValueError:
        await message.answer("Нужны подпись и число на отдельных строках. Без балла: —")
        return
    data = await state.get_data()
    try:
        async with async_session_maker() as session:
            async with translation_coordination_lock(session):
                question = await session.get(RESOURCES["test_question"].model, data["question_id"])
                if question is None:
                    raise ValueError("Вопрос удалён. Вернитесь к списку.")
                item = await insert_answer(session, question, len(get_answer_options(question)), value=value)
                await save_content_value(session, "test_question", question, f"option.{item['translation_slot']}.text", data["locale"], label)
                await session.commit()
    except ValueError as exc:
        await message.answer(html.escape(str(exc)))
        return
    await state.clear()
    await message.answer("Ответ добавлен.", reply_markup=keyboard([[('К вопросу', f"ca:view:test_question:{data['question_id']}")]]))
