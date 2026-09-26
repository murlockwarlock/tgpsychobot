from __future__ import annotations

import html
import re
import math

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import delete, func, or_, select

from content_authoring import (
    AUTHORING_RESOURCES, RESOURCES, authoring_locales, create_resource,
    multilingual_authoring_enabled, read_content_value,
    save_content_value,
)
from database import (
    AutomationAction,
    BotTranslation,
    Content,
    ContentMedia,
    FollowupStep,
    Mailing,
    ReferralTemplate,
    Topic,
    UserMenuBinding,
    async_session_maker,
)
from translation_pack_manager import translation_coordination_lock
from translation_service import LOCALE_LABELS, refresh_translation_cache
from universal_tests import get_answer_options


router = Router(name="content_authoring")


class ContentAuthoringStates(StatesGroup):
    value = State()
    answer_value = State()


PAGE_SIZE = 15
CONTENT_SYSTEM_KEYS = {"test_button", "test_intro", "test_results", "secret_test_outro"}
CONTENT_SPECIAL_TITLES = {
    "start_message": "Приветствие (/start)",
    "menu": "Меню",
    "disclaimer": "Дисклеймер",
}


def keyboard(rows):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=text, callback_data=callback) for text, callback in row]
        for row in rows
    ])


def locale_heading(locale):
    return f"Язык: {LOCALE_LABELS.get(locale, locale)}"


async def locale_for(event):
    return "ru"


def is_localized_resource(kind):
    return kind in AUTHORING_RESOURCES


def _content_list_predicate():
    return or_(
        Content.button_title.is_not(None),
        Content.key.in_(tuple(CONTENT_SPECIAL_TITLES)),
    )


def _content_list_query():
    return select(Content).where(
        _content_list_predicate(),
        Content.key.not_in(tuple(CONTENT_SYSTEM_KEYS)),
    ).order_by(Content.sort_order, Content.key)


def _content_display_value(resource, value, locale):
    if locale == "ru":
        return value.text or CONTENT_SPECIAL_TITLES.get(resource.key) or "Не задано"
    return value.admin_label()


def _content_list_label(resource, value):
    identity = str(resource.key)
    special_title = CONTENT_SPECIAL_TITLES.get(identity)
    if special_title:
        return special_title
    title = (value.text or "Не задано").replace("\n", " ").strip()
    return title or "Не задано"


def _content_status_label(resource):
    return "✅ Виден пользователям" if resource.is_visible else "❌ Скрыт от пользователей"


def _content_locale_status(value, locale):
    if locale == "ru":
        return "✅ Русский" if value.text else "❌ Русский"
    return f"✅ {LOCALE_LABELS.get(locale, locale)}" if value.text else f"❌ {LOCALE_LABELS.get(locale, locale)}"


async def content_dependency_report(session, content_key: str) -> list[str]:
    report = []
    if content_key in CONTENT_SPECIAL_TITLES:
        report.append(
            f"встроенный раздел «{CONTENT_SPECIAL_TITLES[content_key]}» используется системным маршрутом"
        )
    menu_refs = await session.scalar(
        select(func.count()).select_from(UserMenuBinding).where(
            UserMenuBinding.resource_kind == "content",
            UserMenuBinding.resource_id == content_key,
        )
    )
    if menu_refs:
        report.append(f"{menu_refs} сохранённых пользовательских кнопок меню")

    pattern = f"%btn:svc:content:{content_key}%"
    canonical_refs = await session.scalar(
        select(func.count()).select_from(Content).where(
            Content.key != content_key,
            or_(
                Content.text_content.ilike(pattern),
                Content.action_btn_payload.ilike(pattern),
            ),
        )
    )
    translated_refs = await session.scalar(
        select(func.count()).select_from(BotTranslation).where(BotTranslation.text.ilike(pattern))
    )
    template_refs = 0
    for model, field in (
        (Topic, Topic.start_button_payload),
        (AutomationAction, AutomationAction.message_template),
        (FollowupStep, FollowupStep.message_text),
        (Mailing, Mailing.text),
        (ReferralTemplate, ReferralTemplate.text),
    ):
        template_refs += await session.scalar(
            select(func.count()).select_from(model).where(field.ilike(pattern))
        ) or 0
    if canonical_refs or translated_refs or template_refs:
        report.append(
            f"ссылки btn:svc:content:{content_key} в тексте ({canonical_refs + translated_refs + template_refs})"
        )

    deep_link_pattern = f"%?start={content_key}%"
    deep_link_refs = await session.scalar(
        select(func.count()).select_from(Content).where(Content.text_content.ilike(deep_link_pattern))
    )
    deep_link_refs += await session.scalar(
        select(func.count()).select_from(BotTranslation).where(BotTranslation.text.ilike(deep_link_pattern))
    ) or 0
    for model, field in (
        (Topic, Topic.description),
        (Topic, Topic.start_message),
        (AutomationAction, AutomationAction.message_template),
        (FollowupStep, FollowupStep.message_text),
        (Mailing, Mailing.text),
        (ReferralTemplate, ReferralTemplate.text),
    ):
        deep_link_refs += await session.scalar(
            select(func.count()).select_from(model).where(field.ilike(deep_link_pattern))
        ) or 0
    if deep_link_refs:
        report.append(f"прямые ссылки на {content_key} в контенте ({deep_link_refs})")

    return report


async def _delete_content_resource(session, content_key: str) -> None:
    from max_messenger_bot.storage import MaxContentMedia

    await session.execute(
        delete(BotTranslation).where(
            BotTranslation.translation_key.like(f"content.{content_key}.%")
        )
    )
    await session.execute(delete(ContentMedia).where(ContentMedia.content_key == content_key))
    await session.execute(delete(MaxContentMedia).where(MaxContentMedia.content_key == content_key))
    await session.execute(delete(Content).where(Content.key == content_key))


async def authoring_enabled_for(event):
    async with async_session_maker() as session:
        return await multilingual_authoring_enabled(session)


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
    if not await authoring_enabled_for(event):
        raise SkipHandler()
    if state is not None:
        await state.clear()
    import keyboards as kb
    markup = kb.admin_panel_keyboard()
    text = "Добро пожаловать в админ-панель!"
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


@router.callback_query(F.data == "admin_manage_buttons")
async def menu_management_card(callback: CallbackQuery, state):
    await state.clear()
    from handlers import _show_admin_manage_buttons
    await _show_admin_manage_buttons(
        bot=callback.bot,
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
    )
    await callback.answer()


@router.callback_query(F.data == "admin_menu_labels")
async def menu_labels_card(callback: CallbackQuery, state):
    await state.clear()
    await resource_card(callback, "subscription_config", "1")
    await callback.answer()


async def resource_list(event, kind, page=0):
    spec = AUTHORING_RESOURCES.get(kind, RESOURCES[kind])
    async with async_session_maker() as session:
        query = select(spec.model)
        if kind == "content":
            query = _content_list_query()
        else:
            query = query.order_by(getattr(spec.model, spec.identity_field))
        if hasattr(spec.model, "sort_order") and kind != "content":
            query = select(spec.model).order_by(spec.model.sort_order, getattr(spec.model, spec.identity_field))
        if kind == "automation_action":
            query = query.where(spec.model.action_type == "send_message", ~spec.model.recipient_type.contains("admin"))
        if kind == "followup_step":
            query = query.where(spec.model.message_type == "static")
        total = await session.scalar(select(func.count()).select_from(query.order_by(None).subquery())) or 0
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        page = max(0, min(int(page), total_pages - 1))
        resources = (await session.scalars(query.offset(page * PAGE_SIZE).limit(PAGE_SIZE + 1))).all()
        has_next = len(resources) > PAGE_SIZE
        resources = resources[:PAGE_SIZE]
        rows = []
        for resource in resources:
            identity = getattr(resource, spec.identity_field)
            value = await read_content_value(session, kind, resource, spec.fields[0][0], "ru")
            label = _content_list_label(resource, value) if kind == "content" else value.admin_label()
            label = label.replace("\n", " ")
            row = [(label[:64], f"ca:view:{kind}:{identity}:{page}")]
            if kind == "topic":
                row.extend([("Выше", f"move_topic_up_{identity}_{page}"), ("Ниже", f"move_topic_down_{identity}_{page}")])
            rows.append(row)
        if kind == "topic":
            from database import SubscriptionConfig
            config = await session.get(SubscriptionConfig, 1)
            if config:
                value = await read_content_value(session, "subscription_config", config, "topics_btn_name", "ru")
                rows.extend([
                    [("Темы: " + ("включены" if config.topics_enabled else "выключены"), "admin_toggle_topics")],
                    [("Кнопка тем: " + ("сверху" if config.topics_btn_on_top else "в списке"), "admin_toggle_topics_on_top")],
                    [("Название кнопки: " + value.admin_label()[:40], "admin_rename_topics_btn")],
                ])
    nav = []
    if page:
        nav.append(("⬅️ Назад", f"ca:list:{kind}:{page - 1}"))
    if has_next:
        nav.append(("Далее ➡️", f"ca:list:{kind}:{page + 1}"))
    if nav:
        rows.append(nav)
    if kind in {"topic", "content", "plan", "test_question", "secret_test_question", "referral_template", "case_study"}:
        rows.append([("Добавить", f"ca:new:{kind}")])
    rows.append([('В админ-панель', 'admin_panel')])
    page_suffix = f"\nСтраница {page + 1}/{total_pages}" if total_pages > 1 else ""
    await event.message.edit_text(f"<b>{spec.title}</b>{page_suffix}", reply_markup=keyboard(rows), parse_mode="HTML")


async def resource_card(event, kind, identity, locale="ru", page=0):
    spec = AUTHORING_RESOURCES.get(kind, RESOURCES[kind])
    identity = identity if spec.identity_field == "key" else int(identity)
    async with async_session_maker() as session:
        enabled = await multilingual_authoring_enabled(session)
        if not enabled:
            locale = "ru"
        localized = enabled and is_localized_resource(kind)
        locales = await authoring_locales(session) if localized else ("ru",)
        if locale not in locales:
            locale = "ru"
        resource = await session.get(spec.model, identity)
        if resource is None:
            await event.answer("Материал не найден.", show_alert=True)
            return
        fields = list(spec.fields)
        text = [f"<b>{spec.title} #{html.escape(str(identity))}</b>"]
        if kind == "content":
            text.append(f"ID: <code>{html.escape(str(identity))}</code>")
        if localized:
            text.append(locale_heading(locale))
        if kind == "content":
            text.append(f"Статус: {_content_status_label(resource)}")
            try:
                bot_info = await event.bot.get_me()
            except Exception:
                bot_info = None
            if bot_info and getattr(bot_info, "username", None):
                text.append(
                    f"Ссылка: <code>https://t.me/{html.escape(bot_info.username)}?start={html.escape(str(identity))}</code>"
                )
        rows = []
        if localized and len(locales) > 1:
            rows.append([
                (
                    LOCALE_LABELS[item],
                    f"ca:locale:{kind}:{identity}:{item}"
                    if page == 0
                    else f"ca:locale:{kind}:{identity}:{item}:{page}",
                )
                for item in locales
            ])
            if kind == "content":
                completeness = []
                for field in ("button_title", "text_content"):
                    statuses = []
                    for item in locales:
                        localized_value = await read_content_value(session, kind, resource, field, item)
                        statuses.append(_content_locale_status(localized_value, item))
                    label = "Название кнопки" if field == "button_title" else "Контент раздела"
                    completeness.append(f"{label}: " + " · ".join(statuses))
                text.extend(("", *completeness))
        for field, title, _ in fields:
            value = await read_content_value(session, kind, resource, field, locale)
            preview = _content_display_value(resource, value, locale) if kind == "content" else value.admin_label()
            if len(preview) > 250:
                preview = preview[:247] + "…"
            if kind == "content" and field == "text_content":
                from handlers import render_admin_content_preview
                preview = render_admin_content_preview(preview)
            else:
                preview = html.escape(preview)
            text.append(f"\n<b>{title}:</b>\n{preview}")
            if field.startswith("option."):
                _, slot, part = field.split(".")
                option = next(option for option in get_answer_options(resource) if option.translation_slot == slot)
                route = f"ca:opt:{identity}:{option.callback_id}:{part}"
            else:
                route = f"ca:edit:{kind}:{identity}:{locale}:{fields.index((field, title, _))}:{page}"
            if kind == "content" and field == "text_content":
                continue
            rows.append([(f"Изменить: {title}", route)])
        globals_markup = None
        import keyboards as kb
        if kind == "topic":
            globals_markup = kb.edit_topic_keyboard(identity, resource.is_active, in_menu=resource.show_in_main_menu, in_list=resource.show_in_list, admin_only=resource.admin_only, auto_start=resource.auto_start_dialogue)
        elif kind == "plan":
            rows.append([("Общие настройки тарифа", f"ca:settings:plan:{identity}:{locale}")])
        elif kind == "content":
            rows.append([("✏️ Изменить: контент", f"ca:edit_content:content:{identity}:{locale}:{page}")])
            rows.append([(
                "👁 Скрыть раздел" if resource.is_visible else "👁 Показать раздел",
                f"ca:visibility:content:{identity}:{locale}:{page}",
            )])
            rows.append([("✏️ Переименовать ID", f"ca:rename:content:{identity}:{locale}:{page}")])
            rows.append([("🗑 Удалить раздел", f"ca:delete:content:{identity}:{locale}:{page}")])
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
        if kind == "followup_step":
            rows.append([("⬅️ Назад", f"followup_step_{resource.campaign_id}_{identity}")])
        else:
            rows.append([('К списку', f'ca:list:{kind}:{page}')])
    await event.message.edit_text("\n".join(text), reply_markup=keyboard(rows), parse_mode="HTML")


def resource_fields(kind, resource):
    fields = list(AUTHORING_RESOURCES.get(kind, RESOURCES[kind]).fields)
    if kind == "test_question":
        for index, option in enumerate(get_answer_options(resource)):
            fields.extend([
                (f"option.{option.translation_slot}.text", f"Ответ {index + 1}", "text"),
                (f"option.{option.translation_slot}.button_text", f"Кнопка ответа {index + 1}", "inline_button"),
            ])
    return fields


async def begin_edit(event, state, kind, identity, field_index, locale="ru"):
    spec = AUTHORING_RESOURCES.get(kind, RESOURCES[kind])
    async with async_session_maker() as session:
        if not await multilingual_authoring_enabled(session):
            locale = "ru"
        elif locale not in await authoring_locales(session):
            locale = "ru"
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
    data = await state.get_data()
    page = int(data.get("parent_page", 0) or 0)
    await state.set_state(ContentAuthoringStates.value)
    await state.set_data({
        "kind": kind,
        "identity": identity,
        "field": field,
        "locale": locale,
        "authoring_locale": locale,
        "parent_page": page,
    })
    rows = []
    if value.needs_review:
        rows.append([("Подтвердить перевод", "ca:confirm")])
    rows.append([("Отмена", f"ca:view:{kind}:{identity}:{locale}:{page}")])
    reference = f"\n\nРусский исходник:\n{html.escape((value.russian or 'Не задан')[:1000])}" if locale != "ru" else ""
    heading = f"{locale_heading(locale)}\n\n" if is_localized_resource(kind) else ""
    current_label = value.admin_label() if locale != "ru" else (value.text or "Не задано")
    await event.message.edit_text(
        f"{heading}{title}:\n{html.escape(current_label[:1500])}{reference}\n\nВведите новое значение.",
        reply_markup=keyboard(rows),
    )


@router.callback_query(F.data.startswith("ca:"))
async def content_callback(callback: CallbackQuery, state):
    if not await allowed(callback):
        return
    parts = callback.data.split(":")
    action = parts[1]
    if action in {
        "language", "list", "locale", "view", "edit", "edit_content", "new",
        "settings", "visibility", "rename", "delete", "delete_confirm", "confirm",
    }:
        kind = parts[2] if len(parts) > 2 and action in {
            "list", "locale", "view", "edit", "edit_content", "new", "settings",
            "visibility", "rename", "delete", "delete_confirm",
        } else None
        if action == "language" and not await authoring_enabled_for(callback):
            raise SkipHandler()
    if action == "language":
        await state.clear()
        await callback.message.edit_text("Выбор языка теперь доступен внутри карточки материала.", reply_markup=keyboard([[('В админ-панель', 'admin_panel')]]))
    elif action == "locale":
        if len(parts) not in {5, 6}:
            await callback.answer("Откройте карточку материала заново.", show_alert=True)
            return
        page = int(parts[5]) if len(parts) == 6 else 0
        await resource_card(callback, parts[2], parts[3], parts[4], page)
        await callback.answer()
        return
    elif action == "list":
        await state.clear()
        await resource_list(callback, parts[2], int(parts[3]))
    elif action == "view":
        await state.clear()
        tail = parts[4] if len(parts) > 4 else "ru"
        if len(parts) > 5:
            locale, page = tail or "ru", int(parts[5])
        elif tail in LOCALE_LABELS:
            locale, page = tail, 0
        else:
            locale, page = "ru", int(tail or 0)
        await resource_card(callback, parts[2], parts[3], locale, page)
    elif action == "edit":
        if len(parts) >= 6:
            locale = parts[4] or "ru"
            field_token = parts[5]
            page = int(parts[6]) if len(parts) > 6 else 0
        else:
            locale = "ru"
            field_token = parts[4]
            page = 0
        field_index = int(field_token) if field_token.isdigit() else field_token
        await state.update_data(parent_page=page)
        await begin_edit(callback, state, parts[2], parts[3], field_index, locale)
    elif action == "edit_content":
        from handlers import start_content_edit
        from admin_authoring_context import content_editing_locale
        locale = parts[4] if len(parts) > 4 else "ru"
        page = int(parts[5]) if len(parts) > 5 else 0
        await state.update_data(
            authoring_locale=locale,
            parent_page=page,
            parent_kind="content",
            parent_identity=parts[3],
        )
        token = content_editing_locale.set(locale)
        try:
            await start_content_edit(callback.model_copy(update={"data": "edit_content_" + parts[3]}), state)
        finally:
            content_editing_locale.reset(token)
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
        await state.set_state(ContentAuthoringStates.value)
        spec = AUTHORING_RESOURCES.get(kind, RESOURCES[kind])
        await state.set_data({
            "kind": kind,
            "identity": None,
            "field": spec.fields[0][0],
            "locale": "ru",
            "authoring_locale": "ru",
            "parent_page": 0,
        })
        await callback.message.edit_text(
            f"{spec.fields[0][1]}:",
            reply_markup=keyboard([[('Отмена', f'ca:list:{kind}:0')]]),
        )
    elif action == "settings":
        from handlers import _show_admin_edit_plan_menu, start_content_edit
        from admin_authoring_context import content_editing_locale
        locale = "ru"
        if len(parts) > 4:
            locale = parts[4]
        token = content_editing_locale.set(locale)
        try:
            await state.update_data(authoring_locale=locale)
            if parts[2] == "plan":
                await _show_admin_edit_plan_menu(callback.bot, callback.message.chat.id, callback.message.message_id, int(parts[3]))
            elif parts[2] == "content":
                await start_content_edit(callback.model_copy(update={"data": "edit_content_" + parts[3]}), state)
            if await state.get_state():
                await state.update_data(authoring_locale=locale)
        finally:
            content_editing_locale.reset(token)
    elif action == "visibility":
        page = int(parts[5]) if len(parts) > 5 else 0
        async with async_session_maker() as session:
            resource = await session.get(Content, parts[3])
            if resource is None:
                await callback.answer("Раздел уже удалён.", show_alert=True)
                return
            resource.is_visible = not bool(resource.is_visible)
            await session.commit()
            locale = parts[4] if len(parts) > 4 else "ru"
        await resource_card(callback, "content", parts[3], locale, page)
    elif action == "rename":
        page = int(parts[5]) if len(parts) > 5 else 0
        locale = parts[4] if len(parts) > 4 else "ru"
        async with async_session_maker() as session:
            dependencies = await content_dependency_report(session, parts[3])
        dependencies.append("внешние Telegram deep-link URL и уже отправленные callback-ссылки не отслеживаются")
        await callback.message.edit_text(
            "Переименование заблокировано, чтобы не оставить нерабочие ссылки.\n\n"
            "Зависимости:\n" + "\n".join(f"• {item}" for item in dependencies) +
            "\n\nСоздайте новый раздел и перенесите текст вручную.",
            reply_markup=keyboard([[('К карточке', f"ca:view:content:{parts[3]}:{locale}:{page}")]]),
        )
        await callback.answer("Переименование отменено: ключ является публичной идентичностью.", show_alert=True)
        return
    elif action == "delete":
        page = int(parts[5]) if len(parts) > 5 else 0
        locale = parts[4] if len(parts) > 4 else "ru"
        async with async_session_maker() as session:
            resource = await session.get(Content, parts[3])
            if resource is None:
                await callback.answer("Раздел уже удалён.", show_alert=True)
                return
            dependencies = await content_dependency_report(session, parts[3])
        if dependencies:
            await callback.message.edit_text(
                "Нельзя удалить раздел без риска оставить нерабочие ссылки.\n\n" +
                "Зависимости:\n" + "\n".join(f"• {item}" for item in dependencies) +
                "\n\nСначала удалите или измените эти ссылки.",
                reply_markup=keyboard([[('К карточке', f"ca:view:content:{parts[3]}:{locale}:{page}")]]),
            )
        else:
            await callback.message.edit_text(
                f"Удалить раздел <b>{html.escape(parts[3])}</b> и его локализованные версии?",
                parse_mode="HTML",
                reply_markup=keyboard([
                    [("🗑 Да, удалить", f"ca:delete_confirm:content:{parts[3]}:{locale}:{page}")],
                    [("Отмена", f"ca:view:content:{parts[3]}:{locale}:{page}")],
                ]),
            )
    elif action == "delete_confirm":
        page = int(parts[5]) if len(parts) > 5 else 0
        async with async_session_maker() as session:
            dependencies = await content_dependency_report(session, parts[3])
            if dependencies:
                await callback.answer("Зависимости изменились. Удаление отменено.", show_alert=True)
                return
            await _delete_content_resource(session, parts[3])
            await session.commit()
        await resource_list(callback, "content", page)
    elif action == "confirm":
        data = await state.get_data()
        if not data.get("identity"):
            await callback.answer("Откройте материал заново.")
            return
        spec = RESOURCES[data["kind"]]
        async with async_session_maker() as session:
            async with translation_coordination_lock(session):
                resource = await session.get(spec.model, data["identity"] if spec.identity_field == "key" else int(data["identity"]))
                if resource is None:
                    await callback.answer("Материал удалён. Откройте список заново.", show_alert=True)
                    return
                value = await read_content_value(session, data["kind"], resource, data["field"], data["locale"])
                if value.text:
                    await save_content_value(session, data["kind"], resource, data["field"], data["locale"], value.text)
                    await session.commit()
        await state.clear()
        await resource_card(
            callback,
            data["kind"],
            data["identity"],
            data.get("locale", "ru"),
            int(data.get("parent_page", 0) or 0),
        )
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
    locale = data.get("locale", "ru")
    heading = f"{locale_heading(locale)}\n\n" if is_localized_resource(data["kind"]) else ""
    page = int(data.get("parent_page", 0) or 0)
    route = (
        f"ca:view:{data['kind']}:{identity}:{locale}:{page}"
        if is_localized_resource(data["kind"])
        else f"ca:view:{data['kind']}:{identity}:{page}"
    )
    await message.answer(f"{heading}Сохранено.", reply_markup=keyboard([[('Открыть материал', route)]]))


ENTRY_LISTS = {
    "admin_content": "content",
    "admin_plans": "plan",
    "admin_test_questions": "test_question",
    "admin_referral_templates": "referral_template",
}


@router.callback_query(lambda event: event.data in ENTRY_LISTS or bool(re.fullmatch(r"admin_(?:topics|case_studies)_page_\d+", event.data or "")))
async def list_entry(callback: CallbackQuery, state):
    if not await allowed(callback):
        return
    paginated = callback.data.startswith(("admin_topics_page_", "admin_case_studies_page_"))
    kind = "case_study" if callback.data.startswith("admin_case_studies_page_") else ENTRY_LISTS.get(callback.data, "topic")
    if is_localized_resource(kind) and kind != "content" and not await authoring_enabled_for(callback):
        raise SkipHandler()
    await state.clear()
    await resource_list(callback, kind, int(callback.data.rsplit("_", 1)[1]) if paginated else 0)
    await callback.answer()


DETAIL_ROUTES = ((r"edit_topic_(\d+)", "topic"), (r"admin_edit_plan_(\d+)", "plan"), (r"edit_content_(?!btn_)(.+)", "content"))


@router.callback_query(lambda event: any(re.fullmatch(pattern, event.data or "") for pattern, _ in DETAIL_ROUTES))
async def detail_entry(callback: CallbackQuery, state):
    if not await allowed(callback):
        return
    if not await authoring_enabled_for(callback):
        raise SkipHandler()
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
    if not await authoring_enabled_for(callback):
        raise SkipHandler()
    if callback.data in SINGLE_FIELDS:
        kind, index = SINGLE_FIELDS[callback.data]
        if kind not in AUTHORING_RESOURCES:
            raise SkipHandler()
        await resource_card(callback, kind, "1")
    else:
        for pattern, kind, index in FIELD_ROUTES:
            match = re.fullmatch(pattern, callback.data)
            if match:
                if kind not in AUTHORING_RESOURCES:
                    raise SkipHandler()
                await resource_card(callback, kind, match[1])
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
    await callback.message.edit_text(f"Вопрос #{question_id}\nПорядок, значения и удаление общие для всех языков.", reply_markup=keyboard(rows))


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
