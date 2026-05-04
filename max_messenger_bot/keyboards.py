from __future__ import annotations

from typing import Iterable

from sqlalchemy import select

from .legacy import Content, SubscriptionConfig, SubscriptionPlan, TestConfig, Topic, async_session_maker


def callback_button(text: str, payload: str) -> dict:
    return {"type": "callback", "text": text, "payload": payload}


def message_button(text: str, message: str | None = None) -> dict:
    return {"type": "message", "text": text, "message": message or text}


def link_button(text: str, url: str) -> dict:
    return {"type": "link", "text": text, "url": url}


def inline_keyboard(rows: list[list[dict]]) -> list[dict]:
    return [{"type": "inline_keyboard", "payload": {"buttons": rows}}]


def main_menu_row(text: str = "⬅️ В главное меню") -> list[dict]:
    return [callback_button(text, "main_menu")]


async def build_main_menu() -> list[dict]:
    async with async_session_maker() as session:
        content_items = (
            await session.execute(
                select(Content)
                .where(
                    Content.is_visible == True,
                    Content.button_title != None,
                    Content.key.not_in(["disclaimer", "test_results", "test_intro", "secret_test_outro", "test_button"]),
                )
                .order_by(Content.sort_order.asc())
            )
        ).scalars().all()
        topics = (
            await session.execute(
                select(Topic)
                .where(Topic.is_active == True, Topic.show_in_main_menu == True)
                .order_by(Topic.sort_order.asc(), Topic.id.asc())
            )
        ).scalars().all()
        sub_config = await session.get(SubscriptionConfig, 1)
        test_config = await session.get(TestConfig, 1)

    topics_btn_name = sub_config.topics_btn_name if sub_config else "📚 Темы диалога"
    referral_btn_name = sub_config.referral_btn_name if sub_config else "👥 Пригласить друзей"
    rows: list[list[dict]] = []

    if sub_config and sub_config.topics_enabled and sub_config.topics_btn_on_top:
        rows.append([message_button(topics_btn_name)])

    for index in range(0, len(content_items), 2):
        rows.append([message_button(item.button_title) for item in content_items[index:index + 2] if item.button_title])

    for index in range(0, len(topics), 2):
        rows.append([message_button(item.name) for item in topics[index:index + 2]])

    static_row: list[dict] = []
    if not test_config or test_config.is_enabled:
        static_row.append(message_button("📝 Пройти тест"))
    if not sub_config or sub_config.subscriptions_enabled:
        static_row.append(message_button("⭐️ Подписка"))
    if static_row:
        rows.append(static_row)

    if sub_config and sub_config.referral_enabled:
        rows.append([message_button(referral_btn_name)])

    if not sub_config or sub_config.topics_enabled:
        if not sub_config or not sub_config.topics_btn_on_top:
            rows.append([message_button(topics_btn_name)])

    bottom_row = [message_button("⚙️ Настройки"), message_button("🗑️ Новый диалог")]
    rows.append(bottom_row)
    return inline_keyboard(rows)


def admin_panel_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("📊 Статистика", "admin_stats"), callback_button("👥 Клиенты", "admin_clients")],
            [callback_button("🧩 Тест", "admin_test_menu"), callback_button("🤖 ИИ", "admin_ai_settings")],
            [callback_button("⭐️ Подписки", "admin_subscriptions"), callback_button("💬 Темы", "admin_topics")],
            [callback_button("📚 База знаний", "admin_kb"), callback_button("✏️ Контент", "admin_content")],
            [callback_button("🎛️ Кнопки меню", "admin_manage_buttons"), callback_button("👮 Админы", "admin_manage_admins")],
            [callback_button("👫 Рефералы", "admin_referral_menu")],
            [callback_button("✉️ Рассылки", "admin_mailing_menu")],
            [callback_button("🎨 Коллекции", "admin_collections_page_0")],
        ]
    )


def settings_keyboard(user) -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("✏️ Имя", "settings_change_name"), callback_button("👤 Пол", "settings_change_gender")],
            [callback_button("🔢 Возраст", "settings_change_age"), callback_button("📏 Длина ответов", "settings_toggle_length")],
            main_menu_row(),
        ]
    )


def gender_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("👨 Мужской", "gender_male"), callback_button("👩 Женский", "gender_female")],
        ]
    )


def disclaimer_keyboard() -> list[dict]:
    return inline_keyboard([[callback_button("✅ Я понимаю и принимаю", "disclaimer_accepted")]])


def topics_keyboard(topics: Iterable, current_topic_id: int | None) -> list[dict]:
    rows: list[list[dict]] = []
    for topic in topics:
        text = f"✅ {topic.name}" if topic.id == current_topic_id else topic.name
        rows.append([callback_button(text, f"select_topic_{topic.id}")])
    rows.append([callback_button("🏠 Перейти в основной диалог", "reset_topic")])
    rows.append(main_menu_row())
    return inline_keyboard(rows)


def subscription_keyboard(
    sub_info: dict | None,
    referral_enabled: bool,
    referral_btn_name: str,
    tg_user_id: int | None = None,
) -> list[dict]:
    rows = [[callback_button("💳 Оформить/Сменить тариф", "sub_select_plan")]]
    if sub_info and sub_info.get("allow_auto_renewal", True):
        callback_data = "sub_disable_renewal" if sub_info.get("auto_renewal") else "sub_enable_renewal"
        text = "❌ Отменить автопродление" if sub_info.get("auto_renewal") else "✅ Включить автопродление"
        rows.append([callback_button(text, callback_data)])
    rows.append([callback_button("🎁 Ввести промокод", "sub_enter_promo")])
    if tg_user_id is None:
        rows.append([callback_button("🔗 Привязать TG аккаунт", "link_tg_start")])
    if referral_enabled:
        rows.append([callback_button(referral_btn_name, "referral_sub_info")])
    rows.append(main_menu_row())
    return inline_keyboard(rows)


def retry_subscription_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("💳 Списать сейчас", "sub_retry_now")],
            [callback_button("❌ Отменить автопродление", "sub_disable_renewal")],
            [callback_button("🔄 Оформить заново", "sub_cancel_retry")],
            [callback_button("🎁 Ввести промокод", "sub_enter_promo")],
            main_menu_row(),
        ]
    )


def plans_keyboard(plans: Iterable[SubscriptionPlan], discount_percent: int, user_promos: list) -> list[dict]:
    rows: list[list[dict]] = []
    for plan in plans:
        price = plan.price
        plan_discount = discount_percent
        specific = next(
            (
                promo
                for promo in user_promos
                if not promo.applies_to_all_plans and any(item.id == plan.id for item in promo.applicable_plans)
            ),
            None,
        )
        if specific:
            plan_discount = specific.discount_percent
        elif not specific and plan_discount == 0:
            all_plans = next((promo for promo in user_promos if promo.applies_to_all_plans), None)
            if all_plans:
                plan_discount = all_plans.discount_percent
        if plan_discount > 0 and not plan.is_trial:
            price = price * (1 - plan_discount / 100)
        unit = "дн." if plan.duration_unit == "days" else "мес."
        label = f"{plan.name} ({plan.duration_value} {unit}) - {price:.2f} руб."
        rows.append([callback_button(label, f"sub_pay_{plan.id}")])
    rows.append([callback_button("⬅️ Назад", "back_to_sub_info")])
    rows.append(main_menu_row())
    return inline_keyboard(rows)


def payment_providers_keyboard(providers: list[dict]) -> list[dict]:
    rows = [[button] for button in providers]
    rows.append([callback_button("⬅️ К тарифам", "sub_select_plan")])
    rows.append(main_menu_row())
    return inline_keyboard(rows)


def test_answers_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("1", "test_ans_1"), callback_button("2", "test_ans_2"), callback_button("3", "test_ans_3")],
            [callback_button("4", "test_ans_4"), callback_button("5", "test_ans_5")],
        ]
    )


def case_study_keyboard() -> list[dict]:
    return inline_keyboard([[callback_button("✅ Показать результаты", "test_confirm_case")]])


def secret_test_keyboard(marathon_url: str) -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("🔐 Пройти секретный тест", "start_secret_test")],
            [link_button("Сразу на марафон 🚀", marathon_url)],
        ]
    )


def final_test_keyboard(marathon_url: str) -> list[dict]:
    return inline_keyboard(
        [
            [link_button("🔥 Программа марафона", marathon_url)],
            [callback_button("🗣 Продолжить общение", "continue_dialogue_after_test")],
        ]
    )


def admin_clients_keyboard(page: int, total_pages: int, clients: list) -> list[dict]:
    rows: list[list[dict]] = []
    for client in clients:
        name = client.name or client.first_name or str(client.id)
        username = f"@{client.username}" if client.username else "без username"
        rows.append([callback_button(f"{name} ({username})", f"view_client_{client.id}")])

    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"admin_clients_page_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"admin_clients_page_{page + 1}"))
    if nav_row:
        rows.append(nav_row)
    if total_pages > 1:
        rows.append([
            callback_button("⏮ В начало", "admin_clients_page_0"),
            callback_button("В конец ⏭", f"admin_clients_page_{total_pages - 1}"),
        ])
    rows.append([callback_button("🔍 Поиск", "admin_client_search"), callback_button("📤 Экспорт", "admin_export")])
    rows.append([callback_button("⬅️ В админ-панель", "admin_panel")])
    return inline_keyboard(rows)


def admin_client_search_keyboard() -> list[dict]:
    return inline_keyboard([[callback_button("⬅️ К клиентам", "admin_clients")]])


def admin_export_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("👥 Пользователи (CSV)", "admin_export_users_csv")],
            [callback_button("💬 Сообщения (CSV)", "admin_export_date_filter")],
            [callback_button("◀️ Назад", "admin_clients")],
        ]
    )


def admin_date_filter_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("🗓 Все даты", "admin_export_date_all")],
            [callback_button("📅 7 дней", "admin_export_date_preset_7"), callback_button("📅 30 дней", "admin_export_date_preset_30")],
            [callback_button("📅 90 дней", "admin_export_date_preset_90"), callback_button("✏️ Вручную", "admin_export_date_manual")],
            [callback_button("◀️ Назад", "admin_export")],
        ]
    )


def admin_client_profile_keyboard(user_id: int) -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("📜 История диалога", f"client_history_{user_id}_0")],
            [callback_button("📥 Скачать историю", f"download_history_{user_id}"), callback_button("🗑️ Удалить историю", f"admin_delete_history_{user_id}")],
            [callback_button("💳 Платежи", f"client_payment_info_{user_id}"), callback_button("🔄 Сбросить подписку", f"admin_reset_sub_{user_id}")],
            [callback_button("🔄 Сбросить промокоды", f"reset_user_promos_{user_id}")],
            [callback_button("⬅️ К клиентам", "admin_clients")],
        ]
    )


def admin_history_keyboard(user_id: int, page: int, total_pages: int) -> list[dict]:
    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"client_history_{user_id}_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"client_history_{user_id}_{page + 1}"))
    rows = [nav_row] if nav_row else []
    if total_pages > 1:
        rows.append([
            callback_button("⏮ В начало", f"client_history_{user_id}_0"),
            callback_button("В конец ⏭", f"client_history_{user_id}_{total_pages - 1}"),
        ])
    rows.append([callback_button("⬅️ К профилю", f"view_client_{user_id}")])
    return inline_keyboard(rows)


def admin_content_list_keyboard(items: list[tuple[str, str, bool]]) -> list[dict]:
    rows: list[list[dict]] = []
    for key, title, is_visible in items:
        status = "✅" if is_visible else "❌"
        rows.append([callback_button(f"{status} {title}", f"admin_edit_content_{key}")])
    rows.append([callback_button("⬅️ В админ-панель", "admin_panel")])
    return inline_keyboard(rows)


def admin_content_editor_keyboard(content_key: str, is_visible: bool) -> list[dict]:
    status_text = "Скрыть" if is_visible else "Показать"
    return inline_keyboard(
        [
            [callback_button("✏️ Изменить текст", f"admin_content_edit_text_{content_key}")],
            [callback_button(f"👁 {status_text}", f"admin_toggle_content_visibility_{content_key}")],
            [callback_button("⬅️ К контенту", "admin_content")],
        ]
    )


def admin_buttons_keyboard(items: list) -> list[dict]:
    rows: list[list[dict]] = []
    for item in items:
        status = "✅" if item.is_visible else "❌"
        title = item.button_title or item.key
        rows.append([callback_button(f"{status} {title}", f"admin_button_open_{item.key}")])
    rows.append([callback_button("➕ Добавить новую кнопку", "admin_add_button")])
    rows.append([callback_button("⬅️ В админ-панель", "admin_panel")])
    return inline_keyboard(rows)


def admin_button_editor_keyboard(button_key: str, is_visible: bool) -> list[dict]:
    visibility_text = "👁 Скрыть" if is_visible else "👁 Показать"
    return inline_keyboard(
        [
            [callback_button("✏️ Переименовать", f"admin_button_edit_title_{button_key}")],
            [callback_button(visibility_text, f"admin_button_toggle_visibility_{button_key}")],
            [callback_button("⬆️ Выше", f"admin_button_move_up_{button_key}"), callback_button("⬇️ Ниже", f"admin_button_move_down_{button_key}")],
            [callback_button("🗑️ Удалить кнопку", f"admin_button_delete_{button_key}")],
            [callback_button("⬅️ К кнопкам", "admin_manage_buttons")],
        ]
    )


def admin_admins_keyboard(admins: list) -> list[dict]:
    rows: list[list[dict]] = []
    for admin in admins:
        name = admin.first_name or admin.name or str(admin.id)
        username = f" @{admin.username}" if admin.username else ""
        rows.append([callback_button(f"👮 {name}{username}", f"admin_profile_{admin.id}")])
    rows.append([callback_button("➕ Добавить администратора", "admin_add_admin")])
    rows.append([callback_button("⬅️ В админ-панель", "admin_panel")])
    return inline_keyboard(rows)


def admin_profile_keyboard(admin_id: int, can_view_history: bool, can_revoke: bool) -> list[dict]:
    rows = [[callback_button("❌ Забрать доступ к истории" if can_view_history else "✅ Дать доступ к истории", f"admin_toggle_history_{admin_id}")]]
    if can_revoke:
        rows.append([callback_button("➖ Разжаловать", f"admin_revoke_{admin_id}")])
    rows.append([callback_button("⬅️ К списку админов", "admin_manage_admins")])
    return inline_keyboard(rows)


def admin_referral_templates_keyboard(templates: list) -> list[dict]:
    rows: list[list[dict]] = []
    for tpl in templates:
        short_text = (tpl.text or "")[:40].replace("\n", " ")
        suffix = "..." if len(tpl.text or "") > 40 else ""
        status = "✅" if tpl.is_enabled else "❌"
        rows.append([callback_button(f"{status} {tpl.order_num + 1}. {short_text}{suffix}", f"admin_ref_tpl_{tpl.id}")])
    rows.append([callback_button("➕ Добавить шаблон", "admin_ref_tpl_add")])
    rows.append([callback_button("⬅️ Назад", "admin_referral_menu")])
    return inline_keyboard(rows)


def admin_referral_template_detail_keyboard(tpl_id: int, is_enabled: bool) -> list[dict]:
    toggle_label = "❌ Отключить" if is_enabled else "✅ Включить"
    return inline_keyboard(
        [
            [callback_button("✏️ Редактировать", f"admin_ref_tpl_edit_{tpl_id}")],
            [callback_button(toggle_label, f"admin_ref_tpl_toggle_{tpl_id}")],
            [callback_button("⬆️ Выше", f"admin_ref_tpl_up_{tpl_id}"), callback_button("⬇️ Ниже", f"admin_ref_tpl_down_{tpl_id}")],
            [callback_button("🗑 Удалить", f"admin_ref_tpl_del_{tpl_id}")],
            [callback_button("⬅️ К шаблонам", "admin_referral_templates")],
        ]
    )


def admin_referral_template_confirm_delete_keyboard(tpl_id: int) -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("🗑 Да, удалить", f"admin_ref_tpl_del_confirm_{tpl_id}")],
            [callback_button("⬅️ Отмена", f"admin_ref_tpl_{tpl_id}")],
        ]
    )


def admin_referral_template_input_cancel_keyboard() -> list[dict]:
    return inline_keyboard([[callback_button("⬅️ Отмена", "admin_referral_templates")]])


def admin_topics_list_keyboard(topics: list) -> list[dict]:
    rows: list[list[dict]] = []
    for topic in topics:
        status = "🟢" if topic.is_active else "⚪️"
        admin_only = " 🔒" if topic.admin_only else ""
        rows.append([callback_button(f"{status} {topic.name}{admin_only}", f"admin_edit_topic_{topic.id}")])
    rows.append([callback_button("➕ Создать тему", "admin_create_topic")])
    rows.append([callback_button("⬅️ В админ-панель", "admin_panel")])
    return inline_keyboard(rows)


def admin_topic_editor_keyboard(topic) -> list[dict]:
    active_text = "⚪️ Сделать неактивной" if topic.is_active else "🟢 Сделать активной"
    admin_only_text = "🔒 Только для админов: ВКЛ" if topic.admin_only else "🔓 Только для админов: ВЫКЛ"
    menu_text = "❌ Убрать из меню" if topic.show_in_main_menu else "✅ Показать в меню"
    list_text = "❌ Убрать из списка" if topic.show_in_list else "✅ Показать в списке"
    return inline_keyboard(
        [
            [callback_button("✏️ Название", f"admin_topic_edit_name_{topic.id}")],
            [callback_button("📝 Системный промпт", f"admin_topic_edit_prompt_{topic.id}")],
            [callback_button("💬 Приветственное сообщение", f"admin_topic_edit_intro_{topic.id}")],
            [callback_button(active_text, f"admin_topic_toggle_active_{topic.id}")],
            [callback_button(admin_only_text, f"admin_topic_toggle_admin_{topic.id}")],
            [callback_button(menu_text, f"admin_topic_toggle_menu_{topic.id}")],
            [callback_button(list_text, f"admin_topic_toggle_list_{topic.id}")],
            [callback_button("📚 База знаний темы", f"admin_topic_kb_{topic.id}_page_0")],
            [callback_button("🖼️ Медиатека", f"admin_topic_media_{topic.id}_0")],
            [callback_button("🗑️ Удалить тему", f"admin_topic_delete_{topic.id}")],
            [callback_button("⬅️ К темам", "admin_topics")],
        ]
    )


def admin_kb_list_keyboard(entries: list, page: int, total_pages: int) -> list[dict]:
    rows: list[list[dict]] = []
    for entry in entries:
        general = "✅" if entry.use_in_general_mode else "⭕️"
        topic_marker = " 🎯" if getattr(entry, "topics", None) else ""
        title = entry.filename or f"KB #{entry.id}"
        rows.append([callback_button(f"{general} {title}{topic_marker}", f"admin_kb_open_{entry.id}")])
    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"admin_kb_page_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"admin_kb_page_{page + 1}"))
    if nav_row:
        rows.append(nav_row)
    rows.append([callback_button("➕ Добавить запись", "admin_kb_create")])
    rows.append([callback_button("⬅️ В админ-панель", "admin_panel")])
    return inline_keyboard(rows)


def admin_kb_editor_keyboard(kb_id: int, use_in_general_mode: bool) -> list[dict]:
    general_text = "❌ Убрать из общего режима" if use_in_general_mode else "✅ Добавить в общий режим"
    return inline_keyboard(
        [
            [callback_button("✏️ Название", f"admin_kb_edit_filename_{kb_id}")],
            [callback_button("📝 Содержимое", f"admin_kb_edit_content_{kb_id}")],
            [callback_button(general_text, f"admin_kb_toggle_general_{kb_id}")],
            [callback_button("🗑️ Удалить запись", f"admin_kb_delete_{kb_id}")],
            [callback_button("⬅️ К базе знаний", "admin_kb")],
        ]
    )


def admin_topic_kb_keyboard(topic_id: int, entries: list, assigned_ids: set[int], page: int, total_pages: int) -> list[dict]:
    rows: list[list[dict]] = []
    for entry in entries:
        marker = "✅ " if entry.id in assigned_ids else ""
        title = entry.filename or f"KB #{entry.id}"
        rows.append([callback_button(f"{marker}{title}", f"admin_topic_kb_toggle_{topic_id}_{entry.id}_{page}")])
    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"admin_topic_kb_{topic_id}_page_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"admin_topic_kb_{topic_id}_page_{page + 1}"))
    if nav_row:
        rows.append(nav_row)
    rows.append([callback_button("⬅️ К теме", f"admin_edit_topic_{topic_id}")])
    return inline_keyboard(rows)


def admin_test_menu_keyboard(config) -> list[dict]:
    status_text = "✅ Включен" if config.is_enabled else "❌ Выключен"
    return inline_keyboard(
        [
            [callback_button(f"Статус теста: {status_text}", "admin_test_toggle_status")],
            [callback_button("✏️ Приветствие теста", "admin_edit_content_test_intro")],
            [callback_button("✏️ Результаты теста", "admin_edit_content_test_results")],
            [callback_button("✏️ Финал секретного теста", "admin_edit_content_secret_test_outro")],
            [callback_button("🔗 Настройка ссылок", "admin_test_links")],
            [callback_button("❓ Вопросы теста", "admin_test_questions"), callback_button("📖 Кейсы", "admin_case_studies_page_0")],
            [callback_button("🔐 Секретные вопросы", "admin_secret_questions")],
            [callback_button("📝 Промпт теста", "admin_edit_test_prompt")],
            [callback_button("⬅️ В админ-панель", "admin_panel")],
        ]
    )


def admin_test_links_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("✏️ Username админа", "admin_test_set_link_admin")],
            [callback_button("✏️ Ссылка на марафон", "admin_test_set_link_marathon")],
            [callback_button("⬅️ Назад", "admin_test_menu")],
        ]
    )


def admin_secret_questions_keyboard(items: list) -> list[dict]:
    rows: list[list[dict]] = []
    for item in items:
        title = item.text[:40] + ("..." if len(item.text) > 40 else "")
        rows.append([callback_button(f"🗑️ {title}", f"admin_secret_question_delete_{item.id}")])
    rows.append([callback_button("➕ Добавить вопрос", "admin_secret_question_add")])
    rows.append([callback_button("⬅️ Назад", "admin_test_menu")])
    return inline_keyboard(rows)


def admin_test_questions_keyboard(questions: list) -> list[dict]:
    rows: list[list[dict]] = []
    for item in questions:
        title = item.text[:44] + ("..." if len(item.text) > 44 else "")
        marker = "🔁 " if item.is_reverse else ""
        rows.append([callback_button(f"{marker}{item.sort_order}. {title}", f"admin_edit_test_question_{item.id}")])
    rows.append([callback_button("➕ Добавить вопрос", "admin_test_question_create")])
    rows.append([callback_button("⬅️ Назад", "admin_test_menu")])
    return inline_keyboard(rows)


def admin_test_question_editor_keyboard(question_id: int, is_reverse: bool) -> list[dict]:
    reverse_text = "🔁 Обратный вопрос: ВКЛ" if is_reverse else "➡️ Прямой вопрос"
    return inline_keyboard(
        [
            [callback_button("✏️ Текст", f"admin_test_question_edit_text_{question_id}")],
            [callback_button("🏷 Категория", f"admin_test_question_category_menu_{question_id}")],
            [callback_button(reverse_text, f"admin_test_question_toggle_reverse_{question_id}")],
            [callback_button("↕️ Порядок", f"admin_test_question_edit_sort_{question_id}")],
            [callback_button("🗑️ Удалить вопрос", f"admin_test_question_delete_{question_id}")],
            [callback_button("⬅️ К вопросам", "admin_test_questions")],
        ]
    )


def admin_test_question_categories_keyboard(prefix: str, current_category: str | None = None) -> list[dict]:
    categories = [
        ("body", "Тело"),
        ("face", "Лицо"),
        ("age", "Возраст"),
        ("health", "Здоровье"),
        ("abilities", "Способности"),
        ("relations", "Отношения"),
        ("success", "Успех"),
    ]
    rows: list[list[dict]] = []
    for key, label in categories:
        text = f"✅ {label}" if key == current_category else label
        rows.append([callback_button(text, f"{prefix}_{key}")])
    return inline_keyboard(rows)


def admin_case_studies_keyboard(cases: list, page: int, total_pages: int) -> list[dict]:
    rows: list[list[dict]] = []
    for item in cases:
        title = item.text[:36].replace("\n", " ") + ("..." if len(item.text) > 36 else "")
        rows.append([callback_button(f"Кейс #{item.id}: {title}", f"admin_edit_case_study_{item.id}_{page}")])
    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"admin_case_studies_page_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"admin_case_studies_page_{page + 1}"))
    if nav_row:
        rows.append(nav_row)
    rows.append([callback_button("➕ Добавить кейс", "admin_case_study_create")])
    rows.append([callback_button("⬅️ Назад", "admin_test_menu")])
    return inline_keyboard(rows)


def admin_case_study_editor_keyboard(case_id: int, page: int | None = None) -> list[dict]:
    page_value = page if page is not None else 0
    back_target = f"admin_case_studies_page_{page_value}"
    return inline_keyboard(
        [
            [callback_button("✏️ Изменить текст", f"admin_case_study_edit_text_{case_id}_{page_value}")],
            [callback_button("🗑️ Удалить кейс", f"admin_case_study_delete_{case_id}_{page_value}")],
            [callback_button("⬅️ К кейсам", back_target)],
        ]
    )


def admin_mailing_menu_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("🚀 Создать рассылку", "mailing_create")],
            [callback_button("📜 История рассылок", "mailing_history_page_0")],
            [callback_button("⬅️ В админ-панель", "admin_panel")],
        ]
    )


def admin_mailing_audience_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("Всем пользователям", "mailing_audience_all")],
            [callback_button("Кто не начал диалог", "mailing_audience_no_dialogue")],
            [callback_button("Кто ни разу не платил", "mailing_audience_no_subscription")],
            [callback_button("Активным подписчикам", "mailing_audience_active_subscription")],
            [callback_button("Без активной подписки", "mailing_audience_inactive_subscription")],
            [callback_button("👤 Только себе", "mailing_audience_self")],
            [callback_button("⬅️ Назад", "admin_mailing_menu")],
        ]
    )


def admin_mailing_preview_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("✅ Отправить", "mailing_confirm_send"), callback_button("✏️ Изменить", "mailing_edit_content")],
            [callback_button("❌ Отмена", "admin_mailing_menu")],
        ]
    )


def admin_mailing_history_keyboard(mailings: list, page: int, total_pages: int) -> list[dict]:
    rows: list[list[dict]] = []
    status_map = {"pending": "⏳", "sending": "🚀", "completed": "✅", "failed": "❌"}
    for mailing in mailings:
        preview = (mailing.text or "Без текста").replace("\n", " ")[:28]
        rows.append([callback_button(f"{status_map.get(mailing.status, '❓')} #{mailing.id} {preview}", f"mailing_details_{mailing.id}")])
    nav_row: list[dict] = []
    if page > 0:
        nav_row.append(callback_button("⬅️", f"mailing_history_page_{page - 1}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("➡️", f"mailing_history_page_{page + 1}"))
    if nav_row:
        rows.append(nav_row)
    rows.append([callback_button("⬅️ Назад", "admin_mailing_menu")])
    return inline_keyboard(rows)


def admin_mailing_details_keyboard() -> list[dict]:
    return inline_keyboard([[callback_button("⬅️ К истории", "mailing_history_page_0")]])


def admin_payment_settings_keyboard(config) -> list[dict]:
    notif_status = "✅ Включены" if config.notifications_enabled else "❌ Выключены"
    sub_status = "✅ Включены" if config.subscriptions_enabled else "❌ Выключены"
    return inline_keyboard(
        [
            [callback_button(f"Уведомления: {notif_status}", "admin_toggle_sub_notifs")],
            [callback_button(f"Подписки: {sub_status}", "admin_toggle_subscriptions")],
            [callback_button(f"🎁 Бонус при входе: {config.welcome_bonus_days} дн.", "admin_set_welcome_bonus")],
            [callback_button("🔑 Платёжные ключи", "admin_payment_keys_menu")],
            [callback_button("⬅️ К подпискам", "admin_subscriptions")],
        ]
    )


def admin_payment_keys_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("ЮKassa Shop ID", "set_payment_key_yookassa_shop_id")],
            [callback_button("ЮKassa Secret Key", "set_payment_key_yookassa_secret_key")],
            [callback_button("Robokassa Merchant", "set_payment_key_robokassa_merchant_login")],
            [callback_button("Robokassa Password 1", "set_payment_key_robokassa_password_1")],
            [callback_button("Robokassa Password 2", "set_payment_key_robokassa_password_2")],
            [callback_button("Telegram Pay Token", "set_payment_key_telegram_pay_token")],
            [callback_button("URL оферты", "set_payment_key_offer_agreement_url")],
            [callback_button("URL политики", "set_payment_key_privacy_policy_url")],
            [callback_button("⬅️ Назад", "admin_payment_settings")],
        ]
    )


def admin_subscriptions_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("📦 Тарифы", "admin_plans"), callback_button("🎁 Промокоды", "admin_promocodes")],
            [callback_button("⚙️ Платёжные настройки", "admin_payment_settings")],
            [callback_button("📊 Статистика платежей", "admin_payment_stats"), callback_button("📋 Журнал платежей", "admin_plog_0_all")],
            [callback_button("⬅️ В админ-панель", "admin_panel")],
        ]
    )


def admin_ai_settings_keyboard(current_provider: str) -> list[dict]:
    providers = ["Deepseek", "Claude", "Gemini", "OpenAI", "KIE"]
    rows: list[list[dict]] = []
    for index in range(0, len(providers), 2):
        chunk = []
        for provider in providers[index:index + 2]:
            label = f"✅ {provider}" if provider.lower() == current_provider.lower() else provider
            chunk.append(callback_button(label, f"admin_ai_provider_{provider}"))
        rows.append(chunk)
    rows.append([callback_button("🔑 Ключи и модели", "admin_ai_keys")])
    rows.append([callback_button("📝 Системный промпт", "admin_ai_system_prompt")])
    rows.append([callback_button("📥 Скачать системный промпт", "admin_ai_download_system_prompt")])
    rows.append([callback_button("📎 Общий блок для всех промптов", "admin_ai_global_prompt_appendix")])
    rows.append([callback_button("📥 Скачать общий блок", "admin_ai_download_global_prompt_appendix")])
    rows.append([callback_button("⬅️ В админ-панель", "admin_panel")])
    return inline_keyboard(rows)


def admin_ai_keys_keyboard(
    transcription_provider: str,
    context_first: int,
    context_recent: int,
    vision_provider: str,
    vision_model: str,
    temperature: float,
    memory_mode_label: str,
    audio_limit: int,
) -> list[dict]:
    trans_label = f"🗣️ Аудио: {transcription_provider}" if transcription_provider != "None" else "🗣️ Аудио: ❌ Выкл"
    return inline_keyboard(
        [
            # Keys column | Models column — grouped same as TG bot
            [callback_button("🔑 Deepseek", "admin_ai_key_Deepseek"), callback_button("🧠 Deepseek", "admin_ai_models_Deepseek")],
            [callback_button("🔑 Claude",   "admin_ai_key_Claude"),   callback_button("🧠 Claude",   "admin_ai_models_Claude")],
            [callback_button("🔑 Gemini",   "admin_ai_key_Gemini"),   callback_button("🧠 Gemini",   "admin_ai_models_Gemini")],
            [callback_button("🔑 OpenAI",   "admin_ai_key_OpenAI"),   callback_button("🧠 OpenAI",   "admin_ai_models_OpenAI")],
            # Context
            [callback_button(f"📌 Первые: {context_first}", "admin_ai_set_context_first"), callback_button(f"🔄 Последние: {context_recent}", "admin_ai_set_context_recent")],
            # Audio
            [callback_button(trans_label, "admin_ai_toggle_transcription"), callback_button(f"⏱️ Лимит аудио: {audio_limit}", "admin_ai_set_audio_limit")],
            # Temp + Memory
            [callback_button(f"🌡️ Температура: {round(temperature, 2)}", "admin_ai_set_temperature"), callback_button(f"🧠 Память: {memory_mode_label}", "admin_ai_cycle_memory_scope")],
            # Vision
            [callback_button(f"👁️ Фото: {vision_provider}", "admin_ai_toggle_vision"), callback_button(f"Модель: {vision_model[:16]}", "admin_ai_vision_models")],
            [callback_button("⬅️ Назад", "admin_ai_settings")],
        ]
    )


def admin_ai_model_selection_keyboard(provider: str, current_model: str, models: list[str]) -> list[dict]:
    rows: list[list[dict]] = []
    for model in models:
        label = f"✅ {model}" if model == current_model else model
        rows.append([callback_button(label, f"admin_ai_set_model_{provider}_{model}")])
    rows.append([callback_button("⬅️ Назад", "admin_ai_keys")])
    return inline_keyboard(rows)


def admin_ai_vision_models_keyboard(current_model: str, models: list[str]) -> list[dict]:
    rows = [[callback_button(f"✅ {model}" if model == current_model else model, f"admin_ai_set_vision_model_{model}")] for model in models]
    rows.append([callback_button("⬅️ Назад", "admin_ai_keys")])
    return inline_keyboard(rows)


def admin_plans_list_keyboard(plans: list) -> list[dict]:
    rows: list[list[dict]] = []
    for plan in plans:
        status = "🟢" if plan.is_active else "⚪️"
        trial = " 🧪" if plan.is_trial else ""
        admin_only = " 🔒" if getattr(plan, "admin_only", False) else ""
        rows.append([callback_button(f"{status} {plan.name}{trial}{admin_only}", f"admin_edit_plan_{plan.id}")])
    rows.append([callback_button("➕ Создать тариф", "admin_create_plan")])
    rows.append([callback_button("⬅️ К подпискам", "admin_subscriptions")])
    return inline_keyboard(rows)


def admin_plan_editor_keyboard(plan) -> list[dict]:
    active_text = "⚪️ Сделать неактивным" if plan.is_active else "🟢 Сделать активным"
    admin_text = "🔒 Только для админов: ВКЛ" if getattr(plan, "admin_only", False) else "🔓 Только для админов: ВЫКЛ"
    renewal_text = "🔁 Автопродление: ВКЛ" if getattr(plan, "allow_auto_renewal", True) else "⛔️ Автопродление: ВЫКЛ"
    trial_text = "🧪 Пробный тариф: ВКЛ" if plan.is_trial else "🧪 Пробный тариф: ВЫКЛ"
    return inline_keyboard(
        [
            [callback_button("✏️ Название", f"admin_plan_edit_name_{plan.id}")],
            [callback_button("📝 Описание", f"admin_plan_edit_description_{plan.id}")],
            [callback_button("💰 Цена", f"admin_plan_edit_price_{plan.id}")],
            [callback_button("⏱ Длительность", f"admin_plan_edit_duration_value_{plan.id}")],
            [callback_button("📏 Единица длительности", f"admin_plan_duration_unit_menu_{plan.id}")],
            [callback_button(trial_text, f"admin_plan_toggle_trial_{plan.id}")],
            [callback_button("🔗 Апгрейд после триала", f"admin_plan_upgrade_menu_{plan.id}")],
            [callback_button("🧊 Кулдаун триала", f"admin_plan_edit_cooldown_{plan.id}")],
            [callback_button(active_text, f"admin_plan_toggle_active_{plan.id}")],
            [callback_button(admin_text, f"admin_plan_toggle_admin_{plan.id}")],
            [callback_button(renewal_text, f"admin_plan_toggle_renewal_{plan.id}")],
            [callback_button("🗑️ Удалить тариф", f"admin_plan_delete_{plan.id}")],
            [callback_button("⬅️ К тарифам", "admin_plans")],
        ]
    )


def admin_plan_duration_unit_keyboard(plan_id: int, prefix: str) -> list[dict]:
    return inline_keyboard(
        [
            [callback_button("Дни", f"{prefix}_{plan_id}_days"), callback_button("Месяцы", f"{prefix}_{plan_id}_months")],
            [callback_button("⬅️ Назад", f"admin_edit_plan_{plan_id}")],
        ]
    )


def admin_plan_create_duration_unit_keyboard() -> list[dict]:
    return inline_keyboard(
        [
            [
                callback_button("Дни", "admin_plan_create_duration_unit_days"),
                callback_button("Месяцы", "admin_plan_create_duration_unit_months"),
            ],
            [callback_button("⬅️ К тарифам", "admin_plans")],
        ]
    )


def admin_plan_upgrade_keyboard(plan_id: int, plans: list, current_target_id: int | None) -> list[dict]:
    rows: list[list[dict]] = []
    current_text = "✅ Без апгрейда" if current_target_id is None else "Без апгрейда"
    rows.append([callback_button(current_text, f"admin_plan_set_upgrade_none_{plan_id}")])
    for plan in plans:
        marker = "✅ " if plan.id == current_target_id else ""
        rows.append([callback_button(f"{marker}{plan.name}", f"admin_plan_set_upgrade_{plan_id}_{plan.id}")])
    rows.append([callback_button("⬅️ К тарифу", f"admin_edit_plan_{plan_id}")])
    return inline_keyboard(rows)


def admin_promocodes_list_keyboard(promos: list) -> list[dict]:
    rows: list[list[dict]] = []
    for promo in promos:
        status = "🟢" if promo.is_active else "⚪️"
        scope = " 🌍" if promo.applies_to_all_plans else " 🎯"
        rows.append([callback_button(f"{status} {promo.code}{scope}", f"admin_edit_promo_{promo.id}")])
    rows.append([callback_button("➕ Создать промокод", "admin_create_promo")])
    rows.append([callback_button("⬅️ К подпискам", "admin_subscriptions")])
    return inline_keyboard(rows)


def admin_promo_editor_keyboard(promo) -> list[dict]:
    active_text = "⚪️ Сделать неактивным" if promo.is_active else "🟢 Сделать активным"
    scope_text = "🌍 Применять ко всем тарифам" if not promo.applies_to_all_plans else "🎯 Ограничить по тарифам"
    return inline_keyboard(
        [
            [callback_button("✏️ Код", f"admin_promo_edit_code_{promo.id}")],
            [callback_button("💸 Скидка", f"admin_promo_edit_discount_{promo.id}")],
            [callback_button("🎁 Бесплатные дни", f"admin_promo_edit_days_{promo.id}")],
            [callback_button("🔢 Макс. использований", f"admin_promo_edit_uses_{promo.id}")],
            [callback_button(scope_text, f"admin_promo_toggle_all_{promo.id}")],
            [callback_button("📦 Привязка к тарифам", f"admin_promo_assign_menu_{promo.id}")],
            [callback_button(active_text, f"admin_promo_toggle_active_{promo.id}")],
            [callback_button("🗑️ Удалить промокод", f"admin_promo_delete_{promo.id}")],
            [callback_button("⬅️ К промокодам", "admin_promocodes")],
        ]
    )


def admin_promo_assign_keyboard(promo_id: int, plans: list, assigned_plan_ids: set[int], applies_to_all: bool) -> list[dict]:
    rows: list[list[dict]] = []
    header = "✅ Ко всем тарифам" if applies_to_all else "🎯 Выборочные тарифы"
    rows.append([callback_button(header, f"admin_promo_assign_toggle_all_{promo_id}")])
    for plan in plans:
        marker = "✅ " if plan.id in assigned_plan_ids else ""
        rows.append([callback_button(f"{marker}{plan.name}", f"admin_promo_assign_toggle_{promo_id}_{plan.id}")])
    rows.append([callback_button("⬅️ К промокоду", f"admin_edit_promo_{promo_id}")])
    return inline_keyboard(rows)
