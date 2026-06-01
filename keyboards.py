from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from sqlalchemy import select
from database import async_session_maker, Content, SubscriptionConfig, TestConfig, Topic
from config import OWNER_IDS
from datetime import timezone, timedelta, date
from mailing_utils import MAILING_AUDIENCE_LABELS, get_mailing_status_label, is_birthday_mailing
from memory_mode import memory_mode_label
from time_helpers import to_msk


async def main_client_keyboard():
    async with async_session_maker() as session:
        stmt = select(Content).where(
            Content.is_visible == True,
            Content.button_title != None,
            Content.key.not_in(['disclaimer', 'test_results', 'test_intro', 'secret_test_outro', 'test_button'])
        ).order_by(Content.sort_order.asc())
        result = await session.execute(stmt)
        buttons = result.scalars().all()

        sub_config = await session.get(SubscriptionConfig, 1)

        subscriptions_active = sub_config.subscriptions_enabled if sub_config else True
        topics_active = sub_config.topics_enabled if sub_config else True
        topics_btn_name = sub_config.topics_btn_name if sub_config else "📚 Темы диалога"
        topics_on_top = sub_config.topics_btn_on_top if sub_config else False
        change_name_active = sub_config.change_name_button_enabled if sub_config else True
        referral_active = sub_config.referral_enabled if sub_config else False
        referral_btn_name = sub_config.referral_btn_name if sub_config else "👥 Пригласить друзей"

        test_config = await session.get(TestConfig, 1)
        test_active = test_config.is_enabled if test_config else True

        topic_stmt = select(Topic).where(Topic.is_active == True, Topic.show_in_main_menu == True).order_by(Topic.sort_order.asc(), Topic.id.asc())
        topic_res = await session.execute(topic_stmt)
        menu_topics = topic_res.scalars().all()

    keyboard_rows = []

    if topics_active and topics_on_top:
        keyboard_rows.append([KeyboardButton(text=topics_btn_name)])

    content_rows = [
        [KeyboardButton(text=btn.button_title) for btn in buttons[i:i + 2]]
        for i in range(0, len(buttons), 2)
    ]
    keyboard_rows.extend(content_rows)

    topic_rows = [
        [KeyboardButton(text=topic.name) for topic in menu_topics[i:i + 2]]
        for i in range(0, len(menu_topics), 2)
    ]
    keyboard_rows.extend(topic_rows)

    static_row = []
    if test_active:
        static_row.append(KeyboardButton(text="📝 Пройти тест"))
    if subscriptions_active:
        static_row.append(KeyboardButton(text="⭐️ Подписка"))

    if static_row:
        keyboard_rows.append(static_row)

    if referral_active:
        keyboard_rows.append([KeyboardButton(text=referral_btn_name)])

    last_static_row = []
    if topics_active and not topics_on_top:
        last_static_row.append(KeyboardButton(text=topics_btn_name))

    if last_static_row:
        keyboard_rows.append(last_static_row)

    bottom_row = []
    if change_name_active:
        bottom_row.append(KeyboardButton(text="⚙️ Настройки"))
    bottom_row.append(KeyboardButton(text="🗑️ Новый диалог"))
    keyboard_rows.append(bottom_row)

    full_keyboard = [row for row in keyboard_rows if row]
    return ReplyKeyboardMarkup(keyboard=full_keyboard, resize_keyboard=True)


def admin_panel_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Статистика", callback_data="admin_stats")
    builder.button(text="👥 Клиенты", callback_data="admin_clients_page_0")
    builder.button(text="🧩 Управление Тестом", callback_data="admin_test_menu")
    builder.button(text="🤖 Настройки ИИ", callback_data="admin_ai_settings")
    builder.button(text="⭐️ Подписки", callback_data="admin_subscriptions")
    builder.button(text="📚 База знаний", callback_data="admin_kb_page_0")
    builder.button(text="🎨 Медиа-коллекции", callback_data="admin_collections_page_0")
    builder.button(text="✏️ Контент", callback_data="admin_content")
    builder.button(text="💬 Темы диалогов", callback_data="admin_topics_page_0")
    builder.button(text="🎛️ Кнопки меню", callback_data="admin_manage_buttons")
    builder.button(text="👮‍♂️ Администраторы", callback_data="admin_manage_admins")
    builder.button(text="✉️ Рассылка", callback_data="admin_mailing_menu")
    builder.button(text="🔁 Перезагрузить бота", callback_data="admin_restart_bot")
    builder.adjust(2)
    return builder.as_markup()


def admin_test_menu_keyboard(is_enabled: bool):
    builder = InlineKeyboardBuilder()
    status_text = "✅ Включен" if is_enabled else "❌ Выключен"

    builder.button(text=f"Статус теста: {status_text}", callback_data="admin_test_toggle_status")
    builder.button(text="✏️ Приветствие теста", callback_data="edit_content_test_intro")
    builder.button(text="✏️ Результаты теста", callback_data="edit_content_test_results")
    builder.button(text="✏️ Финал секретного теста", callback_data="edit_content_secret_test_outro")
    builder.button(text="📝 Промпт теста", callback_data="admin_edit_test_prompt")
    builder.button(text="❓ Вопросы (Excel)", callback_data="admin_upload_questions")
    builder.button(text="🔐 Секретные вопросы", callback_data="admin_secret_questions")
    builder.button(text="📖 Истории/Кейсы", callback_data="admin_case_studies_page_0")
    builder.button(text="🔗 Настройка ссылок", callback_data="admin_test_links")
    builder.button(text="⬅️ В админ-панель", callback_data="admin_panel")
    builder.adjust(1)
    return builder.as_markup()

def admin_test_links_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Username Админа (ЛС)", callback_data="set_test_link_admin")
    builder.button(text="✏️ Ссылка на Марафон", callback_data="set_test_link_marathon")
    builder.button(text="⬅️ Назад", callback_data="admin_test_menu")
    builder.adjust(1)
    return builder.as_markup()


def admin_secret_questions_keyboard(questions: list):
    builder = InlineKeyboardBuilder()
    for q in questions:
        builder.button(text=f"🗑️ {q.text[:30]}...", callback_data=f"delete_secret_q_{q.id}")

    builder.button(text="➕ Добавить вопрос", callback_data="add_secret_question")
    builder.button(text="⬅️ Назад", callback_data="admin_test_menu")
    builder.adjust(1)
    return builder.as_markup()


def admin_case_studies_keyboard(cases: list, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()
    for case in cases:
        builder.button(text=f"🗑️ Кейс #{case.id} ({case.text[:20]}...)", callback_data=f"delete_case_{case.id}")

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_case_studies_page_{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_case_studies_page_{page + 1}"))

    if nav_buttons:
        builder.row(*nav_buttons)

    builder.row(InlineKeyboardButton(text="➕ Добавить кейс", callback_data="add_case_study"))
    builder.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_test_menu"))
    return builder.as_markup()

def ai_settings_keyboard(current_provider: str):
    providers = {
        "Deepseek": "ai_provider_Deepseek",
        "Claude": "ai_provider_Claude",
        "Gemini": "ai_provider_Gemini",
        "KIE": "ai_provider_KIE",
        "OpenAI": "ai_provider_OpenAI"
    }
    builder = InlineKeyboardBuilder()
    for name, data in providers.items():
        text = f"✅ {name}" if name == current_provider else name
        builder.button(text=text, callback_data=data)
    builder.button(text="⚙️ Настроить ключи и модели", callback_data="admin_ai_keys")
    builder.button(text="📝 Изменить системный промпт", callback_data="admin_edit_system_prompt")
    builder.button(text="🧩 Общий блок для тем", callback_data="admin_edit_shared_prompt_block")
    builder.button(text="📦 Служебный блок промпта", callback_data="admin_edit_service_prompt_block")
    builder.button(text="⬅️ Назад", callback_data="admin_panel")
    builder.adjust(1)
    return builder.as_markup()


def ai_keys_models_keyboard(current_transcription_provider: str, context_first: int, context_recent: int,
                            current_vision_provider: str, current_vision_model: str,
                            image_generation_provider: str, image_generation_model: str,
                            image_edit_provider: str, image_edit_model: str,
                            kie_credit_alert_threshold: float,
                            temperature: float = 0.7, memory_mode: str = "reset",
                            fallback_provider: str | None = None, fallback_model: str | None = None,
                            use_proxy: bool = True):
    def short_model(model: str, limit: int = 16) -> str:
        if len(model) <= limit:
            return model
        return f"{model[:limit - 1]}…"

    builder = InlineKeyboardBuilder()

    builder.button(text="🔑 Deepseek", callback_data="set_key_Deepseek")
    builder.button(text="🔑 Claude", callback_data="set_key_Claude")
    builder.button(text="🔑 Gemini", callback_data="set_key_Gemini")
    builder.button(text="🔑 KIE", callback_data="set_key_KIE")
    builder.button(text="🔑 OpenAI", callback_data="set_key_OpenAI")

    builder.button(text="🧠 Deepseek", callback_data="view_models_Deepseek")
    builder.button(text="🧠 Claude", callback_data="view_models_Claude")
    builder.button(text="🧠 Gemini", callback_data="view_models_Gemini")
    builder.button(text="🧠 KIE", callback_data="view_models_KIE")
    builder.button(text="🧠 OpenAI", callback_data="view_models_OpenAI")

    builder.button(text=f"📌 Первые: {context_first}", callback_data="set_context_first")
    builder.button(text=f"🔄 Последние: {context_recent}", callback_data="set_context_recent")

    trans_label = f"🗣️ Аудио: {current_transcription_provider}" if current_transcription_provider != 'None' else "🗣️ Аудио: выкл"
    builder.button(text=trans_label, callback_data="admin_toggle_transcription")
    builder.button(text="⏱️ Лимит аудио", callback_data="set_audio_limit")

    threshold_label = int(kie_credit_alert_threshold) if float(kie_credit_alert_threshold).is_integer() else round(kie_credit_alert_threshold, 2)
    builder.button(text=f"💳 KIE порог: {threshold_label}", callback_data="set_kie_credit_threshold")
    builder.button(text=f"🌡️ Температура: {round(temperature, 2)}", callback_data="set_temperature")
    builder.button(
        text=f"🧠 Память: {memory_mode_label(memory_mode)}",
        callback_data="toggle_preserve_topic_context"
    )
    proxy_status = "✅ ВКЛ" if use_proxy else "❌ ВЫКЛ"
    builder.button(text=f"🌍 Прокси Deepseek: {proxy_status}", callback_data="admin_toggle_proxy")
    fb_label = f"🔄 Резерв: {fallback_provider}" if fallback_provider else "🔄 Резерв: выкл"
    builder.button(text=fb_label, callback_data="admin_toggle_fallback")
    if fallback_provider:
        fb_model_short = short_model(fallback_model) if fallback_model else "не задана"
        builder.button(text=f"Модель: {fb_model_short}", callback_data="admin_change_fallback_model")

    builder.button(text=f"👁️ Фото: {current_vision_provider}",
                   callback_data="admin_toggle_vision")
    builder.button(text=f"Модель: {short_model(current_vision_model)}", callback_data="admin_change_vision_model")

    builder.button(text=f"🖼 Ген: {image_generation_provider}",
                   callback_data="admin_toggle_image_generation")
    builder.button(text=f"Модель: {short_model(image_generation_model)}", callback_data="admin_change_image_generation_model")

    builder.button(text=f"🎨 Редакт: {image_edit_provider}",
                   callback_data="admin_toggle_image_edit")
    builder.button(text=f"Модель: {short_model(image_edit_model)}", callback_data="admin_change_image_edit_model")

    timeout_val = getattr(ai_config, "fallback_timeout", 60) if 'ai_config' in locals() else 60
    # Wait, ai_config is not passed! I can't do this easily. I will just pass it to the button text?
    # No, I can't read ai_config here.
    builder.button(text="⏱️ Таймаут ИИ", callback_data="set_ai_timeout")

    builder.button(text="⬅️ Назад", callback_data="admin_ai_settings")

    # Layout: keys 2+2+1, models 2+2+1, context 2, audio+limit 2,
    # KIE+temp 2, mem+proxy 2, fallback 1 (or 2), vision 2, gen 2, edit 2, timeout 1, back 1
    if fallback_provider:
        builder.adjust(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1)
    else:
        builder.adjust(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1)
    return builder.as_markup()


def model_selection_keyboard(provider: str, models: dict):
    builder = InlineKeyboardBuilder()
    for model_key, model_info in models.items():
        builder.button(text=model_info['name'], callback_data=f"set_model_{provider}_{model_key}")
    builder.button(text="⬅️ Назад", callback_data="admin_ai_keys")
    builder.adjust(1)
    return builder.as_markup()


def clients_paginator_keyboard(page: int, total_pages: int, clients: list, is_searching: bool = False,
                               export_mode: bool = False, selected_ids: list = None):
    if selected_ids is None:
        selected_ids = []

    builder = InlineKeyboardBuilder()
    for client in clients:
        display_name = client.name or client.first_name
        status = "✅ " if client.id in selected_ids else ""
        cb_data = f"toggle_export_{client.id}_{page}" if export_mode else f"view_client_{client.id}"
        builder.button(text=f"{status}{display_name} (@{client.username or 'N/A'})",
                       callback_data=cb_data)

    builder.adjust(1)

    nav_buttons = []
    prefix = "admin_export_page_" if export_mode else "admin_clients_page_"

    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"{prefix}{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"{prefix}{page + 1}"))

    builder.row(*nav_buttons)

    if export_mode:
        builder.row(InlineKeyboardButton(text="☑️ Выбрать всех (кроме админов)", callback_data="export_select_all_no_admins"))
        builder.row(
            InlineKeyboardButton(text="📥 Экспортировать выбранных", callback_data="admin_export_confirm_options"))
        builder.row(InlineKeyboardButton(text="📥 Экспортировать ВСЕХ", callback_data="admin_export_all_confirm"))
        builder.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin_clients_reset_export"))
    else:
        if is_searching:
            builder.row(InlineKeyboardButton(text="❌ Сбросить поиск", callback_data="admin_clients_reset_search"))
        else:
            builder.row(InlineKeyboardButton(text="🔍 Поиск", callback_data="admin_clients_start_search"))

        builder.row(InlineKeyboardButton(text="📦 Режим экспорта истории", callback_data="admin_export_mode_start"))
        builder.row(InlineKeyboardButton(text="⬅️ В админ-панель", callback_data="admin_panel"))

    return builder.as_markup()


def client_profile_keyboard(user_id: int, is_target_admin: bool, target_can_view: bool, is_owner: bool):
    builder = InlineKeyboardBuilder()
    builder.button(text="💳 Платежная инфо", callback_data=f"client_payment_info_{user_id}")
    builder.button(text="📜 История диалога", callback_data=f"client_history_{user_id}")
    builder.button(text="📥 Скачать историю", callback_data=f"download_history_{user_id}")
    builder.button(text="🎁 Сбросить промокоды", callback_data=f"reset_user_promos_{user_id}")
    builder.button(text="🔄 Сбросить подписку", callback_data=f"admin_reset_sub_{user_id}")
    builder.button(text="🗑️ Удалить историю", callback_data=f"admin_delete_client_history_{user_id}")
    builder.button(text="♻️ Сбросить аккаунт", callback_data=f"admin_reset_client_{user_id}")

    if is_owner and is_target_admin:
        btn_text = "❌ Забрать доступ к истории" if target_can_view else "✅ Дать доступ к истории"
        builder.button(text=btn_text, callback_data=f"toggle_history_access_{user_id}")

    builder.button(text="⬅️ К списку клиентов", callback_data="admin_clients_page_0")
    builder.adjust(1)
    return builder.as_markup()


def knowledge_base_paginator_keyboard(page: int, total_pages: int, files: list):
    builder = InlineKeyboardBuilder()
    for file in files:
        builder.button(text=f"📄 {file.filename}", callback_data=f"noop_kb_{file.id}")

        gen_status = "✅" if file.use_in_general_mode else "⭕️"
        builder.button(text=f"Gen: {gen_status}", callback_data=f"toggle_kb_general_{page}_{file.id}")

        builder.button(text="🗑️", callback_data=f"delete_kb_{file.id}")

    builder.adjust(3)

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_kb_page_{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_kb_page_{page + 1}"))

    if nav_buttons:
        builder.row(*nav_buttons)

    builder.row(InlineKeyboardButton(text="➕ Добавить файл", callback_data="add_kb_file"))
    builder.row(InlineKeyboardButton(text="⬅️ Назад в админ-панель", callback_data="admin_panel"))
    return builder.as_markup()


async def content_management_keyboard():
    builder = InlineKeyboardBuilder()

    async with async_session_maker() as session:
        start_msg = await session.get(Content, "start_message")
        if start_msg:
            builder.button(text="✏️ Приветствие (/start)", callback_data="edit_content_start_message")

        disclaimer_msg = await session.get(Content, "disclaimer")
        if disclaimer_msg:
            status = "✅" if disclaimer_msg.is_visible else "❌"
            builder.button(text=f"✏️ Дисклеймер {status}", callback_data="edit_content_disclaimer")

        stmt = select(Content).where(Content.button_title != None).order_by(Content.key)
        result = await session.execute(stmt)
        content_items = result.scalars().all()

    for item in content_items:
        builder.button(text=f"✏️ {item.button_title}", callback_data=f"edit_content_{item.key}")

    builder.button(text="⬅️ Назад", callback_data="admin_panel")
    builder.adjust(1)
    return builder.as_markup()


def back_to_admin_panel():
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ В админ-панель", callback_data="admin_panel")
    return builder.as_markup()


def back_to_previous_menu(callback_data: str):
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ Назад", callback_data=f"cancel_state_{callback_data}")
    return builder.as_markup()


def back_to_client_profile(user_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ К профилю", callback_data=f"view_client_{user_id}")
    return builder.as_markup()

def finish_upload_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Завершить добавление", callback_data="finish_kb_upload")
    return builder.as_markup()


def balance_refilled_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Я пополнил баланс", callback_data="admin_balance_refilled")
    return builder.as_markup()


def user_history_keyboard(page: int, total_pages: int, for_admin_user_id: int | None = None):
    builder = InlineKeyboardBuilder()

    if for_admin_user_id:
        base_callback = f"admin_history_{for_admin_user_id}"
    else:
        base_callback = "user_history"

    nav_buttons = []

    if total_pages > 1:
        prev_page = page - 1 if page > 0 else total_pages - 1
        next_page = page + 1 if page < total_pages - 1 else 0

        nav_buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"{base_callback}_{prev_page}"))
        nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
        nav_buttons.append(InlineKeyboardButton(text="Вперёд ➡️", callback_data=f"{base_callback}_{next_page}"))
    else:
        nav_buttons.append(InlineKeyboardButton(text=f"1/1", callback_data="noop"))

    builder.row(*nav_buttons)

    if total_pages > 1:
        builder.row(
            InlineKeyboardButton(text="⏮ В начало", callback_data=f"{base_callback}_0"),
            InlineKeyboardButton(text="В конец ⏭", callback_data=f"{base_callback}_{total_pages - 1}"),
        )

    if for_admin_user_id:
        builder.row(InlineKeyboardButton(text="⬅️ К профилю", callback_data=f"view_client_{for_admin_user_id}"))

    return builder.as_markup()

def confirm_delete_history_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="🗑️ Да, удалить", callback_data="delete_history_confirm")
    builder.button(text="❌ Отмена", callback_data="delete_history_cancel")
    return builder.as_markup()


def topic_reset_options_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="Начать новый диалог в данной теме", callback_data="reset_topic_keep")
    builder.button(text="Перейти в основной диалог", callback_data="reset_topic_to_main")
    builder.button(text="Отмена", callback_data="delete_history_cancel")
    builder.adjust(1)
    return builder.as_markup()


def confirm_disclaimer_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Я понимаю и принимаю", callback_data="disclaimer_accepted")
    return builder.as_markup()


def confirm_delete_kb_keyboard(file_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Да, удалить навсегда", callback_data=f"confirm_delete_kb_{file_id}")
    builder.button(text="❌ Нет, отмена", callback_data="admin_kb_page_0")
    builder.adjust(1)
    return builder.as_markup()


def content_editing_keyboard(content_key: str, media_items: list, content_order: str = 'media_top',
                             is_visible: bool = True):
    builder = InlineKeyboardBuilder()

    order_text = "🖼 Медиа сверху ⬇️ Текст снизу" if content_order == 'media_top' else "📝 Текст сверху ⬇️ Медиа снизу"
    builder.button(text=order_text, callback_data=f"toggle_order_{content_key}")

    visibility_text = "👁 Скрыть раздел" if is_visible else "👁 Показать раздел"
    builder.button(text=visibility_text, callback_data=f"toggle_content_visibility_{content_key}")

    if content_key == 'start_message':
        builder.button(text="🔘 Название кнопки действия", callback_data=f"edit_content_btn_text_{content_key}")
        builder.button(text="📩 Текст кнопки действия", callback_data=f"edit_content_btn_payload_{content_key}")
        builder.button(text="🗑️ Удалить кнопку действия", callback_data=f"clear_content_btn_{content_key}")

    for index, media in enumerate(media_items):
        file_type_emoji = "🖼️" if media['type'] == 'photo' else "📹"
        builder.button(
            text=f"❌ Удалить {file_type_emoji} #{index + 1}",
            callback_data=f"delete_media_{content_key}_{index}"
        )

    builder.adjust(1)

    builder.row(
        InlineKeyboardButton(text="✅ Сохранить и выйти", callback_data=f"save_content_{content_key}"),
        InlineKeyboardButton(text="❌ Отмена", callback_data=f"cancel_content_edit_{content_key}")
    )
    return builder.as_markup()


def system_prompt_keyboard(prompt_too_long: bool = False):
    builder = InlineKeyboardBuilder()
    if prompt_too_long:
        builder.button(text="📥 Скачать полный промпт", callback_data="download_system_prompt")
    builder.button(text="⬅️ Назад", callback_data="cancel_state_admin_ai_settings")
    builder.adjust(1)
    return builder.as_markup()


def prompt_block_keyboard(download_callback: str):
    builder = InlineKeyboardBuilder()
    builder.button(text="📥 Скачать блок", callback_data=download_callback)
    builder.button(text="⬅️ Назад", callback_data="cancel_state_admin_ai_settings")
    builder.adjust(1)
    return builder.as_markup()


def select_topic_keyboard(topics: list, current_topic_id: int | None):
    builder = InlineKeyboardBuilder()
    for topic in topics:
        text = f"✅ {topic.name}" if topic.id == current_topic_id else topic.name
        builder.button(text=text, callback_data=f"select_topic_{topic.id}")
    builder.adjust(1)
    builder.row(InlineKeyboardButton(text="🏠 Перейти в основной диалог", callback_data="reset_topic"))
    builder.row(InlineKeyboardButton(text="❌ Отмена", callback_data="topic_select_cancel"))
    return builder.as_markup()


def topics_admin_list_keyboard(topics: list, page: int, total_pages: int, config):
    builder = InlineKeyboardBuilder()
    for topic in topics:
        status = "🔒" if getattr(topic, 'admin_only', False) else ("🟢" if topic.is_active else "⚪️")
        builder.row(
            InlineKeyboardButton(text=f"{status} {topic.name}", callback_data=f"edit_topic_{topic.id}"),
            InlineKeyboardButton(text="⬆️", callback_data=f"move_topic_up_{topic.id}_{page}"),
            InlineKeyboardButton(text="⬇️", callback_data=f"move_topic_down_{topic.id}_{page}")
        )

    builder.adjust(*([3] * len(topics)), 1)

    if topics and total_pages > 1:
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_topics_page_{page - 1}"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_topics_page_{page + 1}"))
        if nav_buttons:
            builder.row(*nav_buttons)

    builder.row(InlineKeyboardButton(text="➕ Создать новую тему", callback_data="create_topic"))
    status_text = "✅ Включены у всех" if config.topics_enabled else "❌ Выключены у всех"
    builder.row(InlineKeyboardButton(text=f"Темы диалогов: {status_text}", callback_data="admin_toggle_topics"))
    topics_pos = "⬆️ Сверху" if config.topics_btn_on_top else "⬇️ В списке"
    builder.row(InlineKeyboardButton(text=f"Кнопка тем: {topics_pos}", callback_data="admin_toggle_topics_on_top"))
    builder.row(InlineKeyboardButton(text=f"Название кнопки тем: {config.topics_btn_name}",
                                     callback_data="admin_rename_topics_btn"))
    builder.row(InlineKeyboardButton(text="⬅️ В админ-панель", callback_data="admin_panel"))
    return builder.as_markup()


def edit_topic_keyboard(topic_id: int, is_active: bool, in_menu: bool = False, in_list: bool = True, admin_only: bool = False):
    builder = InlineKeyboardBuilder()
    status_text = "⚪️ Сделать неактивной" if is_active else "🟢 Сделать активной"
    admin_only_text = "🔒 Только для админов: ВКЛ" if admin_only else "🔓 Только для админов: ВЫКЛ"
    menu_text = "❌ Убрать из меню" if in_menu else "✅ Показать в меню"
    list_text = "❌ Убрать из списка" if in_list else "✅ Показать в списке"

    builder.button(text="✏️ Название", callback_data=f"edit_topic_name_{topic_id}")
    builder.button(text=status_text, callback_data=f"toggle_topic_activity_{topic_id}")
    builder.button(text=admin_only_text, callback_data=f"toggle_topic_admin_only_{topic_id}")
    builder.button(text=menu_text, callback_data=f"toggle_topic_display_menu_{topic_id}")
    builder.button(text=list_text, callback_data=f"toggle_topic_display_list_{topic_id}")

    builder.button(text="📝 Системный промпт", callback_data=f"edit_topic_prompt_{topic_id}")
    builder.button(text="🎲 Случайные фразы", callback_data=f"topic_random_phrases_{topic_id}")

    builder.button(text="💬 Приветственное сообщение", callback_data=f"edit_topic_intro_{topic_id}")
    builder.button(text="🔘 Название кнопки действия", callback_data=f"edit_topic_btn_text_{topic_id}")
    builder.button(text="📩 Текст кнопки действия", callback_data=f"edit_topic_btn_payload_{topic_id}")
    builder.button(text="🗑️ Удалить кнопку действия", callback_data=f"clear_topic_btn_{topic_id}")

    builder.button(text="📎 Привязать файлы БЗ", callback_data=f"assign_kb_topic_{topic_id}_page_0")

    builder.button(text="📁 Медиа-файлы темы", callback_data=f"admin_topic_media_{topic_id}")
    builder.button(text="🎨 Привязать коллекции", callback_data=f"assign_coll_topic_{topic_id}_page_0")

    builder.button(text="🗑️ Удалить тему", callback_data=f"delete_topic_{topic_id}")
    builder.button(text="⬅️ К списку тем", callback_data="admin_topics_page_0")

    builder.adjust(1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1)
    return builder.as_markup()


def confirm_delete_topic_keyboard(topic_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Да, удалить тему", callback_data=f"confirm_delete_topic_{topic_id}")
    builder.button(text="❌ Отмена", callback_data=f"edit_topic_{topic_id}")
    return builder.as_markup()


def assign_kb_to_topic_keyboard(topic_id: int, all_files: list, assigned_file_ids: set, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()
    for file in all_files:
        is_assigned = file.id in assigned_file_ids
        text = f"✅ {file.filename}" if is_assigned else f"⭕️ {file.filename}"
        action = "remove" if is_assigned else "add"
        callback_data = f"kb_topic_{action}_{topic_id}_{file.id}_{page}"
        builder.button(text=text, callback_data=callback_data)
    builder.adjust(1)

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"assign_kb_topic_{topic_id}_page_{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"assign_kb_topic_{topic_id}_page_{page + 1}"))

    if nav_buttons:
        builder.row(*nav_buttons)

    builder.row(InlineKeyboardButton(text="⬅️ Назад к теме", callback_data=f"edit_topic_{topic_id}"))
    return builder.as_markup()


def subscription_info_keyboard(sub_info: dict | None, referral_info: dict | None = None):
    builder = InlineKeyboardBuilder()
    builder.button(text="💳 Оформить/Сменить тариф", callback_data="sub_select_plan")
    if sub_info and sub_info.get('allow_auto_renewal', True):
        if sub_info['auto_renewal']:
            builder.button(text="❌ Отменить автопродление", callback_data="sub_toggle_renewal")
        else:
            builder.button(text="✅ Включить автопродление", callback_data="sub_toggle_renewal")
    builder.button(text="🎁 Ввести промокод", callback_data="sub_enter_promo")

    if referral_info and referral_info.get('enabled'):
        builder.button(
            text=referral_info.get('sub_btn_name', '🤝 Бонус за приглашение'),
            callback_data="referral_sub_info"
        )

    builder.adjust(1)
    return builder.as_markup()


def confirm_client_action_keyboard(confirm_callback: str, cancel_callback: str):
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Подтвердить", callback_data=confirm_callback)
    builder.button(text="❌ Отмена", callback_data=cancel_callback)
    builder.adjust(1)
    return builder.as_markup()


def subscription_retry_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="💳 Списать сейчас", callback_data="sub_retry_now")
    builder.button(text="🔄 Отменить и оформить заново", callback_data="sub_cancel_retry")
    builder.button(text="🎁 Ввести промокод", callback_data="sub_enter_promo")
    builder.adjust(1)
    return builder.as_markup()


def subscription_pending_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="🔄 Проверить статус", callback_data="sub_retry_now")
    builder.button(text="🔄 Отменить и оформить заново", callback_data="sub_cancel_retry")
    builder.button(text="🎁 Ввести промокод", callback_data="sub_enter_promo")
    builder.adjust(1)
    return builder.as_markup()


def plan_selection_keyboard(plans: list, global_discount_percent: int = 0, user_promos: list = None):
    if user_promos is None:
        user_promos = []

    builder = InlineKeyboardBuilder()

    for plan in plans:
        price = plan.price
        text = f"{plan.name}"

        duration_unit_text = "дн." if plan.duration_unit == 'days' else "мес."
        text += f" ({plan.duration_value} {duration_unit_text})"

        plan_discount_percent = global_discount_percent
        plan_specific_promo = next((
            p for p in user_promos
            if not p.applies_to_all_plans and any(ap.id == plan.id for ap in p.applicable_plans)
        ), None)

        if plan_specific_promo:
            plan_discount_percent = plan_specific_promo.discount_percent
        elif not plan_specific_promo and plan_discount_percent == 0:
            all_plans_promo = next((
                p for p in user_promos if p.applies_to_all_plans
            ), None)
            if all_plans_promo:
                plan_discount_percent = all_plans_promo.discount_percent

        has_discount = False
        if plan_discount_percent > 0 and not plan.is_trial:
            has_discount = True
            price = price * (1 - plan_discount_percent / 100)

        text += f" - {price:.2f} руб."

        if plan.is_trial and plan.upgrades_to_plan:
            upgrade_duration_unit_text = "дн." if plan.upgrades_to_plan.duration_unit == 'days' else "мес."

            upgrade_price = plan.upgrades_to_plan.price
            upgrade_plan_id = plan.upgrades_to_plan.id
            upgrade_plan_discount_percent = global_discount_percent

            upgrade_specific_promo = next((
                p for p in user_promos
                if not p.applies_to_all_plans and any(ap.id == upgrade_plan_id for ap in p.applicable_plans)
            ), None)

            if upgrade_specific_promo:
                upgrade_plan_discount_percent = upgrade_specific_promo.discount_percent
            elif not upgrade_specific_promo and upgrade_plan_discount_percent == 0:
                all_plans_promo = next((
                    p for p in user_promos if p.applies_to_all_plans
                ), None)
                if all_plans_promo:
                    upgrade_plan_discount_percent = all_plans_promo.discount_percent

            if upgrade_plan_discount_percent > 0:
                upgrade_price = upgrade_price * (1 - upgrade_plan_discount_percent / 100)

            text += f" (далее {upgrade_price:.2f} руб./{plan.upgrades_to_plan.duration_value} {upgrade_duration_unit_text})"

        if has_discount:
            text += f" (со ск. {int(plan_discount_percent)}%)"

        builder.button(text=text, callback_data=f"sub_pay_{plan.id}")
    builder.button(text="⬅️ Назад", callback_data="back_to_sub_info")
    builder.adjust(1)
    return builder.as_markup()


def promo_plan_selection_keyboard(plans: list, global_discount_percent: int = 0, user_promos: list = None):
    if user_promos is None:
        user_promos = []

    builder = InlineKeyboardBuilder()

    for plan in plans:
        price = plan.price
        text = f"{plan.name}"

        duration_unit_text = "дн." if plan.duration_unit == 'days' else "мес."
        text += f" ({plan.duration_value} {duration_unit_text})"

        plan_discount_percent = global_discount_percent
        plan_specific_promo = next((
            p for p in user_promos
            if not p.applies_to_all_plans and any(ap.id == plan.id for ap in p.applicable_plans)
        ), None)

        if plan_specific_promo:
            plan_discount_percent = plan_specific_promo.discount_percent
        elif not plan_specific_promo and plan_discount_percent == 0:
            all_plans_promo = next((
                p for p in user_promos if p.applies_to_all_plans
            ), None)
            if all_plans_promo:
                plan_discount_percent = all_plans_promo.discount_percent

        has_discount = False
        if plan_discount_percent > 0 and not plan.is_trial:
            has_discount = True
            price = price * (1 - plan_discount_percent / 100)

        text += f" - {price:.2f} руб."

        if plan.is_trial and plan.upgrades_to_plan:
            upgrade_duration_unit_text = "дн." if plan.upgrades_to_plan.duration_unit == 'days' else "мес."

            upgrade_price = plan.upgrades_to_plan.price
            upgrade_plan_id = plan.upgrades_to_plan.id
            upgrade_plan_discount_percent = global_discount_percent

            upgrade_specific_promo = next((
                p for p in user_promos
                if not p.applies_to_all_plans and any(ap.id == upgrade_plan_id for ap in p.applicable_plans)
            ), None)

            if upgrade_specific_promo:
                upgrade_plan_discount_percent = upgrade_specific_promo.discount_percent
            elif not upgrade_specific_promo and upgrade_plan_discount_percent == 0:
                all_plans_promo = next((
                    p for p in user_promos if p.applies_to_all_plans
                ), None)
                if all_plans_promo:
                    upgrade_plan_discount_percent = all_plans_promo.discount_percent

            if upgrade_plan_discount_percent > 0:
                upgrade_price = upgrade_price * (1 - upgrade_plan_discount_percent / 100)

            text += f" (далее {upgrade_price:.2f} руб./{plan.upgrades_to_plan.duration_value} {upgrade_duration_unit_text})"

        if has_discount:
            text += f" (со ск. {int(plan_discount_percent)}%)"

        builder.button(text=text, callback_data=f"sub_pay_{plan.id}")

    builder.adjust(1)
    return builder.as_markup()


async def payment_provider_keyboard(plan_id: int, final_price: float):
    builder = InlineKeyboardBuilder()

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)

    if config:
        if config.yookassa_shop_id and config.yookassa_secret_key:
            builder.button(text="ЮKassa", callback_data=f"pay_yookassa_{plan_id}_{final_price}")
        if config.telegram_pay_token:
            builder.button(text="Telegram Pay", callback_data=f"pay_tg_{plan_id}_{final_price}")
        if config.robokassa_merchant_login and config.robokassa_password_1:
            builder.button(text="Robokassa", callback_data=f"pay_robokassa_{plan_id}_{final_price}")

    builder.button(text="⬅️ Назад к выбору тарифа", callback_data="sub_select_plan")
    builder.adjust(1)
    return builder.as_markup()

def admin_subscriptions_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="📈 Тарифные планы", callback_data="admin_plans")
    builder.button(text="🎁 Промокоды", callback_data="admin_promocodes")
    builder.button(text="⚙️ Настройки платежей", callback_data="admin_payment_settings")
    builder.button(text="👥 Подписчики", callback_data="admin_subs_0_all")
    builder.button(text="📋 Лог платежей", callback_data="admin_plog_0_all")
    builder.button(text="👫 Рефералы", callback_data="admin_referral_menu")
    builder.button(text="⬅️ В админ-панель", callback_data="admin_panel")
    builder.adjust(1)
    return builder.as_markup()

def admin_plans_keyboard(plans: list):
    builder = InlineKeyboardBuilder()
    for plan in plans:
        status = "🔒" if getattr(plan, 'admin_only', False) else ("🟢" if plan.is_active else "⚪️")

        duration_unit_text = "дн." if plan.duration_unit == 'days' else "мес."
        plan_text = f"{plan.name} ({plan.duration_value} {duration_unit_text}) ({plan.price} руб)"

        if plan.is_trial and plan.upgrades_to_plan:
            plan_text += f" ➡️ «{plan.upgrades_to_plan.name}»"

        builder.button(text=f"{status} {plan_text}", callback_data=f"admin_edit_plan_{plan.id}")
    builder.button(text="➕ Создать новый тариф", callback_data="admin_create_plan")
    builder.button(text="⬅️ Назад", callback_data="admin_subscriptions")
    builder.adjust(1)
    return builder.as_markup()


def admin_edit_plan_keyboard(plan_id: int, is_active: bool, is_trial: bool, allow_auto_renewal: bool = True, admin_only: bool = False):
    builder = InlineKeyboardBuilder()
    status_text = "⚪️ Сделать неактивным" if is_active else "🟢 Сделать активным"
    admin_only_text = "🔒 Только для админов: ВКЛ" if admin_only else "🔓 Только для админов: ВЫКЛ"
    trial_status_text = "❌ Сделать обычным" if is_trial else "⭐️ Сделать пробным"
    renewal_text = "🔄 Автопродление: ВКЛ → ВЫКЛ" if allow_auto_renewal else "🔄 Автопродление: ВЫКЛ → ВКЛ"

    builder.button(text="✏️ Название", callback_data=f"edit_plan_field_name_{plan_id}")
    builder.button(text="📝 Описание", callback_data=f"edit_plan_field_description_{plan_id}")
    builder.button(text="💰 Цена", callback_data=f"edit_plan_field_price_{plan_id}")
    builder.button(text="⏳ Длительность (число)", callback_data=f"edit_plan_field_duration_value_{plan_id}")
    builder.button(text="📅 Длительность (ед.)", callback_data=f"edit_plan_field_duration_unit_{plan_id}")
    builder.adjust(2)

    builder.row(InlineKeyboardButton(text=trial_status_text, callback_data=f"toggle_plan_is_trial_{plan_id}"))
    if is_trial:
        builder.row(InlineKeyboardButton(text="🎯 Назначить тариф для перехода",
                                         callback_data=f"set_plan_upgrade_target_{plan_id}"))
        builder.row(InlineKeyboardButton(text="⏳ Кулдаун (0=нельзя, >0=дни)",
                                         callback_data=f"edit_plan_field_trial_cooldown_days_{plan_id}"))

    builder.row(InlineKeyboardButton(text=renewal_text, callback_data=f"toggle_plan_allow_auto_renewal_{plan_id}"))
    builder.row(InlineKeyboardButton(text=admin_only_text, callback_data=f"toggle_plan_admin_only_{plan_id}"))
    builder.row(InlineKeyboardButton(text=status_text, callback_data=f"toggle_plan_activity_{plan_id}"))
    builder.row(InlineKeyboardButton(text="🗑️ Удалить тариф", callback_data=f"admin_delete_plan_{plan_id}"))
    builder.row(InlineKeyboardButton(text="⬅️ К списку тарифов", callback_data="admin_plans"))

    return builder.as_markup()


def admin_select_upgrade_plan_keyboard(plans: list, current_plan_id: int):
    builder = InlineKeyboardBuilder()
    for plan in plans:
        builder.button(text=f"{plan.name} ({plan.price} руб)", callback_data=f"set_upgrade_plan_{plan.id}")
    builder.adjust(1)
    builder.row(InlineKeyboardButton(text="⬅️ Отмена", callback_data=f"cancel_state_admin_edit_plan_{current_plan_id}"))
    return builder.as_markup()


def admin_edit_promo_keyboard(promo_id: int, is_active: bool):
    builder = InlineKeyboardBuilder()
    status_text = "⚪️ Сделать неактивным" if is_active else "🟢 Сделать активным"

    builder.button(text="✏️ Изменить скидку (%)", callback_data=f"edit_promo_field_discount_percent_{promo_id}")
    builder.button(text="🎁 Изменить бонусные дни (Триал)", callback_data=f"edit_promo_field_free_days_{promo_id}")
    builder.button(text="🔄 Изменить кол-во использований", callback_data=f"edit_promo_field_max_uses_{promo_id}")
    builder.button(text="📎 Привязать тарифы", callback_data=f"admin_assign_promo_{promo_id}_page_0")
    builder.adjust(1)

    builder.row(InlineKeyboardButton(text=status_text, callback_data=f"toggle_promo_activity_{promo_id}"))
    builder.row(InlineKeyboardButton(text="🗑️ Удалить промокод", callback_data=f"admin_delete_promo_{promo_id}"))
    builder.row(InlineKeyboardButton(text="⬅️ К списку промокодов", callback_data="admin_promocodes"))

    return builder.as_markup()


def admin_promocodes_keyboard(codes: list):
    builder = InlineKeyboardBuilder()
    for code in codes:
        status = "🟢" if code.is_active else "⚪️"
        builder.button(text=f"{status} {code.code} ({code.times_used}/{code.max_uses})", callback_data=f"admin_edit_promo_{code.id}")
    builder.button(text="➕ Создать промокод", callback_data="admin_create_promo")
    builder.button(text="⬅️ Назад", callback_data="admin_subscriptions")
    builder.adjust(1)
    return builder.as_markup()


def admin_payment_settings_keyboard(config):
    notif_status = "✅ Включены" if config.notifications_enabled else "❌ Выключены"
    sub_status = "✅ Включены" if config.subscriptions_enabled else "❌ Выключены"
    bonus_days = config.welcome_bonus_days

    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Статистика и Доход", callback_data="admin_payment_stats")
    builder.button(text="🔑 API Ключи", callback_data="admin_payment_keys_menu")
    builder.button(text=f"Уведомления о покупках: {notif_status}", callback_data="admin_toggle_sub_notifs")
    builder.button(text=f"Система подписок: {sub_status}", callback_data="admin_toggle_subscriptions")
    builder.button(text=f"🎁 Бонус при входе: {bonus_days} дн.", callback_data="admin_set_welcome_bonus")

    builder.button(text="⬅️ Назад", callback_data="admin_subscriptions")
    builder.adjust(1)
    return builder.as_markup()


def admin_management_keyboard(admins: list):
    builder = InlineKeyboardBuilder()

    for admin in admins:
        if admin.id not in OWNER_IDS:
            display_name = admin.first_name or f"ID: {admin.id}"
            username = f"(@{admin.username})" if admin.username else ""
            builder.button(text=f"👮‍♂️ {display_name} {username}", callback_data=f"admin_panel_profile_{admin.id}")

    builder.adjust(1)
    builder.row(
        InlineKeyboardButton(text="➕ Добавить", callback_data="admin_add_admin")
    )
    builder.row(InlineKeyboardButton(text="⬅️ В админ-панель", callback_data="admin_panel"))
    return builder.as_markup()


def admin_view_admin_profile_keyboard(admin_id: int, can_view_history: bool):
    builder = InlineKeyboardBuilder()

    history_btn_text = "❌ Забрать доступ к истории" if can_view_history else "✅ Дать доступ к истории"

    builder.button(text=history_btn_text, callback_data=f"admin_panel_toggle_history_{admin_id}")
    builder.button(text="➖ Разжаловать (забрать права)", callback_data=f"admin_panel_revoke_{admin_id}")
    builder.button(text="⬅️ К списку админов", callback_data="admin_manage_admins")
    builder.adjust(1)
    return builder.as_markup()

def mailing_menu_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="🚀 Создать рассылку", callback_data="mailing_create")
    builder.button(text="🎂 Шаблон ДР", callback_data="mailing_birthday_template")
    builder.button(text="📜 История рассылок", callback_data="mailing_history_page_0")
    builder.button(text="⬅️ В админ-панель", callback_data="admin_panel")
    builder.adjust(1)
    return builder.as_markup()

def mailing_audience_keyboard():
    builder = InlineKeyboardBuilder()
    for key, text in MAILING_AUDIENCE_LABELS.items():
        builder.button(text=text, callback_data=f"mailing_audience_{key}")
    builder.button(text="⬅️ Назад", callback_data="admin_mailing_menu")
    builder.adjust(1)
    return builder.as_markup()

def mailing_content_keyboard(audience: str):
    builder = InlineKeyboardBuilder()
    if audience == "birthday_today":
        builder.button(text="✨ Подставить шаблон ДР", callback_data="mailing_use_birthday_template")
    back_callback = "mailing_birthday_template" if audience == "birthday_today" else "mailing_create"
    back_text = "⬅️ Назад к шаблону ДР" if audience == "birthday_today" else "⬅️ Назад к выбору аудитории"
    builder.button(text=back_text, callback_data=back_callback)
    builder.adjust(1)
    return builder.as_markup()

def mailing_media_position_keyboard(audience: str | None = None):
    builder = InlineKeyboardBuilder()
    if audience == "birthday_today":
        builder.button(text="✨ Подставить шаблон ДР", callback_data="mailing_use_birthday_template")
    builder.button(text="🖼 Медиа сверху", callback_data="mailing_pos_media_top")
    builder.button(text="📝 Текст сверху", callback_data="mailing_pos_text_top")
    builder.button(text="⬅️ Назад", callback_data="mailing_edit_content")
    builder.adjust(1)
    return builder.as_markup()

def mailing_confirmation_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Отправить", callback_data="mailing_confirm_send")
    builder.button(text="✏️ Изменить", callback_data="mailing_edit_content")
    builder.button(text="❌ Отмена", callback_data="admin_mailing_menu")
    builder.adjust(2)
    return builder.as_markup()


def manage_buttons_keyboard(buttons: list, change_name_status: bool):
    builder = InlineKeyboardBuilder()
    sorted_buttons = sorted(buttons, key=lambda x: (x.sort_order if x.sort_order is not None else 0, x.key))

    name_btn_emoji = "✅" if change_name_status else "❌"
    builder.row(InlineKeyboardButton(
        text=f"{name_btn_emoji} Кнопка '⚙️ Настройки'",
        callback_data="admin_toggle_change_name_btn_from_menu"
    ))

    for button in sorted_buttons:
        status = "✅" if button.is_visible else "❌"
        builder.row(InlineKeyboardButton(
            text=f"{status} {button.button_title}",
            callback_data=f"edit_button_visibility_{button.key}"
        ))
        builder.row(
            InlineKeyboardButton(text="✏️ Переим.", callback_data=f"edit_button_title_{button.key}"),
            InlineKeyboardButton(text="⬆️", callback_data=f"move_btn_up_{button.key}"),
            InlineKeyboardButton(text="⬇️", callback_data=f"move_btn_down_{button.key}"),
            InlineKeyboardButton(text="🗑️", callback_data=f"delete_button_{button.key}")
        )

    builder.row(InlineKeyboardButton(text="➕ Добавить новую кнопку", callback_data="admin_add_button"))
    builder.row(InlineKeyboardButton(text="⬅️ В админ-панель", callback_data="admin_panel"))
    return builder.as_markup()

def topic_prompt_keyboard(topic_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="📥 Скачать промпт", callback_data=f"download_topic_prompt_{topic_id}")
    builder.button(text="🗑️ Сбросить промпт", callback_data=f"reset_topic_prompt_{topic_id}")
    builder.button(text="⬅️ Назад", callback_data=f"cancel_state_edit_topic_{topic_id}")
    builder.adjust(2, 1)
    return builder.as_markup()


def mailing_history_keyboard(mailings: list, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()

    for mailing in mailings:
        date_utc = mailing.start_time or mailing.created_at
        date = to_msk(date_utc)
        status = "🎂" if is_birthday_mailing(mailing) else get_mailing_status_label(mailing).split(" ", 1)[0]

        text_preview = (mailing.text or "Без текста")[:20]
        builder.button(
            text=f"{status} {date.strftime('%d.%m %H:%M')} | {text_preview}...",
            callback_data=f"mailing_details_{mailing.id}"
        )
    builder.adjust(1)

    if total_pages > 1:
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"mailing_history_page_{page - 1}"))
        nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"mailing_history_page_{page + 1}"))
        builder.row(*nav_buttons)

    builder.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_mailing_menu"))
    return builder.as_markup()

def birthday_template_keyboard(mailing_id: int | None = None, is_enabled: bool = True):
    builder = InlineKeyboardBuilder()
    if mailing_id is None:
        builder.button(text="➕ Создать шаблон", callback_data="mailing_create_birthday_template")
    else:
        builder.button(text="✏️ Обновить шаблон", callback_data="mailing_create_birthday_template")
        builder.button(text="🧪 Отправить себе сейчас", callback_data=f"mailing_send_test_{mailing_id}")
        toggle_text = "⏸ Выключить шаблон" if is_enabled else "▶️ Включить шаблон"
        builder.button(text=toggle_text, callback_data=f"mailing_toggle_enabled_{mailing_id}")
        builder.button(text="🗑 Удалить шаблон", callback_data=f"mailing_delete_birthday_{mailing_id}")
    builder.button(text="⬅️ В меню рассылок", callback_data="admin_mailing_menu")
    builder.adjust(1)
    return builder.as_markup()

def birthday_template_delete_keyboard(mailing_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="🗑 Да, удалить шаблон", callback_data=f"mailing_delete_birthday_confirm_{mailing_id}")
    builder.button(text="⬅️ Назад к шаблону", callback_data="mailing_birthday_template")
    builder.adjust(1)
    return builder.as_markup()

def mailing_details_keyboard(mailing):
    builder = InlineKeyboardBuilder()
    if is_birthday_mailing(mailing):
        return birthday_template_keyboard(mailing.id, mailing.is_enabled)
    builder.button(text="⬅️ Назад к истории", callback_data="mailing_history_page_0")
    builder.adjust(1)
    return builder.as_markup()

def admin_payment_keys_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="ЮKassa Shop ID", callback_data="set_payment_key_yookassa_shop_id")
    builder.button(text="ЮKassa Secret Key", callback_data="set_payment_key_yookassa_secret_key")
    builder.button(text="Robokassa Merchant Login", callback_data="set_payment_key_robokassa_merchant_login")
    builder.button(text="Robokassa Password 1", callback_data="set_payment_key_robokassa_password_1")
    builder.button(text="Robokassa Password 2", callback_data="set_payment_key_robokassa_password_2")
    builder.button(text="Telegram Pay Token", callback_data="set_payment_key_telegram_pay_token")
    builder.button(text="URL: Договор оферты", callback_data="set_payment_key_offer_agreement_url")
    builder.button(text="URL: Политика конфид.", callback_data="set_payment_key_privacy_policy_url")
    builder.button(text="⬅️ Назад", callback_data="admin_payment_settings")
    builder.adjust(1)
    return builder.as_markup()

def admin_select_duration_unit_keyboard(editing: bool = False):
    builder = InlineKeyboardBuilder()
    builder.button(text="Дни", callback_data="set_duration_unit_days")
    builder.button(text="Месяцы", callback_data="set_duration_unit_months")
    if editing:
        builder.row(InlineKeyboardButton(text="⬅️ Отмена", callback_data="cancel_state_admin_plans"))
    return builder.as_markup()

def assign_promo_to_plan_keyboard(promo_id: int, all_plans: list, assigned_plan_ids: set, applies_to_all: bool, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()

    all_plans_status = "✅ Применяется ко всем тарифам" if applies_to_all else "❌ Применяется только к выбранным"
    builder.button(text=all_plans_status, callback_data=f"promo_toggle_all_plans_{promo_id}_{page}")

    if not applies_to_all:
        for plan in all_plans:
            is_assigned = plan.id in assigned_plan_ids
            text = f"✅ {plan.name}" if is_assigned else f"⭕️ {plan.name}"
            action = "remove" if is_assigned else "add"
            callback_data = f"promo_plan_toggle_{action}_{promo_id}_{plan.id}_{page}"
            builder.button(text=text, callback_data=callback_data)
        builder.adjust(1)

        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_assign_promo_{promo_id}_page_{page - 1}"))
        if total_pages > 1:
            nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_assign_promo_{promo_id}_page_{page + 1}"))

        if nav_buttons:
            builder.row(*nav_buttons)

    builder.row(InlineKeyboardButton(text="⬅️ Назад к промокоду", callback_data=f"admin_edit_promo_{promo_id}"))
    return builder.as_markup()


def test_answer_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="1 ⛔️", callback_data="test_ans_1")
    builder.button(text="2 👎", callback_data="test_ans_2")
    builder.button(text="3 😐", callback_data="test_ans_3")
    builder.button(text="4 👍", callback_data="test_ans_4")
    builder.button(text="5 🔥", callback_data="test_ans_5")
    builder.adjust(5)
    return builder.as_markup()


def case_study_confirmation_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="Поехали дальше", callback_data="test_confirm_case")
    builder.adjust(1)
    return builder.as_markup()


def test_prompt_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="📥 Скачать промпт (.txt)", callback_data="download_test_prompt")
    builder.button(text="⬅️ Назад", callback_data="cancel_state_admin_test_menu")
    builder.adjust(1)
    return builder.as_markup()


def gender_selection_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="👨 Мужской", callback_data="gender_male")
    builder.button(text="👩 Женский", callback_data="gender_female")
    builder.adjust(2)
    return builder.as_markup()


def user_settings_keyboard(user):
    builder = InlineKeyboardBuilder()

    builder.button(text="✏️ Изменить имя", callback_data="settings_change_name")

    gender_label = "👨 Мужской" if user.gender == 'male' else ("👩 Женский" if user.gender == 'female' else "❓ Не указан")
    builder.button(text=f"👤 Пол: {gender_label}", callback_data="settings_change_gender")

    age_label = user.age if user.age else "Не указан"
    builder.button(text=f"🎂 Возраст: {age_label}", callback_data="settings_change_age")

    length_label = "📏 Обычный" if getattr(user, 'response_length', 'normal') != 'short' else "📏 Короткий"
    builder.button(text=f"Длина ответов: {length_label}", callback_data="settings_toggle_length")

    builder.button(text="❌ Закрыть", callback_data="settings_close")
    builder.adjust(1)
    return builder.as_markup()


def action_button_keyboard(text: str, payload: str):
    builder = InlineKeyboardBuilder()
    builder.button(text=text, callback_data="action_btn_click")
    return builder.as_markup()


def export_date_filter_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="📅 За последние 7 дней", callback_data="export_date_preset_7")
    builder.button(text="📅 За последние 30 дней", callback_data="export_date_preset_30")
    builder.button(text="📅 За последние 90 дней", callback_data="export_date_preset_90")
    builder.button(text="✏️ Ввести даты вручную", callback_data="export_date_manual")
    builder.button(text="♾️ Все даты (без фильтра)", callback_data="export_date_preset_0")
    builder.adjust(1)
    return builder.as_markup()


def mass_export_options_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="TXT (Обычный)", callback_data="run_export_txt_no")
    builder.button(text="TXT (Анонимно)", callback_data="run_export_txt_yes")
    builder.button(text="JSON (Обычный)", callback_data="run_export_json_no")
    builder.button(text="JSON (Анонимно)", callback_data="run_export_json_yes")
    builder.button(text="⬅️ Назад к выбору", callback_data="admin_export_page_0")
    builder.adjust(2, 2, 1)
    return builder.as_markup()


def single_export_options_keyboard(user_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="TXT (Обычный)", callback_data=f"run_single_txt_no_{user_id}")
    builder.button(text="TXT (Анонимно)", callback_data=f"run_single_txt_yes_{user_id}")
    builder.button(text="JSON (Обычный)", callback_data=f"run_single_json_no_{user_id}")
    builder.button(text="JSON (Анонимно)", callback_data=f"run_single_json_yes_{user_id}")
    builder.button(text="⬅️ Назад в профиль", callback_data=f"view_client_{user_id}")
    builder.adjust(2, 2, 1)
    return builder.as_markup()


def card_selection_keyboard(category: str, card_ids: list):
    builder = InlineKeyboardBuilder()
    for i, card_id in enumerate(card_ids, 1):
        builder.button(text=str(i), callback_data=f"card_select_{category}_{card_id}")
    n = len(card_ids)
    if n <= 3:
        builder.adjust(n)
    elif n == 4:
        builder.adjust(2, 2)
    elif n <= 6:
        builder.adjust(3, 3)
    elif n <= 8:
        builder.adjust(4, 4)
    elif n <= 16:
        builder.adjust(4)
    else:
        builder.adjust(3)
    return builder.as_markup()


def topic_media_manage_keyboard(topic_id: int, media_list: list, page: int = 0, total_pages: int = 1):
    builder = InlineKeyboardBuilder()
    for m in media_list:
        if m.media_type == 'audio':
            icon = "🎵"
        elif m.file_name == '_back':
            icon = "🃏"
        else:
            icon = "🖼️"
        builder.button(text=f"{icon} {m.file_name}", callback_data=f"admin_media_view_{m.id}")

    if total_pages > 1:
        if page > 0:
            builder.button(text="⬅️", callback_data=f"admin_topic_media_{topic_id}_{page - 1}")
        builder.button(text=f"{page + 1}/{total_pages}", callback_data="noop")
        if page < total_pages - 1:
            builder.button(text="➡️", callback_data=f"admin_topic_media_{topic_id}_{page + 1}")

    builder.button(text="➕ Добавить файл", callback_data=f"admin_media_add_{topic_id}")
    builder.button(text="⬅️ Назад к теме", callback_data=f"edit_topic_{topic_id}")

    # Файлы по 1 в ряд, пагинация в одну строку, кнопки действий по 1
    rows = [1] * len(media_list)
    if total_pages > 1:
        nav_buttons = (1 if page > 0 else 0) + 1 + (1 if page < total_pages - 1 else 0)
        rows.append(nav_buttons)
    rows.extend([1, 1])
    builder.adjust(*rows)
    return builder.as_markup()


def media_edit_keyboard(media_id: int, topic_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Имя", callback_data=f"admin_media_editname_{media_id}")
    builder.button(text="📂 Категория", callback_data=f"admin_media_editcat_{media_id}")
    builder.button(text="📝 Описание", callback_data=f"admin_media_editdesc_{media_id}")
    builder.button(text="🔄 Заменить файл", callback_data=f"admin_media_editfile_{media_id}")
    builder.button(text="🗑️ Удалить", callback_data=f"admin_media_delete_{media_id}_{topic_id}")
    builder.button(text="⬅️ Назад к списку", callback_data=f"admin_topic_media_{topic_id}")
    builder.adjust(2, 2, 1, 1)
    return builder.as_markup()


def assign_decks_to_topic_keyboard(topic_id: int, all_decks: list[str], assigned_decks: set[str], page: int, total_pages: int):
    builder = InlineKeyboardBuilder()
    for deck in all_decks:
        is_assigned = deck in assigned_decks
        text = f"✅ {deck}" if is_assigned else f"⭕️ {deck}"
        action = "remove" if is_assigned else "add"
        callback_data = f"deck_topic_{action}_{topic_id}_{deck}_{page}"
        builder.button(text=text, callback_data=callback_data)
    builder.adjust(1)

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"assign_deck_topic_{topic_id}_page_{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"assign_deck_topic_{topic_id}_page_{page + 1}"))

    if nav_buttons:
        builder.row(*nav_buttons)

    builder.row(InlineKeyboardButton(text="⬅️ Назад к теме", callback_data=f"edit_topic_{topic_id}"))
    return builder.as_markup()


# ──────────────── Медиа-коллекции — клавиатуры ────────────────

def admin_collections_list_keyboard(collections: list, page: int = 0, total_pages: int = 1):
    builder = InlineKeyboardBuilder()
    for c in collections:
        builder.button(text=f"📂 {c['name']} ({c['count']})", callback_data=f"admin_coll_view_{c['id']}")
    builder.adjust(1)

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_collections_page_{page - 1}"))
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_collections_page_{page + 1}"))
        builder.row(*nav)

    builder.row(InlineKeyboardButton(text="➕ Создать коллекцию", callback_data="admin_coll_create"))
    builder.row(InlineKeyboardButton(text="⬅️ В админ-панель", callback_data="admin_panel"))
    return builder.as_markup()


def admin_collection_detail_keyboard(coll_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Переименовать", callback_data=f"admin_coll_rename_{coll_id}")
    builder.button(text="📎 Управление файлами", callback_data=f"admin_coll_files_{coll_id}_0")
    builder.button(text="➕ Загрузить файл", callback_data=f"admin_coll_upload_{coll_id}")
    builder.button(text="🗑️ Удалить коллекцию", callback_data=f"admin_coll_delete_{coll_id}")
    builder.button(text="⬅️ К списку коллекций", callback_data="admin_collections_page_0")
    builder.adjust(1)
    return builder.as_markup()


def admin_collection_files_keyboard(coll_id: int, all_media: list, assigned_ids: set, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()
    for m in all_media:
        is_in = m.id in assigned_ids
        icon = "✅" if is_in else "⭕️"
        action = "remove" if is_in else "add"
        label = f"{icon} {m.file_name or m.id} [{m.media_type}]"
        builder.button(text=label, callback_data=f"coll_file_{action}_{coll_id}_{m.id}_{page}")
    builder.adjust(1)

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_coll_files_{coll_id}_{page - 1}"))
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_coll_files_{coll_id}_{page + 1}"))
        builder.row(*nav)

    builder.row(InlineKeyboardButton(text="⬅️ К коллекции", callback_data=f"admin_coll_view_{coll_id}"))
    return builder.as_markup()


def assign_collections_to_topic_keyboard(topic_id: int, all_colls: list, assigned_ids: set, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()
    for c in all_colls:
        is_assigned = c['id'] in assigned_ids
        text = f"✅ {c['name']}" if is_assigned else f"⭕️ {c['name']}"
        action = "remove" if is_assigned else "add"
        builder.button(text=text, callback_data=f"topcoll_{action}_{topic_id}_{c['id']}_{page}")
    builder.adjust(1)

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"assign_coll_topic_{topic_id}_page_{page - 1}"))
    nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=f"assign_coll_topic_{topic_id}_page_{page + 1}"))
    if nav:
        builder.row(*nav)

    builder.row(InlineKeyboardButton(text="⬅️ Назад к теме", callback_data=f"edit_topic_{topic_id}"))
    return builder.as_markup()


# ──────────────── Реферальная программа — клавиатуры ────────────────

def admin_referral_menu_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="⚙️ Настройки", callback_data="admin_referral_settings")
    builder.button(text="📩 Шаблоны приглашений", callback_data="admin_referral_templates")
    builder.button(text="👥 Список рефереров", callback_data="admin_referral_referrers_0")
    builder.button(text="⬅️ Назад", callback_data="admin_subscriptions")
    builder.adjust(1)
    return builder.as_markup()


def admin_referral_settings_keyboard(config):
    builder = InlineKeyboardBuilder()

    enabled_label = "✅ Программа: Вкл" if config.referral_enabled else "❌ Программа: Выкл"
    builder.button(text=enabled_label, callback_data="admin_referral_toggle_enabled")

    builder.button(
        text=f"🎁 Бонус рефереру: {config.referral_bonus_days_referrer} дн.",
        callback_data="admin_referral_set_bonus_referrer"
    )
    builder.button(
        text=f"🎁 Бонус рефералу: {config.referral_bonus_days_referral} дн.",
        callback_data="admin_referral_set_bonus_referral"
    )

    pay_label = "✅ Бонус за оплату: Вкл" if config.referral_pay_bonus_enabled else "❌ Бонус за оплату: Выкл"
    builder.button(text=pay_label, callback_data="admin_referral_toggle_pay_bonus")

    builder.button(
        text=f"💰 Дней за оплату: {config.referral_pay_bonus_days}",
        callback_data="admin_referral_set_pay_days"
    )

    first_only_label = "1️⃣ За 1-ю оплату" if config.referral_pay_bonus_first_only else "♾️ За каждую оплату"
    builder.button(text=first_only_label, callback_data="admin_referral_toggle_pay_first_only")

    builder.button(
        text=f"🔤 Кнопка меню: «{config.referral_btn_name}»",
        callback_data="admin_referral_set_btn_name"
    )
    builder.button(
        text=f"🔤 Кнопка подписки: «{config.referral_sub_btn_name}»",
        callback_data="admin_referral_set_sub_btn_name"
    )

    builder.button(text="⬅️ Назад", callback_data="admin_referral_menu")
    builder.adjust(1)
    return builder.as_markup()


def admin_referral_input_cancel_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ Назад", callback_data="admin_referral_cancel_input")
    return builder.as_markup()


def admin_referral_referrers_keyboard(referrers: list, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()
    for r in referrers:
        name = r['name']
        count = r['count']
        total_amt = r['total']
        uid = r['id']
        builder.button(
            text=f"👤 {name} — {count} реф. | {total_amt:.0f} руб.",
            callback_data=f"admin_referral_referrer_{uid}_0"
        )
    builder.adjust(1)  # one referrer per row

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_referral_referrers_{page - 1}"))
    if total_pages > 1:
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_referral_referrers_{page + 1}"))
    if nav:
        builder.row(*nav)

    builder.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_referral_menu"))
    return builder.as_markup()


def admin_referral_referrer_detail_keyboard(referrer_id: int, page: int, total_pages: int):
    builder = InlineKeyboardBuilder()

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin_referral_referrer_{referrer_id}_{page - 1}"))
    if total_pages > 1:
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=f"admin_referral_referrer_{referrer_id}_{page + 1}"))
    if nav:
        builder.row(*nav)

    builder.row(InlineKeyboardButton(text="⬅️ К списку", callback_data="admin_referral_referrers_0"))
    return builder.as_markup()


def referral_template_share_keyboard(share_url: str):
    """Кнопка «Поделиться» под каждым шаблоном приглашения (открывает диалог контактов)."""
    builder = InlineKeyboardBuilder()
    builder.button(text="📤 Поделиться", url=share_url)
    return builder.as_markup()


def admin_referral_templates_keyboard(templates: list):
    """Список шаблонов приглашений для администратора."""
    builder = InlineKeyboardBuilder()
    for tpl in templates:
        short_text = tpl.text[:40].replace('\n', ' ')
        status = "✅" if tpl.is_enabled else "❌"
        builder.button(
            text=f"{status} {tpl.order_num + 1}. {short_text}…",
            callback_data=f"admin_ref_tpl_{tpl.id}"
        )
    builder.button(text="➕ Добавить шаблон", callback_data="admin_ref_tpl_add")
    builder.button(text="⬅️ Назад", callback_data="admin_referral_menu")
    builder.adjust(1)
    return builder.as_markup()


def admin_referral_template_detail_keyboard(tpl_id: int, is_enabled: bool):
    """Кнопки для управления конкретным шаблоном приглашения."""
    builder = InlineKeyboardBuilder()
    toggle_label = "❌ Отключить" if is_enabled else "✅ Включить"
    builder.button(text="✏️ Редактировать", callback_data=f"admin_ref_tpl_edit_{tpl_id}")
    builder.button(text=toggle_label, callback_data=f"admin_ref_tpl_toggle_{tpl_id}")
    builder.button(text="⬆️ Выше", callback_data=f"admin_ref_tpl_up_{tpl_id}")
    builder.button(text="⬇️ Ниже", callback_data=f"admin_ref_tpl_down_{tpl_id}")
    builder.button(text="🗑 Удалить", callback_data=f"admin_ref_tpl_del_{tpl_id}")
    builder.button(text="⬅️ К шаблонам", callback_data="admin_referral_templates")
    builder.adjust(1)
    return builder.as_markup()


def admin_referral_template_confirm_delete_keyboard(tpl_id: int):
    builder = InlineKeyboardBuilder()
    builder.button(text="🗑 Да, удалить", callback_data=f"admin_ref_tpl_del_confirm_{tpl_id}")
    builder.button(text="⬅️ Отмена", callback_data=f"admin_ref_tpl_{tpl_id}")
    builder.adjust(1)
    return builder.as_markup()


def admin_referral_template_input_cancel_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ Отмена", callback_data="admin_referral_templates")
    return builder.as_markup()
