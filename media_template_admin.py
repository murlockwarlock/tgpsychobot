import asyncio
import copy
import html

from aiogram import F
from aiogram.filters import StateFilter
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select
from sqlalchemy.orm import undefer

from certificate_followup_service import (
    build_followup_telegram_keyboard,
    get_or_create_followup_config,
    parse_buttons_config,
    render_caption,
    render_invitation,
    save_followup_caption,
    save_followup_invitation,
    save_followup_stage,
    toggle_followup_config,
    update_followup_button,
    validate_caption_template,
    validate_invitation_text,
    validate_stage_identifier,
)
from database import MediaLibrary, media_collection_items
from admin_html_preview import truncate_html_preview
from media_image_edit import load_media_image_bytes
from media_template_renderer import FONT_FILES, TemplateRenderError, bind_source, normalized_source, render_template, validate_config
from media_templates import current_issue_date, template_config_identity


class TemplateAdminState(StatesGroup):
    editing = State()
    confirming = State()
    editing_caption = State()
    editing_invite = State()
    editing_stage = State()
    editing_button = State()


BUTTON_CONTRACTS = {
    'home': 'navigation',
    'fields': 'navigation',
    'field': 'navigation',
    'edit': 'navigation',
    'add': 'navigation',
    'back': 'navigation',
    'cancel': 'navigation',
    'toggle': 'mutation',
    'bind': 'mutation',
    'save': 'mutation',
    'preview': 'external action',
    'scenario': 'navigation',
    'scen_toggle': 'mutation',
    'scen_caption': 'navigation',
    'scen_invite': 'navigation',
    'scen_stage': 'navigation',
    'scen_buttons': 'navigation',
    'scen_btn_edit': 'navigation',
    'scen_test': 'external action',
    'scen_cancel': 'navigation',
}
FORM_KEYS = ('key', 'label', 'bbox', 'font_key', 'min_font_size', 'max_font_size', 'color', 'align', 'valign', 'uppercase', 'required', 'max_lines', 'sample_value')


def field_form(key, field):
    values = {'key': key, **field}
    return '\n'.join(f'{name}={",".join(map(str, values[name])) if name == "bbox" else str(values[name]).lower() if isinstance(values[name], bool) else values[name]}' for name in FORM_KEYS)


def parse_field_form(text):
    if len(text) > 4096:
        raise TemplateRenderError('Слишком длинная настройка поля.')
    values = {}
    for line in text.splitlines():
        name, separator, value = line.partition('=')
        name = name.strip()
        if not separator or name not in FORM_KEYS or name in values:
            raise TemplateRenderError('Используйте указанные строки key=value без повторов.')
        values[name] = value.strip()
    if set(values) != set(FORM_KEYS):
        raise TemplateRenderError('Заполните все строки настройки поля.')
    key = values.pop('key')
    try:
        values['bbox'] = [int(v.strip()) for v in values['bbox'].split(',')]
        for name in ('min_font_size', 'max_font_size', 'max_lines'):
            values[name] = int(values[name])
        for name in ('uppercase', 'required'):
            if values[name].lower() not in ('true', 'false'):
                raise ValueError()
            values[name] = values[name].lower() == 'true'
    except ValueError as exc:
        raise TemplateRenderError('Проверьте координаты, размеры и true/false.') from exc
    return key, values


def register_template_admin(router, *, session_factory, require_admin, show_media):
    async def media_for(origin):
        media_id, collection_id, _, _ = origin
        async with session_factory() as session:
            stmt = select(MediaLibrary).options(undefer(MediaLibrary.source_image_bytes)).where(
                MediaLibrary.id == media_id,
                MediaLibrary.media_type.in_(('photo', 'image')),
            )
            if collection_id > 0:
                stmt = stmt.where(
                    select(1).select_from(media_collection_items).where(
                        media_collection_items.c.media_id == MediaLibrary.id,
                        media_collection_items.c.collection_id == collection_id,
                    ).exists()
                )
            return await session.scalar(stmt)

    def markup(origin, buttons):
        rows = []
        suffix = ':'.join(map(str, origin))
        for label, action, index in buttons:
            if action not in BUTTON_CONTRACTS:
                raise ValueError('Missing template navigation contract')
            rows.append([InlineKeyboardButton(text=label, callback_data=f'mt:{action}:{suffix}:{index}')])
        return InlineKeyboardMarkup(inline_keyboard=rows)

    async def screen(message, origin, action='home', index=-1):
        media = await media_for(origin)
        if media is None:
            await message.answer('Файл не найден в этой коллекции.')
            return
        config = media.render_template_config
        fields = config.get('fields', {}) if config else {}
        if action == 'fields':
            text = 'Поля шаблона'
            buttons = [(f'{key} — {field["label"]}', 'field', i) for i, (key, field) in enumerate(fields.items())]
            buttons += [('Добавить поле', 'add', -1), ('⬅️ Назад', 'home', -1)]
        elif action == 'field' and 0 <= index < len(fields):
            key = list(fields)[index]
            text = field_form(key, fields[key])
            buttons = [('Изменить', 'edit', index), ('⬅️ Назад', 'fields', -1)]
        elif action == 'scenario':
            async with session_factory() as session:
                scen_cfg = await get_or_create_followup_config(session, media.id, topic_id=media.topic_id)
                await session.commit()
            status_emoji = '🟢 Включен' if scen_cfg.is_active else '⚪️ Выключен'
            caption_preview = truncate_html_preview(scen_cfg.caption_template, 300)
            invite_preview = truncate_html_preview(scen_cfg.invitation_text, 300)
            buttons_list = parse_buttons_config(scen_cfg.buttons_config_json)
            buttons_lines = "\n".join(
                f"{i+1}. {b['text']} ➔ <code>{html.escape(b['target_step'])}</code>"
                for i, b in enumerate(buttons_list)
            )
            text = (
                f"🎓 <b>Сценарий после выдачи сертификата</b>\n\n"
                f"<b>Файл:</b> {html.escape(media.file_name or '')}\n"
                f"<b>Статус:</b> {status_emoji}\n"
                f"<b>Целевой этап:</b> <code>{html.escape(scen_cfg.target_stage or 'не задан')}</code>\n\n"
                f"✉️ <b>Подпись к сертификату (Сообщение 1):</b>\n{caption_preview}\n\n"
                f"📬 <b>Приглашение на встречу (Сообщение 2):</b>\n{invite_preview}\n\n"
                f"🎛️ <b>Кнопки выбора:</b>\n{buttons_lines}"
            )
            buttons = [
                ('⚪️ Выключить' if scen_cfg.is_active else '🟢 Включить', 'scen_toggle', -1),
                ('✏️ Подпись', 'scen_caption', -1),
                ('✉️ Приглашение', 'scen_invite', -1),
                ('🎛️ Кнопки', 'scen_buttons', -1),
                ('🎯 Целевой этап', 'scen_stage', -1),
                ('🚀 Проверить отправку себе', 'scen_test', -1),
                ('⬅️ Назад', 'home', -1),
            ]
            await message.answer(text, reply_markup=markup(origin, buttons), parse_mode='HTML')
            return
        elif action == 'scen_buttons':
            async with session_factory() as session:
                scen_cfg = await get_or_create_followup_config(session, media.id, topic_id=media.topic_id)
            buttons_list = parse_buttons_config(scen_cfg.buttons_config_json)
            buttons = []
            for i, b in enumerate(buttons_list):
                buttons.append((f"✏️ {b['text']} ➔ {b['target_step']}", 'scen_btn_edit', i))
            buttons.append(('⬅️ Назад', 'scenario', -1))
            text = (
                f"🎛️ <b>Кнопки выбора встречи</b>\n\n"
                f"Выберите кнопку для изменения её текста, целевого этапа и ответа бота:"
            )
            await message.answer(text, reply_markup=markup(origin, buttons), parse_mode='HTML')
            return
        else:
            canvas = config.get('canvas', {}) if config else {}
            dimensions = f'{canvas["width"]}×{canvas["height"]}' if canvas else 'не настроен'
            text = f'Шаблон изображения\n\nСтатус: {"включен" if media.render_template_enabled else "выключен"}\nИсточник: {media.file_name}\nРазмер: {dimensions}\nПоля: {len(fields)}'
            buttons = [
                ('❌ Выключить' if media.render_template_enabled else '✅ Включить', 'toggle', -1),
                ('📝 Поля', 'fields', -1),
                ('📬 После выдачи', 'scenario', -1),
                ('🔍 Тестовый preview', 'preview', -1),
                ('Настроить источник', 'bind', -1),
                ('⬅️ Назад', 'back', -1),
            ]
        await message.answer(text, reply_markup=markup(origin, buttons), parse_mode=None)

    @router.callback_query(F.data.startswith('mt:'), StateFilter('*'))
    async def template_callback(callback, state):
        if not await require_admin(callback, state):
            return
        try:
            _, action, mid, cid, page, list_page, idx = callback.data.split(':')
            if action not in BUTTON_CONTRACTS:
                raise TemplateRenderError('Неизвестное действие шаблона.')
            origin = tuple(map(int, (mid, cid, page, list_page)))
            index = int(idx)
            media = await media_for(origin)
            if media is None:
                raise TemplateRenderError('Файл не найден в этой коллекции.')
            data = await state.get_data()
            if action in ('save', 'cancel') and tuple(data.get('template_origin', ())) != origin:
                raise TemplateRenderError('Форма устарела. Откройте шаблон заново.')
            if action == 'back':
                await state.set_state(None)
                await show_media(callback.bot, callback.message.chat.id, origin[1], origin[0], origin[2], origin[3], callback.message)
            elif action in ('edit', 'add'):
                config = copy.deepcopy(media.render_template_config)
                if not config:
                    content = await load_media_image_bytes(media, telegram_bot=callback.bot)
                    config = bind_source(content)
                fields = config['fields']
                if action == 'edit' and not 0 <= index < len(fields):
                    raise TemplateRenderError('Поле больше не существует.')
                key = list(fields)[index] if action == 'edit' else 'new_field'
                field = fields[key] if action == 'edit' else {'label': 'Новое поле', 'bbox': [0, 0, min(config['canvas']['width'], 200), min(config['canvas']['height'], 60)], 'font_key': 'certificate_regular', 'min_font_size': 12, 'max_font_size': 30, 'color': '#FFFFFF', 'align': 'center', 'valign': 'middle', 'uppercase': False, 'required': True, 'max_lines': 1, 'sample_value': 'Пример'}
                await state.update_data(template_origin=origin, template_config=config, template_identity=template_config_identity(media), template_original_key=key if action == 'edit' else None, template_parent_index=index)
                await state.set_state(TemplateAdminState.editing)
                await callback.message.answer('Отправьте настройки поля, изменив нужные строки. Шрифты: ' + ', '.join(FONT_FILES) + '\n\n' + field_form(key, field), reply_markup=markup(origin, [('Отмена', 'cancel', index)]), parse_mode=None)
            elif action == 'save':
                if await state.get_state() != TemplateAdminState.confirming.state or 'template_pending' not in data:
                    raise TemplateRenderError('Сначала отправьте настройки поля.')
                async with session_factory() as session:
                    current = await session.get(MediaLibrary, media.id, with_for_update=True)
                    if template_config_identity(current) != data['template_identity']:
                        raise TemplateRenderError('Настройки изменились. Откройте форму заново.')
                    current.render_template_config = data['template_pending']
                    await session.commit()
                await state.set_state(None)
                await screen(callback.message, origin, 'field', data['template_saved_index'])
            elif action == 'cancel':
                await state.set_state(None)
                await screen(callback.message, origin, 'field' if data.get('template_original_key') else 'fields', data.get('template_parent_index', -1))
            elif action == 'bind':
                content = await load_media_image_bytes(media, telegram_bot=callback.bot)
                config = bind_source(content, (media.render_template_config or {}).get('fields'))
                validate_config(config, allow_empty=True)
                async with session_factory() as session:
                    current = await session.get(MediaLibrary, media.id)
                    current.render_template_config = config
                    current.render_template_enabled = False
                    await session.commit()
                await screen(callback.message, origin)
            elif action == 'toggle':
                if not media.render_template_enabled:
                    validate_config(media.render_template_config)
                    content = await load_media_image_bytes(media, telegram_bot=callback.bot)
                    await asyncio.to_thread(render_template, content, media.render_template_config, {key: field.get('sample_value', field['label']) for key, field in media.render_template_config['fields'].items()})
                async with session_factory() as session:
                    current = await session.get(MediaLibrary, media.id)
                    if template_config_identity(current) != template_config_identity(media):
                        raise TemplateRenderError('Настройки изменились. Повторите действие.')
                    current.render_template_enabled = not media.render_template_enabled
                    await session.commit()
                await screen(callback.message, origin)
            elif action == 'preview':
                config = validate_config(media.render_template_config)
                content = await load_media_image_bytes(media, telegram_bot=callback.bot)
                result = await asyncio.to_thread(render_template, content, config, {key: field.get('sample_value', field['label']) for key, field in config['fields'].items()})
                await callback.message.answer_document(BufferedInputFile(result.image_bytes, filename='template-preview.png'), caption='Тестовый preview', reply_markup=markup(origin, [('⬅️ Назад', 'home', -1)]))
            elif action == 'scenario':
                await state.set_state(None)
                await screen(callback.message, origin, 'scenario')
            elif action == 'scen_toggle':
                async with session_factory() as session:
                    new_active = await toggle_followup_config(session, media.id)
                await callback.answer("Сценарий включен" if new_active else "Сценарий выключен")
                await screen(callback.message, origin, 'scenario')
            elif action == 'scen_caption':
                await state.update_data(template_origin=origin)
                await state.set_state(TemplateAdminState.editing_caption)
                async with session_factory() as session:
                    scen_cfg = await get_or_create_followup_config(session, media.id, topic_id=media.topic_id)
                msg_text = (
                    "✏️ <b>Подпись к сертификату (Сообщение 1)</b>\n\n"
                    "Введите новый текст подписи. Доступные переменные:\n"
                    "• <code>{name}</code> — имя\n"
                    "• <code>{fio}</code> — полное ФИО\n"
                    "• <code>{date}</code> — дата выдачи\n\n"
                    f"Текущий текст:\n<blockquote>{html.escape(scen_cfg.caption_template)}</blockquote>"
                )
                await callback.message.answer(msg_text, reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')
            elif action == 'scen_invite':
                await state.update_data(template_origin=origin)
                await state.set_state(TemplateAdminState.editing_invite)
                async with session_factory() as session:
                    scen_cfg = await get_or_create_followup_config(session, media.id, topic_id=media.topic_id)
                msg_text = (
                    "✉️ <b>Приглашение на встречу (Сообщение 2)</b>\n\n"
                    "Введите новый текст приглашения. Доступная переменная:\n"
                    "• <code>{name}</code> — имя\n\n"
                    f"Текущий текст:\n<blockquote>{html.escape(scen_cfg.invitation_text)}</blockquote>"
                )
                await callback.message.answer(msg_text, reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')
            elif action == 'scen_stage':
                await state.update_data(template_origin=origin)
                await state.set_state(TemplateAdminState.editing_stage)
                async with session_factory() as session:
                    scen_cfg = await get_or_create_followup_config(session, media.id, topic_id=media.topic_id)
                msg_text = (
                    "🎯 <b>Целевой этап диалога</b>\n\n"
                    "Введите название этапа, в который переходит пользователь сразу после отправки приглашения:\n\n"
                    f"Текущий этап: <code>{html.escape(scen_cfg.target_stage or 'не задан')}</code>"
                )
                await callback.message.answer(msg_text, reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')
            elif action == 'scen_buttons':
                await state.set_state(None)
                await screen(callback.message, origin, 'scen_buttons')
            elif action == 'scen_btn_edit':
                await state.update_data(template_origin=origin, button_index=index)
                await state.set_state(TemplateAdminState.editing_button)
                async with session_factory() as session:
                    scen_cfg = await get_or_create_followup_config(session, media.id, topic_id=media.topic_id)
                buttons_list = parse_buttons_config(scen_cfg.buttons_config_json)
                btn = buttons_list[index] if 0 <= index < len(buttons_list) else buttons_list[0]
                msg_text = (
                    f"✏️ <b>Настройка кнопки #{index + 1}</b>\n\n"
                    f"Текущие настройки:\n"
                    f"• Текст: <code>{html.escape(btn['text'])}</code>\n"
                    f"• Целевой этап: <code>{html.escape(btn['target_step'])}</code>\n"
                    f"• Ответ бота: <code>{html.escape(btn.get('response_text', ''))}</code>\n\n"
                    f"Отправьте новые настройки через вертикальную черту <code>|</code>:\n"
                    f"<code>Текст кнопки | ЦЕЛЕВОЙ_ЭТАП | Текст ответа бота</code>"
                )
                await callback.message.answer(msg_text, reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')
            elif action == 'scen_cancel':
                await state.set_state(None)
                await screen(callback.message, origin, 'scenario')
            elif action == 'scen_test':
                async with session_factory() as session:
                    scen_cfg = await get_or_create_followup_config(session, media.id, topic_id=media.topic_id)
                config = validate_config(media.render_template_config)
                content = await load_media_image_bytes(media, telegram_bot=callback.bot)
                sample_values = {key: field.get('sample_value', field['label']) for key, field in config['fields'].items()}
                result = await asyncio.to_thread(render_template, content, config, sample_values)

                # Message 1: Certificate
                test_context = {
                    'name': callback.from_user.first_name or 'Пользователь',
                    'fio': sample_values.get('recipient') or callback.from_user.full_name or 'Пользователь',
                    'date': current_issue_date(),
                }
                caption_text = render_caption(scen_cfg.caption_template, test_context)
                caption_text = truncate_html_preview(caption_text, 1024)
                await callback.bot.send_document(
                    chat_id=callback.from_user.id,
                    document=BufferedInputFile(result.image_bytes, filename='test-certificate.png'),
                    caption=caption_text,
                    parse_mode='HTML',
                )

                # Message 2: Invitation
                invite_text = render_invitation(scen_cfg.invitation_text, test_context)
                test_keyboard = build_followup_telegram_keyboard(scen_cfg)
                await callback.bot.send_message(
                    chat_id=callback.from_user.id,
                    text=invite_text,
                    reply_markup=test_keyboard,
                    parse_mode='HTML',
                )
                await callback.answer("Тестовые сообщения отправлены в ваш чат!")
                await screen(callback.message, origin, 'scenario')
            else:
                await state.set_state(None)
                await screen(callback.message, origin, action, index)
            await callback.answer()
        except (ValueError, TypeError, KeyError, TemplateRenderError) as exc:
            await callback.answer(str(exc) if isinstance(exc, TemplateRenderError) else 'Некорректная настройка шаблона.', show_alert=True)

    @router.message(TemplateAdminState.editing, F.text)
    async def template_field_input(message, state):
        if not await require_admin(message, state):
            return
        data = await state.get_data()
        origin = tuple(data['template_origin'])
        try:
            key, field = parse_field_form(message.text)
            config = copy.deepcopy(data['template_config'])
            old = data['template_original_key']
            if key in config['fields'] and key != old:
                raise TemplateRenderError('Поле с таким key уже существует.')
            if old and key != old:
                del config['fields'][old]
            config['fields'][key] = field
            validate_config(config)
            await state.update_data(template_pending=config, template_saved_index=list(config['fields']).index(key))
            await state.set_state(TemplateAdminState.confirming)
            await message.answer('Проверьте настройки:\n\n' + field_form(key, field), reply_markup=markup(origin, [('Сохранить', 'save', -1), ('Отмена', 'cancel', data['template_parent_index'])]), parse_mode=None)
        except (TemplateRenderError, ValueError, TypeError) as exc:
            await message.answer(str(exc), reply_markup=markup(origin, [('Отмена', 'cancel', data['template_parent_index'])]), parse_mode=None)

    @router.message(TemplateAdminState.editing_caption, F.text)
    async def template_scen_caption_input(message, state):
        if not await require_admin(message, state):
            return
        data = await state.get_data()
        origin = tuple(data['template_origin'])
        media = await media_for(origin)
        if media is None:
            await state.set_state(None)
            await message.answer("Файл не найден.")
            return
        try:
            validate_caption_template(message.text)
            async with session_factory() as session:
                await save_followup_caption(session, media.id, message.text)
            await state.set_state(None)
            await message.answer("✅ Подпись успешно сохранена.")
            await screen(message, origin, 'scenario')
        except ValueError as exc:
            await message.answer(f"❌ Ошибка валидации: {html.escape(str(exc))}\n\nПопробуйте ещё раз или нажмите «Отмена».", reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')

    @router.message(TemplateAdminState.editing_invite, F.text)
    async def template_scen_invite_input(message, state):
        if not await require_admin(message, state):
            return
        data = await state.get_data()
        origin = tuple(data['template_origin'])
        media = await media_for(origin)
        if media is None:
            await state.set_state(None)
            await message.answer("Файл не найден.")
            return
        try:
            validate_invitation_text(message.text)
            async with session_factory() as session:
                await save_followup_invitation(session, media.id, message.text)
            await state.set_state(None)
            await message.answer("✅ Приглашение успешно сохранено.")
            await screen(message, origin, 'scenario')
        except ValueError as exc:
            await message.answer(f"❌ Ошибка валидации: {html.escape(str(exc))}\n\nПопробуйте ещё раз или нажмите «Отмена».", reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')

    @router.message(TemplateAdminState.editing_stage, F.text)
    async def template_scen_stage_input(message, state):
        if not await require_admin(message, state):
            return
        data = await state.get_data()
        origin = tuple(data['template_origin'])
        media = await media_for(origin)
        if media is None:
            await state.set_state(None)
            await message.answer("Файл не найден.")
            return
        try:
            validated = validate_stage_identifier(message.text)
            async with session_factory() as session:
                await save_followup_stage(session, media.id, validated)
            await state.set_state(None)
            await message.answer("✅ Целевой этап успешно сохранен.")
            await screen(message, origin, 'scenario')
        except ValueError as exc:
            await message.answer(f"❌ Ошибка: {html.escape(str(exc))}\n\nПопробуйте ещё раз или нажмите «Отмена».", reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')

    @router.message(TemplateAdminState.editing_button, F.text)
    async def template_scen_button_input(message, state):
        if not await require_admin(message, state):
            return
        data = await state.get_data()
        origin = tuple(data['template_origin'])
        btn_idx = int(data.get('button_index', 0))
        media = await media_for(origin)
        if media is None:
            await state.set_state(None)
            await message.answer("Файл не найден.")
            return
        parts = [p.strip() for p in message.text.split('|')]
        if len(parts) < 2:
            await message.answer("❌ Неверный формат. Отправьте:\n<code>Текст кнопки | ЦЕЛЕВОЙ_ЭТАП | Текст ответа бота</code>", reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')
            return
        btn_text = parts[0]
        btn_stage = parts[1]
        btn_response = parts[2] if len(parts) > 2 else ""
        try:
            async with session_factory() as session:
                await update_followup_button(session, media.id, btn_idx, btn_text, btn_stage, btn_response)
            await state.set_state(None)
            await message.answer("✅ Кнопка успешно обновлена.")
            await screen(message, origin, 'scen_buttons')
        except ValueError as exc:
            await message.answer(f"❌ Ошибка: {html.escape(str(exc))}\n\nПопробуйте ещё раз или нажмите «Отмена».", reply_markup=markup(origin, [('⬅️ Отмена', 'scen_cancel', -1)]), parse_mode='HTML')
