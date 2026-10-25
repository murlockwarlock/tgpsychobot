import asyncio
import copy
import html

from aiogram import F
from aiogram.filters import StateFilter
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select
from sqlalchemy.orm import undefer

from database import MediaLibrary, media_collection_items
from media_image_edit import load_media_image_bytes
from media_template_renderer import FONT_FILES, TemplateRenderError, bind_source, normalized_source, render_template, validate_config
from media_templates import template_config_identity


class TemplateAdminState(StatesGroup):
    editing = State()
    confirming = State()


BUTTON_CONTRACTS = {'home': 'navigation', 'fields': 'navigation', 'field': 'navigation', 'edit': 'navigation', 'add': 'navigation', 'back': 'navigation', 'cancel': 'navigation', 'toggle': 'mutation', 'bind': 'mutation', 'save': 'mutation', 'preview': 'external action'}
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
            return await session.scalar(select(MediaLibrary).options(undefer(MediaLibrary.source_image_bytes)).where(MediaLibrary.id == media_id, MediaLibrary.media_type.in_(('photo', 'image')), select(1).select_from(media_collection_items).where(media_collection_items.c.media_id == MediaLibrary.id, media_collection_items.c.collection_id == collection_id).exists()))

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
        else:
            canvas = config.get('canvas', {}) if config else {}
            dimensions = f'{canvas["width"]}×{canvas["height"]}' if canvas else 'не настроен'
            text = f'Шаблон изображения\n\nСтатус: {"включен" if media.render_template_enabled else "выключен"}\nИсточник: {media.file_name}\nРазмер: {dimensions}\nПоля: {len(fields)}'
            buttons = [('❌ Выключить' if media.render_template_enabled else '✅ Включить', 'toggle', -1), ('📝 Поля', 'fields', -1), ('🔍 Тестовый preview', 'preview', -1), ('Настроить источник', 'bind', -1), ('⬅️ Назад', 'back', -1)]
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
