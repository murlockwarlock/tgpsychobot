import asyncio
import copy
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

os.environ.setdefault('BOT_TOKEN', '123456:TEST_BOT_TOKEN')
os.environ.setdefault('DATABASE_URL', 'sqlite+aiosqlite:///:memory:')

import pytest
from PIL import Image, ImageChops
from aiogram import Bot, Dispatcher
from aiogram.methods import SendDocument, SendMessage, SendPhoto, AnswerCallbackQuery, GetFile
from aiogram.types import CallbackQuery, Chat, Message, Update, User as TelegramUser
from sqlalchemy import select, text

import database
import handlers
import ai_integration
import media_templates
import max_messenger_bot.ai as max_ai
import max_messenger_bot.app as max_app
from database import AIConfig, MediaLibrary, Topic, User, Message as DBMessage, media_collection_items
from media_template_renderer import TemplateRenderError, TemplateRenderInvariantError, bind_source, normalized_source, render_template, validate_config, validate_pixels, validate_fonts
from media_templates import extract_template_directive, format_template_sources, render_media_template, validate_rendered_template
from media_template_admin import BUTTON_CONTRACTS, TemplateAdminState, field_form
from tests.test_media_image_edit import async_engine, session_factory, clean_runtime, setup_tg_user, media_setup, TelegramTransport, MaxTransport
from tests.test_hide_empty_media_prompt import LEGACY_SERVICE_PROMPT_TEMPLATE


VALUES = {'recipient': 'ИВАН СОЛДАТОВ', 'issue_date': '24.10.2026'}


def source_png(size=(600, 900), mode='RGBA'):
    data = io.BytesIO()
    image = Image.new(mode, size, (81, 31, 65, 180) if mode == 'RGBA' else (81, 31, 65))
    image.save(data, format='PNG')
    return data.getvalue()


def template_config(source):
    width, height = normalized_source(source).size
    field = {'label': 'Получатель', 'bbox': [20, 20, width - 20, 95], 'font_key': 'certificate_regular', 'min_font_size': 12, 'max_font_size': 36, 'align': 'center', 'valign': 'middle', 'color': '#FFFFFF', 'uppercase': True, 'max_lines': 1, 'required': True, 'sample_value': VALUES['recipient']}
    date = {**field, 'label': 'Дата выдачи', 'bbox': [20, 120, width - 20, 180], 'uppercase': False, 'sample_value': VALUES['issue_date']}
    return bind_source(source, {'recipient': field, 'issue_date': date})


def assert_pixels(source, result, config):
    before, after = normalized_source(source), normalized_source(result)
    assert before.size == after.size
    difference = ImageChops.difference(before, after)
    for field in config['fields'].values():
        assert any(band.getbbox() for band in difference.crop(field['bbox']).split())
        difference.paste((0, 0, 0, 0), field['bbox'])
    assert all(not band.getbbox() for band in difference.split())


@pytest.mark.parametrize('mode', ['RGB', 'RGBA'])
@pytest.mark.parametrize('align', ['left', 'center', 'right'])
@pytest.mark.parametrize('valign', ['top', 'middle', 'bottom'])
def test_exact_pixels_and_determinism(mode, align, valign):
    source = source_png(mode=mode)
    config = template_config(source)
    for field in config['fields'].values():
        field.update(align=align, valign=valign)
    rendered = render_template(source, config, VALUES)
    assert rendered.image_bytes.startswith(b'\x89PNG')
    assert render_template(source, config, VALUES).image_bytes == rendered.image_bytes
    assert_pixels(source, rendered.image_bytes, config)


def test_invariant_detects_rgb_and_alpha_corruption():
    before = normalized_source(source_png())
    for pixel in [(1, 2, 3, 180), (81, 31, 65, 179)]:
        after = before.copy()
        after.putpixel((0, 0), pixel)
        with pytest.raises(TemplateRenderInvariantError):
            validate_pixels(before, after, [[20, 20, 580, 95]])


@pytest.mark.parametrize('values', [{}, {'recipient': 'ИВАН'}, {**VALUES, 'extra': 'x'}, {**VALUES, 'recipient': 'X' * 257}, {**VALUES, 'recipient': 'a\x00b'}, {**VALUES, 'recipient': ' '}])
def test_invalid_values(values):
    source = source_png()
    with pytest.raises(TemplateRenderError):
        render_template(source, template_config(source), values)


@pytest.mark.parametrize('field,value', [('bbox', [-1, 0, 90, 90]), ('bbox', [0, 0, 601, 900]), ('font_key', '/etc/font.ttf'), ('font_key', '../../font'), ('color', 'url(x)'), ('min_font_size', 0), ('max_lines', 0)])
def test_invalid_layout(field, value):
    config = template_config(source_png())
    config['fields']['recipient'][field] = value
    with pytest.raises(TemplateRenderError):
        validate_config(config)


@pytest.mark.parametrize('value', [None, [], {'version': 1}, {'unexpected': 'x'}])
def test_malformed_config_is_controlled(value):
    with pytest.raises(TemplateRenderError):
        validate_config(value)


def test_missing_glyph_is_rejected():
    source = source_png()
    with pytest.raises(TemplateRenderError, match='символы'):
        render_template(source, template_config(source), {**VALUES, 'recipient': '漢'})


@pytest.mark.parametrize('change', ['sha', 'dimensions', 'too_long', 'invisible'])
def test_source_and_fitting_fail_closed(change):
    source = source_png()
    config = template_config(source)
    values = dict(VALUES)
    if change == 'sha':
        config['source_sha256'] = '0' * 64
    elif change == 'dimensions':
        config['canvas']['height'] -= 1
    elif change == 'too_long':
        config['fields']['recipient']['bbox'] = [20, 20, 30, 30]
    else:
        config['fields']['recipient']['color'] = '#FFFFFF00'
    with pytest.raises(TemplateRenderError):
        render_template(source, config, values)


def test_multiline_and_exif():
    image = Image.new('RGB', (900, 600), 'black')
    exif = Image.Exif(); exif[274] = 6
    data = io.BytesIO(); image.save(data, format='JPEG', exif=exif)
    source = data.getvalue()
    config = template_config(source)
    assert config['canvas'] == {'width': 600, 'height': 900}
    config['fields']['recipient'].update(bbox=[10, 10, 240, 200], min_font_size=30, max_font_size=30, max_lines=4)
    assert_pixels(source, render_template(source, config, VALUES).image_bytes, config)
    validate_fonts()


@pytest.mark.parametrize('reference', ['panda_certificate', '3', ' сертификат '])
def test_parser_structured_escaping(reference):
    values = {'recipient': 'ИВАН "[СОЛДАТОВ]" | test', 'issue_date': '24.10.2026'}
    directive, clean = extract_template_directive('Готово:\n[RENDER_IMG_MEDIA: ' + reference + ' | ' + json.dumps(values, ensure_ascii=False) + ']')
    assert directive.values == values and clean == 'Готово:'


@pytest.mark.parametrize('payload', ['{}', '[]', '{"recipient":3}', '{"recipient":"a","recipient":"b"}', '{"recipient":"a"', '{"recipient":"a\\u0000b"}', '{"recipient":"' + 'x' * 257 + '"}', '{"recipient": {}}'])
def test_parser_rejects_json(payload):
    assert extract_template_directive('[RENDER_IMG_MEDIA: panda_certificate | ' + payload + ']')[0] is None


@pytest.mark.parametrize('reference', ['', '../panda', '/etc/file', 'https://x', 'a\\b', 'a' * 256])
def test_parser_rejects_references(reference):
    assert extract_template_directive('[RENDER_IMG_MEDIA: ' + reference + ' | {"recipient":"a"}]')[0] is None


@pytest.mark.parametrize('other', ['[RENDER_IMG_MEDIA: panda_certificate | {"recipient":"b"}]', '[EDIT_IMG_MEDIA: 3 | blue]', 'GEN_IMG: [blue]', '[SHOW_IMG: x]', '[SEND_AUDIO: x]', 'RUN_TEST'])
def test_mixed_executable_rejected(other):
    assert extract_template_directive('[RENDER_IMG_MEDIA: panda_certificate | {"recipient":"a"}]\n' + other)[0] is None


async def seed_template(sessions, *, source=None):
    source = source or source_png()
    config = template_config(source)
    async with sessions() as session:
        session.add(MediaLibrary(id=3, file_id='telegram-source', file_name='panda_certificate', media_type='photo', render_template_enabled=True, render_template_config=config))
        await session.flush()
        await session.execute(media_collection_items.insert().values(collection_id=1, media_id=3))
        maximum = await session.get(MediaLibrary, 40)
        maximum.render_template_enabled = True
        maximum.render_template_config = config
        ai = await session.get(AIConfig, 1)
        ai.allow_image_edit = False
        ai.service_prompt_block = LEGACY_SERVICE_PROMPT_TEMPLATE
        (await session.get(Topic, 1)).system_prompt = 'Когда наступит этап выдачи сертификата, используй шаблон panda_certificate. Впиши имя ребёнка и дату выдачи. После этого отправь готовый сертификат.'
        await session.commit()
    return source, config


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['telegram', 'max'])
@pytest.mark.parametrize('case', ['ok', 'panda', 'disabled', 'missing_config', 'wrong_scope', 'duplicate', 'missing', 'nonphoto', 'sha', 'dimensions', 'unknown', 'required', 'long', 'control', 'json', 'multiple', 'mixed_edit', 'mixed_gen', 'stale_before', 'stale_after', 'source_replaced_after', 'stale_delivery', 'config_delivery'])
async def test_real_runtime(session_factory, media_setup, monkeypatch, surface, case):
    if case == 'panda' and not os.getenv('PANDA_TEMPLATE_SOURCE'):
        pytest.skip('Private read-only certificate fixture not provided')
    source, config = await seed_template(session_factory, source=Path(os.environ['PANDA_TEMPLATE_SOURCE']).read_bytes() if case == 'panda' else None)
    if case == 'panda':
        config['fields']['recipient'].update(bbox=[70, 340, 616, 409], max_font_size=38, font_key='certificate_bold')
        config['fields']['issue_date'].update(bbox=[234, 914, 386, 944], min_font_size=12, max_font_size=22)
    delivers = case in ('ok', 'panda')
    reference = 'panda_certificate'
    values = dict(VALUES)
    async with session_factory() as session:
        media = await session.get(MediaLibrary, 3 if surface == 'telegram' else 40)
        if surface == 'max':
            (await session.get(MediaLibrary, 3)).file_name = 'telegram-template'
            media.file_name = 'panda_certificate'
        if case == 'panda': media.render_template_config = config
        if case == 'disabled': media.render_template_enabled = False
        if case == 'missing_config': media.render_template_config = None
        if case == 'sha': media.render_template_config = {**config, 'source_sha256': '0' * 64}
        if case == 'dimensions': media.render_template_config = {**config, 'canvas': {'width': 600, 'height': 899}}
        if case == 'nonphoto': media.media_type = 'document'
        if case == 'duplicate':
            session.add(MediaLibrary(id=99, file_id='other', file_name=reference, media_type='photo', render_template_enabled=True, render_template_config=config))
            await session.flush()
            await session.execute(media_collection_items.insert().values(collection_id=1, media_id=99))
        await session.commit()
    if case == 'wrong_scope': reference = 'foreign'
    if case == 'missing': reference = 'absent'
    if case == 'unknown': values['unknown'] = 'x'
    if case == 'required': values.pop('issue_date')
    if case == 'long': values['recipient'] = 'x' * 257
    if case == 'control': values['recipient'] = 'x\x01'
    response = 'Готово, вот сертификат:\n[RENDER_IMG_MEDIA: ' + reference + ' | ' + json.dumps(values, ensure_ascii=False) + ']'
    if case == 'json': response = '[RENDER_IMG_MEDIA: ' + reference + ' | {broken}]'
    if case == 'multiple': response += response
    if case == 'mixed_edit': response += '\n[EDIT_IMG_MEDIA: 10 | blue]'
    if case == 'mixed_gen': response += '\nGEN_IMG: [blue]'
    forbidden = AsyncMock(side_effect=AssertionError('Image provider must never be called'))
    monkeypatch.setattr(ai_integration, 'edit_image', forbidden)
    monkeypatch.setattr(ai_integration, 'generate_image', forbidden)
    from max_messenger_bot.services import common
    monkeypatch.setattr(common, 'edit_image', forbidden)
    monkeypatch.setattr(common, 'generate_image', forbidden)
    async def change_context():
        async with session_factory() as session:
            (await session.get(User, media_setup if surface == 'telegram' else 100000000007)).current_topic_id = 2
            await session.commit()
    async def change_delivery():
        if case == 'stale_delivery':
            await change_context()
        else:
            async with session_factory() as session:
                (await session.get(MediaLibrary, 3 if surface == 'telegram' else 40)).render_template_config = {**config, 'source_sha256': '0' * 64}
                await session.commit()
    if case in ('stale_after', 'source_replaced_after'):
        original_to_thread = asyncio.to_thread
        async def render_then_mutate(function, *args):
            if function is not render_template:
                return await original_to_thread(function, *args)
            result = function(*args)
            if case == 'stale_after':
                await change_context()
            else:
                async with session_factory() as session:
                    (await session.get(MediaLibrary, 3 if surface == 'telegram' else 40)).file_id = 'replaced'
                    await session.commit()
            return result
        monkeypatch.setattr(media_templates.asyncio, 'to_thread', render_then_mutate)
    async def respond(*args, **kwargs):
        if case == 'stale_before': await change_context()
        return response
    if surface == 'telegram':
        transport = TelegramTransport()
        async def stream(*args, **kwargs): yield source
        monkeypatch.setattr(transport, 'stream_content', stream)
        model = AsyncMock(side_effect=respond)
        monkeypatch.setattr(ai_integration, '_call_gemini_api', model)
        if case in ('stale_delivery', 'config_delivery'):
            original_presentation = handlers._prepare_telegram_response_buttons
            async def prepare_then_change(*args, **kwargs):
                result = await original_presentation(*args, **kwargs)
                await change_delivery()
                return result
            monkeypatch.setattr(handlers, '_prepare_telegram_response_buttons', prepare_then_change)
        bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
        handlers.router._parent_router = None
        dispatcher = Dispatcher(); dispatcher.include_router(handlers.router)
        try:
            await dispatcher.feed_update(bot, Update(update_id=5001, message=Message(message_id=5001, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Denis'), text='ИВАН СОЛДАТОВ, 24.10.2026')))
            if media_setup in handlers.user_processing_tasks: await handlers.user_processing_tasks[media_setup]
            documents = [call for call in transport.calls if isinstance(call, SendDocument)]
            assert len(documents) == int(delivers)
            assert not any(isinstance(call, SendPhoto) for call in transport.calls)
            if case in ('disabled', 'missing_config', 'wrong_scope', 'duplicate', 'missing', 'nonphoto', 'json', 'multiple', 'mixed_edit', 'mixed_gen'):
                assert not any(isinstance(call, GetFile) for call in transport.calls)
            if documents:
                document = SendDocument.model_validate(documents[0].model_dump())
                assert document.document.filename == 'certificate.png'
                assert_pixels(source, document.document.data, config)
                assert document.caption == 'Готово, вот сертификат:'
                assert 'RENDER_IMG_MEDIA' in '\n'.join(model.await_args.kwargs['request_layout'].shared_instructions)
            assert await dispatcher.fsm.get_context(bot=bot, chat_id=media_setup, user_id=media_setup).get_state() is None
        finally:
            dispatcher.sub_routers.clear(); handlers.router._parent_router = None; await bot.session.close()
        user_id = media_setup
    else:
        client = MaxTransport()
        if case in ('stale_delivery', 'config_delivery'):
            original_upload = client.upload_file
            async def upload_then_change(*args, **kwargs):
                result = await original_upload(*args, **kwargs)
                await change_delivery()
                return result
            monkeypatch.setattr(client, 'upload_file', upload_then_change)
        monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))
        model = AsyncMock(side_effect=respond)
        monkeypatch.setattr(max_ai, '_dispatch_provider', model)
        application = max_app.MaxBotApplication(client)
        await application.handle_update({'update_type': 'message_created', 'update_id': 'template', 'message': {'sender': {'user_id': 7, 'name': 'Denis'}, 'recipient': {'chat_id': 100}, 'body': {'mid': 'template-input', 'text': 'ИВАН СОЛДАТОВ, 24.10.2026'}}})
        if application.background_tasks: await asyncio.gather(*list(application.background_tasks))
        attachments = [item for _, _, kwargs in client.wire for item in kwargs.get('json_data', {}).get('attachments', [])]
        assert attachments == ([{'type': 'file', 'payload': {'token': 'edited-token'}}] if delivers else [])
        assert len(client.uploads) == int(delivers or case in ('stale_delivery', 'config_delivery'))
        if case in ('disabled', 'missing_config', 'wrong_scope', 'duplicate', 'missing', 'nonphoto', 'json', 'multiple', 'mixed_edit', 'mixed_gen'):
            assert client.sources == []
        if delivers:
            assert client.uploads[0][0] == 'file'
            assert_pixels(source, client.uploads[0][1], config)
            instructions = '\n'.join(model.await_args.args[1].shared_instructions)
            assert 'RENDER_IMG_MEDIA' in instructions
            assert all(command not in instructions for command in ('SHOW_IMG', 'RANDOM_IMG', 'SEND_AUDIO'))
        user_id = 100000000007
    forbidden.assert_not_awaited()
    async with session_factory() as session:
        user = await session.get(User, user_id)
        assert (user.current_dialogue_id, user.current_topic_id) == (1, 2 if case.startswith('stale_') else 1)
        history = (await session.scalars(select(DBMessage).where(DBMessage.user_id == user_id, DBMessage.role == 'assistant'))).all()
        assert bool(history) is (not case.startswith('stale_')), [(row.content, row.dialogue_id, row.topic_id) for row in history]
        assert all(row.topic_id == 1 and row.dialogue_id == 1 and 'RENDER_IMG_MEDIA' not in row.content for row in history)
        if delivers: assert history[-1].content == 'Готово, вот сертификат:'


def test_actual_panda_fixture():
    path = os.getenv('PANDA_TEMPLATE_SOURCE')
    if not path: pytest.skip('Private read-only certificate fixture not provided')
    source = Path(path).read_bytes()
    config = template_config(source)
    config['fields']['recipient'].update(bbox=[70, 340, 616, 409], max_font_size=38, font_key='certificate_bold')
    config['fields']['issue_date'].update(bbox=[234, 914, 386, 944], min_font_size=12, max_font_size=22)
    rendered = render_template(source, config, VALUES)
    assert (rendered.width, rendered.height) == (904, 1280)
    assert_pixels(source, rendered.image_bytes, config)
    output = os.getenv('PANDA_TEMPLATE_PREVIEW')
    if output:
        Path(output).write_bytes(rendered.image_bytes)
        Path(output + '.config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['before', 'after_render'])
async def test_stale_context_suppresses_render_result(session_factory, media_setup, monkeypatch, stage):
    source, config = await seed_template(session_factory)
    calls = []
    async def load(*args, **kwargs):
        calls.append('source')
        return source
    monkeypatch.setattr(media_templates, 'load_media_image_bytes', load)
    if stage == 'before':
        async with session_factory() as session:
            (await session.get(User, media_setup)).current_topic_id = 2
            await session.commit()
    else:
        async def render_then_change(function, *args):
            result = function(*args)
            async with session_factory() as session:
                (await session.get(User, media_setup)).current_topic_id = 2
                await session.commit()
            return result
        monkeypatch.setattr(media_templates.asyncio, 'to_thread', render_then_change)
    directive = media_templates.TemplateDirective('panda_certificate', VALUES)
    assert await render_media_template(directive, session_factory=session_factory, user_id=media_setup, dialogue_id=1, topic_id=1) is None
    assert calls == ([] if stage == 'before' else ['source'])


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['source', 'config', 'disabled', 'collection'])
async def test_revalidate_before_delivery(session_factory, media_setup, monkeypatch, change):
    source, config = await seed_template(session_factory)
    monkeypatch.setattr(media_templates, 'load_media_image_bytes', AsyncMock(return_value=source))
    result = await render_media_template(media_templates.TemplateDirective('panda_certificate', VALUES), session_factory=session_factory, user_id=media_setup, dialogue_id=1, topic_id=1)
    async with session_factory() as session:
        media = await session.get(MediaLibrary, 3)
        if change == 'source': media.file_id = 'replaced'
        if change == 'config': media.render_template_config = {**config, 'source_sha256': '0' * 64}
        if change == 'disabled': media.render_template_enabled = False
        if change == 'collection': await session.execute(media_collection_items.delete().where(media_collection_items.c.media_id == 3))
        await session.commit()
    with pytest.raises(ValueError):
        await validate_rendered_template(session_factory, result, 1)


@pytest.mark.asyncio
async def test_main_scope_template_is_available(session_factory, media_setup, monkeypatch):
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        (await session.get(User, media_setup)).current_topic_id = None
        await session.commit()
    monkeypatch.setattr(media_templates, 'load_media_image_bytes', AsyncMock(return_value=source))
    for reference in ('panda_certificate', 3):
        result = await render_media_template(media_templates.TemplateDirective(reference, VALUES), session_factory=session_factory, user_id=media_setup, dialogue_id=1, topic_id=None)
        assert result.media_id == 3
        assert_pixels(source, result.image_bytes, config)


@pytest.mark.asyncio
async def test_schema_forward_upgrade_and_idempotence(async_engine, monkeypatch):
    async with async_engine.begin() as connection:
        await connection.execute(text('ALTER TABLE media_library DROP COLUMN render_template_enabled'))
        await connection.execute(text('ALTER TABLE media_library DROP COLUMN render_template_config'))
        await connection.execute(text("INSERT INTO media_library (id,file_id,media_type) VALUES (777,'old','photo')"))
        await connection.run_sync(database._migrate_media_templates)
        await connection.run_sync(database._migrate_media_templates)
        row = (await connection.execute(text('SELECT render_template_enabled,render_template_config FROM media_library WHERE id=777'))).one()
        assert not row[0] and row[1] is None
    if async_engine.dialect.name == 'postgresql':
        from sqlalchemy.ext.asyncio import async_sessionmaker
        async with async_engine.begin() as connection:
            await connection.execute(text('ALTER TABLE media_library DROP COLUMN render_template_enabled'))
            await connection.execute(text('ALTER TABLE media_library DROP COLUMN render_template_config'))
        monkeypatch.setattr(database, 'engine', async_engine)
        monkeypatch.setattr(database, 'async_session_maker', async_sessionmaker(async_engine, expire_on_commit=False))
        await database.init_db()


@pytest.mark.asyncio
async def test_admin_actual_navigation_setup_save_cancel_preview(session_factory, media_setup, monkeypatch):
    source = source_png()
    monkeypatch.setattr(handlers, 'is_admin', AsyncMock(return_value=True))
    transport = TelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher(); dispatcher.include_router(handlers.router)
    state = dispatcher.fsm.get_context(bot=bot, chat_id=media_setup, user_id=media_setup)
    await state.update_data(authoring_locale='ru', preserved_origin='test')
    latest = None
    visited = set()
    async def click(prefix, *, initial=False):
        nonlocal latest
        if initial:
            data = prefix
        else:
            buttons = [button for row in latest.reply_markup.inline_keyboard for button in row]
            button = next(button for button in buttons if button.callback_data.startswith(prefix))
            data = button.callback_data
            if data.startswith('mt:'):
                action = data.split(':')[1]
                assert action in BUTTON_CONTRACTS
                visited.add(action)
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(update_id=6000 + start, callback_query=CallbackQuery(id=str(start), from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'), chat_instance='ci', data=data, message=Message(message_id=6000, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='screen'))))
        screens = [call for call in transport.calls[start:] if getattr(call, 'reply_markup', None)]
        if screens: latest = screens[-1]
        return data
    async def enter(value):
        nonlocal latest
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(update_id=7000 + start, message=Message(message_id=7000 + start, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'), text=value)))
        latest = [call for call in transport.calls[start:] if getattr(call, 'reply_markup', None)][-1]
    try:
        await click('admin_panel', initial=True)
        await click('admin_collections_page_0')
        await click('admin_coll_view_1_0')
        await click('admin_coll_files_1_0_0')
        await click('admin_coll_file_view_1_10_0_0')
        await click('mt:home:')
        await state.update_data(authoring_locale='ru', preserved_origin='test')
        await click('mt:bind:')
        await click('mt:fields:')
        config = template_config(source)
        for key, field in config['fields'].items():
            await click('mt:add:')
            await enter(field_form(key, field))
            assert await state.get_state() == TemplateAdminState.confirming.state
            await click('mt:save:')
            assert key in latest.text and 'bbox=' in latest.text
            await click('mt:fields:')
        await click('mt:field:')
        await click('mt:edit:')
        await enter('bbox=invalid')
        assert await state.get_state() == TemplateAdminState.editing.state
        await click('mt:cancel:')
        assert 'recipient' in latest.text
        await click('mt:edit:')
        changed = {**config['fields']['recipient'], 'max_font_size': 35}
        await enter(field_form('recipient', changed))
        await click('mt:save:')
        assert 'max_font_size=35' in latest.text
        await click('mt:fields:')
        await click('mt:home:')
        await click('mt:toggle:')
        assert 'включен' in latest.text
        await click('mt:preview:')
        assert isinstance(latest, SendDocument)
        async with session_factory() as session:
            media = await session.get(MediaLibrary, 10)
            assert media.render_template_enabled
            saved = media.render_template_config
            assert saved['fields']['recipient']['max_font_size'] == 35
        assert_pixels(source, latest.document.data, saved)
        await click('mt:home:')
        await click('mt:back:')
        assert 'mt:home:10:1:0:0:' in str(latest.reply_markup)
        assert (await state.get_data())['authoring_locale'] == 'ru'
        await click('admin_coll_files_1_0_0')
        await click('admin_coll_file_view_1_10_0_0')
        await click('mt:home:')
        assert 'Поля: 2' in latest.text and 'включен' in latest.text
        assert visited == set(BUTTON_CONTRACTS)
    finally:
        dispatcher.sub_routers.clear(); handlers.router._parent_router = None; await bot.session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('action', list(BUTTON_CONTRACTS))
async def test_admin_direct_bypass_denied(session_factory, media_setup, action):
    transport = TelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher(); dispatcher.include_router(handlers.router)
    try:
        await dispatcher.feed_update(bot, Update(update_id=8000, callback_query=CallbackQuery(id='denied', from_user=TelegramUser(id=media_setup, is_bot=False, first_name='User'), chat_instance='ci', data=f'mt:{action}:10:1:2:3:-1', message=Message(message_id=8000, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='screen'))))
        assert any(isinstance(call, AnswerCallbackQuery) and call.show_alert and 'прав' in call.text for call in transport.calls)
        async with session_factory() as session:
            media = await session.get(MediaLibrary, 10)
            assert not media.render_template_enabled and media.render_template_config is None
    finally:
        dispatcher.sub_routers.clear(); handlers.router._parent_router = None; await bot.session.close()


@pytest.mark.asyncio
async def test_template_navigation_retains_nonzero_origin(session_factory, media_setup, monkeypatch):
    monkeypatch.setattr(handlers, 'is_admin', AsyncMock(return_value=True))
    transport = TelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher(); dispatcher.include_router(handlers.router)
    async def click(data):
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(update_id=9000 + start, callback_query=CallbackQuery(id=str(start), from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'), chat_instance='ci', data=data, message=Message(message_id=9000, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='screen'))))
        return [call for call in transport.calls[start:] if getattr(call, 'reply_markup', None)][-1]
    try:
        media = await click('admin_coll_file_view_1_10_2_3')
        template_button = next(b for row in media.reply_markup.inline_keyboard for b in row if b.callback_data.startswith('mt:'))
        template = await click(template_button.callback_data)
        assert all(':10:1:2:3:' in b.callback_data for row in template.reply_markup.inline_keyboard for b in row)
        back = next(b for row in template.reply_markup.inline_keyboard for b in row if b.callback_data.startswith('mt:back:'))
        media = await click(back.callback_data)
        assert 'admin_coll_files_1_2_3' in str(media.reply_markup)
    finally:
        dispatcher.sub_routers.clear(); handlers.router._parent_router = None; await bot.session.close()


@pytest.mark.asyncio
async def test_prompt_only_advertises_enabled_scoped_templates(session_factory, media_setup):
    from prompt_blocks import TELEGRAM_CAPABILITIES, MAX_CAPABILITIES, render_service_prompt
    from media_scope import load_available_media
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        _, media = await load_available_media(session, 1)
        block = format_template_sources(media)
        for caps in (TELEGRAM_CAPABILITIES, MAX_CAPABILITIES):
            prompt = render_service_prompt('Custom service without placeholders', capabilities=caps, media_edit_sources=block)
            assert 'RENDER_IMG_MEDIA' in prompt and 'panda_certificate' in prompt and 'recipient' in prompt
        assert not format_template_sources((await load_available_media(session, 2))[1])
        (await session.get(MediaLibrary, 3)).render_template_enabled = False
        (await session.get(MediaLibrary, 40)).render_template_enabled = False
        await session.commit()
        assert not format_template_sources((await load_available_media(session, 1))[1])


def test_renderer_catches_corruption_before_png_delivery(monkeypatch):
    original = Image.Image.save
    def corrupt(image, *args, **kwargs):
        image.putpixel((0, 0), (1, 2, 3, 4))
        return original(image, *args, **kwargs)
    source = source_png()
    monkeypatch.setattr(Image.Image, 'save', corrupt)
    with pytest.raises(TemplateRenderInvariantError):
        render_template(source, template_config(source), VALUES)
