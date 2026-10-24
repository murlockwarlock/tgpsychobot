import asyncio
import io
import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image
from aiogram import Bot, Dispatcher
from aiogram.methods import EditMessageText, SendPhoto
from aiogram.types import CallbackQuery, Chat, Message, Update, User as TelegramUser
from sqlalchemy import select

os.environ.setdefault('BOT_TOKEN', '123456:TEST_BOT_TOKEN')
os.environ.setdefault('DATABASE_URL', 'sqlite+aiosqlite:///:memory:')

import ai_integration
import handlers
import kie_image_inputs
import max_messenger_bot.ai as max_ai
import max_messenger_bot.app as max_app
from database import AIConfig, MediaLibrary, Message as DBMessage, User
from prompt_blocks import DEFAULT_SERVICE_PROMPT_TEMPLATE, MEDIA_EDIT_INSTRUCTION, TELEGRAM_CAPABILITIES, build_media_instruction_block, render_service_prompt
from tests.test_hide_empty_media_prompt import LEGACY_SERVICE_PROMPT_TEMPLATE
from tests.test_media_image_edit import async_engine, session_factory, clean_runtime, setup_tg_user, media_setup, TelegramTransport, MaxTransport


def png(width, height, *, orientation=None):
    stream = io.BytesIO()
    image = Image.new('RGB', (width, height), 'magenta')
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(stream, format='JPEG', exif=exif)
    else:
        image.save(stream, format='PNG')
    return stream.getvalue()


GENERATION_MODELS = ('bytedance/seedream-v4-text-to-image', 'seedream/4.5-text-to-image', 'google/imagen4-fast', 'google/imagen4-ultra')
EDIT_MODELS = ('bytedance/seedream-v4-edit', 'seedream/4.5-edit', 'google/nano-banana-edit')


@pytest.mark.parametrize('model', GENERATION_MODELS)
@pytest.mark.parametrize('prompt,ratio,size', [
    ('A vertical poster', '3:4', 'portrait_4_3'),
    ('A horizontal tarot poster', '4:3', 'landscape_4_3'),
    ('Обложка в формате 9:16', '9:16', 'portrait_16_9'),
    ('A wide banner 1920x1080', '16:9', 'landscape_16_9'),
    ('Квадратное изображение', '1:1', 'square_hd'),
    ('Vertical portrait, aspect ratio 16:9', '16:9', 'landscape_16_9'),
    ('A cardigan on a mannequin', '1:1', 'square_hd'),
    ('A horizontal view of a town square', '4:3', 'landscape_4_3'),
    ('Source 3:4, change to 16:9', '16:9', 'landscape_16_9'),
    ('A banner in 16:9, not 1:1', '16:9', 'landscape_16_9'),
    ('A vertical poster featuring a landscape painting', '3:4', 'portrait_4_3'),
    ('A horizontal poster featuring a portrait photograph', '4:3', 'landscape_4_3'),
    ('A vertical poster containing a square image', '3:4', 'portrait_4_3'),
    ('A square image containing vertical stripes', '1:1', 'square_hd'),
    ('Generate an image in 6:8', '3:4', 'portrait_4_3'),
    ('Generate an image in 5:7', '3:4', 'portrait_4_3'),
    ('A landscape banner with vertical stripes', '4:3', 'landscape_4_3'),
    ('A portrait poster with horizontal lines', '3:4', 'portrait_4_3'),
])
def test_generation_model_geometry(model, prompt, ratio, size):
    payload = kie_image_inputs.generation_input(model, prompt)
    assert payload['prompt'] == prompt
    if model.startswith('bytedance/'):
        assert payload['image_size'] == size
    else:
        assert payload['aspect_ratio'] == ratio


@pytest.mark.parametrize('model', EDIT_MODELS)
@pytest.mark.parametrize('width,height,ratio,size', [(600, 900, '2:3', 'portrait_3_2'), (1600, 900, '16:9', 'landscape_16_9'), (800, 800, '1:1', 'square_hd'), (904, 1280, '2:3', 'portrait_3_2')])
def test_edit_geometry_comes_from_source(model, width, height, ratio, size):
    prompt = 'Insert "ИВАН СОЛДАТОВ" and "24.10.2026". Preserve everything else, including the vertical certificate label.'
    payload = kie_image_inputs.edit_input(model, prompt, 'https://images.test/source.png', png(width, height))
    assert payload['prompt'] == prompt
    if model.startswith('bytedance/'):
        assert payload['image_size'] == size
    else:
        assert payload['aspect_ratio'] == ratio
    if model == 'google/nano-banana-edit':
        assert 'image_size' not in payload


@pytest.mark.parametrize('prompt,ratio,size', [('Crop the image to 16:9', '16:9', 'landscape_16_9'), ('Сделай изображение горизонтальным', '4:3', 'landscape_4_3'), ('Make the image square', '1:1', 'square_hd')])
def test_explicit_edit_shape_overrides_source(prompt, ratio, size):
    payload = kie_image_inputs.edit_input(EDIT_MODELS[0], prompt, 'https://images.test/source.png', png(600, 900))
    assert payload['image_size'] == size


def test_exif_orientation_is_respected():
    assert kie_image_inputs.source_ratio(png(900, 600, orientation=6)) == 2 / 3


@pytest.mark.parametrize('model', GENERATION_MODELS)
def test_unsupported_ratio_uses_nearest_documented_preset(model):
    payload = kie_image_inputs.generation_input(model, 'Instagram poster 4:5')
    assert payload.get('aspect_ratio') == '3:4' or payload.get('image_size') == 'portrait_4_3'


@pytest.mark.parametrize('template', [LEGACY_SERVICE_PROMPT_TEMPLATE, DEFAULT_SERVICE_PROMPT_TEMPLATE, 'Мой служебный блок'])
@pytest.mark.parametrize('enabled', [True, False])
def test_edit_contract_is_present_with_custom_service_templates(template, enabled):
    block = build_media_instruction_block('[PHOTO id=3] panda_certificate', allow_image_edit=enabled)
    prompt = render_service_prompt(template, capabilities=TELEGRAM_CAPABILITIES, available_media_text='[PHOTO id=3] panda_certificate', media_instruction_block=block)
    assert prompt.count(MEDIA_EDIT_INSTRUCTION) == int(enabled)


class KieWire:
    def __init__(self, source, result):
        self.source = source
        self.result = result
        self.tasks = []
        self.uploads = 0
        self.requests = []

    def handle(self, request):
        self.requests.append((request.method, request.url.path))
        if request.url.path == '/api/file-stream-upload':
            assert self.source in request.read()
            self.uploads += 1
            return httpx.Response(200, json={'code': 200, 'data': {'downloadUrl': 'https://images.test/source.png'}})
        if request.url.path == '/api/v1/jobs/createTask':
            self.tasks.append(json.loads(request.content))
            return httpx.Response(200, json={'code': 200, 'data': {'taskId': 'geometry-task'}})
        if request.url.path == '/api/v1/jobs/recordInfo':
            assert request.url.params['taskId'] == 'geometry-task'
            return httpx.Response(200, json={'code': 200, 'data': {'state': 'success', 'resultJson': json.dumps({'resultUrls': ['https://images.test/result.png']})}})
        if request.url.path == '/api/v1/common/download-url':
            return httpx.Response(200, json={'code': 200, 'data': 'https://images.test/result.png'})
        if request.url.host == 'images.test' and request.url.path == '/result.png':
            return httpx.Response(200, content=self.result, headers={'Content-Type': 'image/png'})
        raise AssertionError(f'Unexpected HTTP request: {request.method} {request.url.host}{request.url.path}')


def mock_kie(monkeypatch, source, result):
    wire = KieWire(source, result)
    client_type = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client_type(**{**kwargs, 'transport': httpx.MockTransport(wire.handle)}))
    return wire


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['telegram', 'max'])
@pytest.mark.parametrize('model', EDIT_MODELS)
async def test_actual_edit_adapter_upload_task_poll_download(monkeypatch, surface, model):
    module = ai_integration if surface == 'telegram' else max_ai
    source, result = png(904, 1280), png(1024, 1536)
    wire = mock_kie(monkeypatch, source, result)
    call = module._call_kie_image_edit if surface == 'telegram' else module._edit_kie
    prompt = 'Insert "ИВАН СОЛДАТОВ" and "24.10.2026". Preserve everything else.'
    assert await call('test-key', 'https://kie.test', 'https://uploads.test', model, prompt, source) == result
    assert wire.uploads == 1 and len(wire.tasks) == 1
    payload = wire.tasks[0]
    assert payload['model'] == model and payload['input']['prompt'] == prompt
    assert payload['input']['image_urls'] == ['https://images.test/source.png']
    assert payload['input'].get('image_size') == 'portrait_3_2' or payload['input'].get('aspect_ratio') == '2:3'
    assert wire.requests[-1] == ('GET', '/result.png')


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['telegram', 'max'])
async def test_invalid_source_fails_before_upload_or_paid_task(monkeypatch, surface):
    module = ai_integration if surface == 'telegram' else max_ai
    wire = mock_kie(monkeypatch, b'invalid image', b'result')
    call = module._call_kie_image_edit if surface == 'telegram' else module._edit_kie
    with pytest.raises(module.AIServiceError, match='исходное изображение'):
        await call('test-key', 'https://kie.test', 'https://uploads.test', EDIT_MODELS[0], 'Edit', b'invalid image')
    assert wire.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['telegram', 'max'])
@pytest.mark.parametrize('operation', ['edit', 'generate'])
async def test_real_runtime_through_actual_kie_adapter(session_factory, media_setup, monkeypatch, surface, operation):
    source = png(904, 1280)
    result = png(1024, 1536) if operation == 'edit' else png(576, 1024)
    wire = mock_kie(monkeypatch, source, result)
    prompt = 'Insert "ИВАН СОЛДАТОВ" and "24.10.2026". Preserve everything else.' if operation == 'edit' else 'A vertical poster in 9:16 format'
    response = f'Готово:\n[EDIT_IMG_MEDIA: panda_certificate | {prompt}]' if operation == 'edit' else f'Готово:\nGEN_IMG: [{prompt}]'
    async with session_factory() as session:
        config = await session.get(AIConfig, 1)
        config.allow_image_generation = True
        config.image_generation_provider = 'KIE'
        config.image_generation_model = GENERATION_MODELS[0]
        config.image_edit_model = EDIT_MODELS[0]
        config.kie_base_url = 'https://kie.test'
        config.kie_upload_base_url = 'https://uploads.test'
        config.service_prompt_block = LEGACY_SERVICE_PROMPT_TEMPLATE
        media = await session.get(MediaLibrary, 10 if surface == 'telegram' else 40)
        media.file_name = 'panda_certificate'
        await session.commit()
    if surface == 'telegram':
        transport = TelegramTransport()
        async def stream(*args, **kwargs):
            yield source
        monkeypatch.setattr(transport, 'stream_content', stream)
        bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
        handlers.router._parent_router = None
        dispatcher = Dispatcher()
        dispatcher.include_router(handlers.router)
        model = AsyncMock(return_value=response)
        monkeypatch.setattr(ai_integration, '_call_gemini_api', model)
        try:
            await dispatcher.feed_update(bot, Update(update_id=990, message=Message(message_id=990, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Denis'), text='ИВАН СОЛДАТОВ, 24.10.2026' if operation == 'edit' else 'Сделай вертикальный постер 9:16')))
            if media_setup in handlers.user_processing_tasks:
                await handlers.user_processing_tasks[media_setup]
            photos = [item for item in transport.calls if isinstance(item, SendPhoto)]
            assert len(photos) == 1
            outgoing = SendPhoto.model_validate(photos[0].model_dump())
            assert outgoing.photo.data == result
            assert all('GEN_IMG' not in (getattr(item, 'text', '') or '') and 'EDIT_IMG_MEDIA' not in (getattr(item, 'text', '') or '') for item in transport.calls)
            if operation == 'edit':
                assert MEDIA_EDIT_INSTRUCTION in '\n'.join(model.await_args.kwargs['request_layout'].shared_instructions)
        finally:
            dispatcher.sub_routers.clear()
            handlers.router._parent_router = None
            await bot.session.close()
        user_id = media_setup
    else:
        client = MaxTransport()
        monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))
        monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=response))
        application = max_app.MaxBotApplication(client)
        await application.handle_update({'update_type': 'message_created', 'update_id': 'geometry', 'message': {'sender': {'user_id': 7, 'name': 'Denis'}, 'recipient': {'chat_id': 100}, 'body': {'mid': 'geometry-input', 'text': 'ИВАН СОЛДАТОВ, 24.10.2026' if operation == 'edit' else 'Сделай вертикальный постер 9:16'}}})
        if application.background_tasks:
            await asyncio.gather(*list(application.background_tasks))
        assert client.uploads == [('image', result)]
        images = [item for _, _, kwargs in client.wire for item in kwargs.get('json_data', {}).get('attachments', []) if item.get('type') == 'image']
        assert images == [{'type': 'image', 'payload': {'token': 'edited-token'}}]
        assert all('GEN_IMG' not in (kwargs.get('json_data', {}).get('text') or '') and 'EDIT_IMG_MEDIA' not in (kwargs.get('json_data', {}).get('text') or '') for _, _, kwargs in client.wire)
        user_id = 100000000007
    assert len(wire.tasks) == 1
    assert wire.tasks[0]['input']['image_size'] == ('portrait_3_2' if operation == 'edit' else 'portrait_16_9')
    async with session_factory() as session:
        user = await session.get(User, user_id)
        assert user.current_topic_id == 1 and user.current_dialogue_id == 1
        history = (await session.scalars(select(DBMessage).where(DBMessage.user_id == user_id, DBMessage.role == 'assistant'))).all()
        assert history and all(row.topic_id == 1 and row.dialogue_id == 1 for row in history)
        if operation == 'edit':
            assert all('EDIT_IMG_MEDIA' not in row.content for row in history)


@pytest.mark.asyncio
@pytest.mark.parametrize('generation,editing', [(False, True), (True, False)])
async def test_telegram_settings_display_capability_status(session_factory, media_setup, monkeypatch, generation, editing):
    async with session_factory() as session:
        config = await session.get(AIConfig, 1)
        config.allow_image_generation = generation
        config.allow_image_edit = editing
        await session.commit()
    transport = TelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)
    try:
        await dispatcher.feed_update(bot, Update(update_id=991, callback_query=CallbackQuery(id='settings-status', from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'), chat_instance='ci', data='admin_ai_settings', message=Message(message_id=991, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='Настройки'))))
        screen = next(item.text for item in transport.calls if isinstance(item, EditMessageText))
        gen = screen.split('Генерация изображений:', 1)[1].split('Редактирование изображений:', 1)[0]
        edit = screen.split('Редактирование изображений:', 1)[1].split('KIE кредиты:', 1)[0]
        assert f'Статус: <b>{"Включено" if generation else "Выключено"}</b>' in gen
        assert f'Статус: <b>{"Включено" if editing else "Выключено"}</b>' in edit
        assert 'seedream/4.5-edit' in edit
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['telegram', 'max'])
@pytest.mark.parametrize('operation', ['edit', 'generate'])
async def test_wrong_provider_geometry_is_rejected_without_paid_retry(monkeypatch, surface, operation):
    module = ai_integration if surface == 'telegram' else max_ai
    source, result = png(904, 1280), png(1024, 1024)
    wire = mock_kie(monkeypatch, source, result)
    with pytest.raises(module.AIServiceError, match='пропорций'):
        if operation == 'edit':
            call = module._call_kie_image_edit if surface == 'telegram' else module._edit_kie
            await call('test-key', 'https://kie.test', 'https://uploads.test', EDIT_MODELS[0], 'Insert the name. Preserve everything else.', source)
        else:
            call = module._call_kie_image_generation if surface == 'telegram' else module._generate_kie
            await call('test-key', 'https://kie.test', GENERATION_MODELS[0], 'Vertical poster 9:16')
    assert len(wire.tasks) == 1


@pytest.mark.parametrize('prompt', ['Do not make the image square', 'Do not change the aspect ratio to 1:1', 'Keep the vertical certificate label unchanged'])
def test_negated_resize_does_not_override_source(prompt):
    payload = kie_image_inputs.edit_input(EDIT_MODELS[0], prompt, 'https://images.test/source.png', png(600, 900))
    assert payload['image_size'] == 'portrait_3_2'


@pytest.mark.parametrize('reference', ['6:8', '5:7'])
@pytest.mark.parametrize('model', EDIT_MODELS)
def test_arbitrary_explicit_ratio_overrides_landscape_source(model, reference):
    payload = kie_image_inputs.edit_input(model, f'Resize to {reference}', 'https://images.test/source.png', png(1600, 900))
    assert payload.get('aspect_ratio') == '3:4' or payload.get('image_size') == 'portrait_4_3'


@pytest.mark.parametrize('model', EDIT_MODELS)
@pytest.mark.parametrize('prompt', [
    'Replace the caption with "Starts at 12:30". Preserve everything else.',
    "Replace the caption with 'Starts at 12:30'. Preserve everything else.",
    'Впиши «Начало в 12:30». Остальное без изменений.',
    'Print “Set aspect ratio 12:30” in the caption. Preserve everything else.',
    'Replace the label with `904x1280`. Preserve everything else.',
    'Write the time 12:30 in the caption. Preserve everything else.',
])
def test_caption_literals_do_not_change_canvas_geometry(model, prompt):
    payload = kie_image_inputs.edit_input(model, prompt, 'https://images.test/source.png', png(1600, 900))
    assert payload.get('aspect_ratio') == '16:9' or payload.get('image_size') == 'landscape_16_9'


@pytest.mark.parametrize('prompt', ['Resize to "6:8"', "Resize to '6:8'", 'Сделай формат «6:8»', 'Make it 6:8', '6:8'])
def test_explicit_resize_supports_quoted_ratios(prompt):
    payload = kie_image_inputs.edit_input(EDIT_MODELS[0], prompt, 'https://images.test/source.png', png(1600, 900))
    assert payload['image_size'] == 'portrait_4_3'


@pytest.mark.parametrize('model', GENERATION_MODELS)
def test_generation_caption_time_does_not_override_orientation(model):
    payload = kie_image_inputs.generation_input(model, 'Horizontal poster with caption "Starts at 12:30"')
    assert payload.get('aspect_ratio') == '4:3' or payload.get('image_size') == 'landscape_4_3'


@pytest.mark.parametrize('model', EDIT_MODELS)
@pytest.mark.parametrize('width,height,ratio,size', [(800, 800, '1:1', 'square_hd'), (600, 900, '2:3', 'portrait_3_2'), (1600, 900, '16:9', 'landscape_16_9')])
@pytest.mark.parametrize('prompt', [
    'Make the portrait brighter. Preserve everything else.',
    'Make the landscape greener. Preserve everything else.',
    'Make the text horizontal. Preserve the image dimensions.',
    'Make horizontal text. Preserve everything else.',
    'Make the logo square. Preserve everything else.',
    'Сделай портрет ярче. Остальное без изменений.',
])
def test_subject_edits_do_not_override_canvas(model, width, height, ratio, size, prompt):
    payload = kie_image_inputs.edit_input(model, prompt, 'https://images.test/source.png', png(width, height))
    assert payload.get('aspect_ratio') == ratio or payload.get('image_size') == size


@pytest.mark.parametrize('prompt,size', [
    ('Make the image portrait', 'portrait_4_3'),
    ('Convert it to landscape', 'landscape_4_3'),
    ('Crop to square', 'square_hd'),
    ('Change canvas orientation to landscape', 'landscape_4_3'),
])
def test_explicit_canvas_orientation_changes_are_supported(prompt, size):
    payload = kie_image_inputs.edit_input(EDIT_MODELS[0], prompt, 'https://images.test/source.png', png(600, 900))
    assert payload['image_size'] == size


@pytest.mark.parametrize('model', EDIT_MODELS)
@pytest.mark.parametrize('prompt', [
    'Write "Make the image portrait" in the caption. Preserve everything else.',
    'Replace the label with «landscape orientation». Preserve everything else.',
    'Write “Resize the image to 6:8” in the caption. Preserve everything else.',
])
def test_literal_orientation_commands_do_not_resize_image(model, prompt):
    payload = kie_image_inputs.edit_input(model, prompt, 'https://images.test/source.png', png(800, 800))
    assert payload.get('aspect_ratio') == '1:1' or payload.get('image_size') == 'square_hd'


@pytest.mark.parametrize('model', GENERATION_MODELS)
def test_generation_literal_orientation_is_caption_content(model):
    payload = kie_image_inputs.generation_input(model, 'A poster with caption "horizontal image"')
    assert payload.get('aspect_ratio') == '3:4' or payload.get('image_size') == 'portrait_4_3'


@pytest.mark.parametrize('model', GENERATION_MODELS)
def test_wrapped_generation_prompt_still_honors_ratio(model):
    payload = kie_image_inputs.generation_input(model, '"A vertical poster in 9:16"')
    assert payload.get('aspect_ratio') == '9:16' or payload.get('image_size') == 'portrait_16_9'
