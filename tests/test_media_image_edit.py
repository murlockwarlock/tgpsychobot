import asyncio
import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault('BOT_TOKEN', '123456:TEST_BOT_TOKEN')
os.environ.setdefault('DATABASE_URL', 'sqlite+aiosqlite:///:memory:')

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.methods import GetFile, SendPhoto, SendMessage
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Chat, File, Message, PhotoSize, Update, User as TelegramUser
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
import handlers
import max_messenger_bot.ai as max_ai
import max_messenger_bot.app as max_app
import max_messenger_bot.storage as max_storage
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.services import common
from database import AIConfig, BotGeneralConfig, SubscriptionConfig, MediaLibrary, MediaCollection, Topic, User, Message as DBMessage, media_collection_items, topic_collection_association, main_dialogue_collection_association
from media_image_edit import extract_media_edit_directive, resolve_editable_media_source, MediaSourceError, load_media_image_bytes
from prompt_blocks import format_media_edit_sources, MAX_CAPABILITIES
from tests.test_ai_terminal_failure_retry import session_factory, clean_runtime, ValidatingTelegramSession


@pytest_asyncio.fixture
async def async_engine():
    from database import Base
    from max_messenger_bot.storage import StorageBase
    url = os.getenv('MEDIA_EDIT_TEST_DATABASE_URL')
    schema = 'media_edit_test_' + uuid.uuid4().hex
    if url:
        from sqlalchemy.engine import make_url
        parsed = make_url(url)
        assert parsed.host in ('localhost', '127.0.0.1') and parsed.database.startswith('tgpsychobot_media_edit_test')
        admin_engine = create_async_engine(url)
        async with admin_engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA {schema}'))
        engine = create_async_engine(url, connect_args={'server_settings': {'search_path': schema}})
    else:
        engine = create_async_engine('sqlite+aiosqlite:///:memory:', poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(StorageBase.metadata.create_all)
        yield engine
    finally:
        await engine.dispose()
        if url:
            async with admin_engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
            await admin_engine.dispose()


@pytest.mark.parametrize('text', ['[EDIT_IMG_MEDIA: 153 | make the sky blue]', '[EDIT_IMG_MEDIA : 153| make the sky blue ]', '[EDIT_IMG_MEDIA: 153 | make\nthe sky blue]'])
def test_parser(text):
    directive, clean = extract_media_edit_directive('Вот вариант:\n' + text)
    assert directive.media_id == 153
    assert directive.prompt == ('make\nthe sky blue' if '\n' in text else 'make the sky blue')
    assert clean == 'Вот вариант:'


@pytest.mark.parametrize('text', ['[EDIT_IMG_MEDIA:]', '[EDIT_IMG_MEDIA: abc | x]', '[EDIT_IMG_MEDIA: -1 | x]', '[EDIT_IMG_MEDIA: 1]', '[EDIT_IMG_MEDIA: 0 | x]', '[EDIT_IMG_MEDIA: 999999999999999999 | x]', '[EDIT_IMG_MEDIA: https://x | x]', '[EDIT_IMG_MEDIA: ../../../ | x]', '[EDIT_IMG_MEDIA: 1 | ]', '[EDIT_IMG_MEDIA: 1 | x', '[EDIT_IMG_MEDIA: 1 | x]\n[EDIT_IMG_MEDIA: 2 | y]'])
def test_malformed(text):
    directive, clean = extract_media_edit_directive(text)
    assert directive is None
    assert 'EDIT_IMG_MEDIA' not in clean


def test_uploaded_contract_is_independent():
    assert extract_media_edit_directive('EDIT_IMG: make sky blue') == (None, 'EDIT_IMG: make sky blue')
    assert handlers._extract_ai_directive_payload('[EDIT_IMG_MEDIA: 1 | blue]', 'EDIT_IMG')[0] is None
    assert common._extract_ai_directive_payload('[EDIT_IMG_MEDIA: 1 | blue]', 'EDIT_IMG')[0] is None


@pytest_asyncio.fixture
async def setup_tg_user(session_factory, monkeypatch):
    async with session_factory() as session:
        session.add_all([
            AIConfig(id=1,provider='Gemini',gemini_model='gemini-2.5-flash',gemini_api_key='test-key'),
            SubscriptionConfig(id=1,subscriptions_enabled=False),
            BotGeneralConfig(id=1,profile_collect_name=False,profile_collect_gender=False,profile_collect_age=False),
            User(id=1001,name='Alice',first_name='Alice',gender='female',age='25',accepted_disclaimer=True,current_dialogue_id=1),
        ])
        await session.commit()
    monkeypatch.setattr(handlers,'async_session_maker',session_factory)
    monkeypatch.setattr(ai_integration,'async_session_maker',session_factory)
    monkeypatch.setattr(handlers,'is_admin',AsyncMock(return_value=False))
    return 1001


@pytest_asyncio.fixture
async def media_setup(session_factory, setup_tg_user, monkeypatch):
    for module in (common, max_ai, max_app, max_storage):
        monkeypatch.setattr(module, 'async_session_maker', session_factory)
    async with session_factory() as session:
        session.add_all([Topic(id=1, name='A'), Topic(id=2, name='B'), MediaCollection(id=1, name='A'), MediaCollection(id=2, name='B')])
        session.add_all([
            MediaLibrary(id=10, file_id='telegram-source', file_name='portrait', description='Portrait', media_type='photo'),
            MediaLibrary(id=20, file_id='other-source', file_name='foreign', media_type='photo', source_platform='telegram'),
            MediaLibrary(id=30, file_id='audio', file_name='audio', media_type='audio'),
            MediaLibrary(id=40, file_id='max-source', file_name='max-portrait', media_type='photo', source_platform='max', source_url='https://img.oneme.ru/source.jpg'),
            MediaLibrary(id=41, file_id='max-token-only', file_name='max-token', media_type='photo', source_platform='max'),
            MediaLibrary(id=42, file_id='legacy-max-source', file_name='legacy-max', media_type='photo'),
        ])
        await session.flush()
        await session.execute(topic_collection_association.insert(), [{'topic_id': 1, 'collection_id': 1}, {'topic_id': 2, 'collection_id': 2}])
        await session.execute(main_dialogue_collection_association.insert().values(collection_id=1))
        await session.execute(media_collection_items.insert(), [{'collection_id': 1, 'media_id': 10}, {'collection_id': 2, 'media_id': 20}, {'collection_id': 1, 'media_id': 30}, {'collection_id': 1, 'media_id': 40}])
        await session.execute(media_collection_items.insert(), [{'collection_id':1,'media_id':41},{'collection_id':1,'media_id':42}])
        user = await session.get(User, setup_tg_user)
        user.current_topic_id = 1
        session.add(User(id=100000000007, name='Max', first_name='Max', gender='male', age='25', accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=1))
        config = await session.get(AIConfig, 1)
        config.image_edit_provider = 'KIE'
        config.image_edit_model = 'seedream/4.5-edit'
        config.kie_api_key = 'test-key'
        config.allow_image_edit = True
        await session.commit()
    return setup_tg_user


@pytest.mark.asyncio
@pytest.mark.parametrize('topic,media_id,allowed', [(1,10,True),(1,20,False),(2,20,True),(None,10,True),(None,20,False),(1,30,False),(1,999,False)])
async def test_scope(session_factory, media_setup, topic, media_id, allowed):
    async with session_factory() as session:
        if allowed:
            assert (await resolve_editable_media_source(session, media_id=media_id, topic_id=topic)).id == media_id
        else:
            with pytest.raises(MediaSourceError):
                await resolve_editable_media_source(session, media_id=media_id, topic_id=topic)


class TelegramTransport(ValidatingTelegramSession):
    async def make_request(self, bot, method, timeout=None):
        type(method).model_validate(method.model_dump())
        if isinstance(method, GetFile):
            self.calls.append(method)
            if method.file_id == 'legacy-max-source':
                raise TelegramBadRequest(method=method,message='wrong file identifier/HTTP URL specified')
            assert method.file_id == 'telegram-source'
            return File(file_id=method.file_id, file_unique_id='unique', file_path='photos/source.jpg')
        return await super().make_request(bot, method, timeout)

    async def stream_content(self, *args, **kwargs):
        yield b'original-telegram-image'


@pytest.mark.asyncio
@pytest.mark.parametrize('media_id,stale,download_failure,enabled', [(10,False,False,True),(20,False,False,True),(30,False,False,True),(999,False,False,True),(10,True,False,True),(10,False,True,True),(10,False,False,False)])
async def test_telegram_dispatcher_journey(session_factory, media_setup, monkeypatch, media_id, stale, download_failure, enabled):
    async with session_factory() as session:
        (await session.get(AIConfig, 1)).allow_image_edit = enabled
        await session.commit()
    transport = TelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)
    chat_calls = []
    async def chat(*args, **kwargs):
        chat_calls.append((args, kwargs))
        return f'Вот вариант:\n[EDIT_IMG_MEDIA: {media_id} | make the sky blue]'
    monkeypatch.setattr(ai_integration, '_call_gemini_api', chat)
    edits = []
    async def edit(key, base, upload_base, model, prompt, source):
        edits.append((model, prompt, source))
        if stale:
            async with session_factory() as session:
                user = await session.get(User, media_setup)
                user.current_topic_id = 2
                await session.commit()
        return b'edited-output'
    monkeypatch.setattr(ai_integration, '_call_kie_image_edit', edit)
    if download_failure:
        monkeypatch.setattr(bot, 'get_file', AsyncMock(side_effect=RuntimeError('download unavailable')))
    try:
        await dispatcher.feed_update(bot, Update(update_id=1, message=Message(message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'), text='Измени портрет')))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]
        assert chat_calls
        photos = [call for call in transport.calls if isinstance(call, SendPhoto)]
        assert len(photos) == int(media_id == 10 and not stale and not download_failure and enabled)
        if photos:
            assert photos[0].photo.data == b'edited-output'
        assert edits == ([('seedream/4.5-edit', 'make the sky blue', b'original-telegram-image')] if media_id == 10 and not download_failure and enabled else [])
        if not enabled:
            assert not any(isinstance(call, GetFile) for call in transport.calls)
            assert 'EDIT_IMG_MEDIA' not in str(chat_calls)
        assert all('EDIT_IMG_MEDIA' not in (getattr(call, 'text', '') or '') for call in transport.calls)
        if stale:
            assert not any('Вот вариант' in (getattr(call,'text','') or '') for call in transport.calls)
        assert await dispatcher.fsm.get_context(bot=bot, chat_id=media_setup, user_id=media_setup).get_state() is None
        async with session_factory() as session:
            history = (await session.scalars(select(DBMessage).where(DBMessage.user_id == media_setup, DBMessage.role == 'assistant'))).all()
            assert bool(history) is (not stale)
            if history and (media_id != 10 or download_failure or not enabled):
                assert '[Изображение]' not in history[0].content
                assert any(text in history[0].content for text in ('недоступно', 'Не удалось', 'отключено'))
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


class MaxTransport(MaxApiClient):
    def __init__(self):
        super().__init__('test', 'https://max.test')
        self.wire = []
        self.uploads = []
        self.sources = []

    async def _request(self, method, path, **kwargs):
        self.wire.append((method, path, kwargs))
        return {'message': {'body': {'mid': str(len(self.wire))}}}

    async def download_attachment_url(self, url):
        self.sources.append(url)
        return b'original-max-image'

    async def download_attachment(self, token, url=None):
        assert token in ('max-token-only','legacy-max-source')
        self.sources.append(token)
        return b'original-max-image'

    async def upload_file(self, media_type, path):
        self.uploads.append((media_type, Path(path).read_bytes()))
        return {'token': 'edited-token'}


@pytest.mark.asyncio
@pytest.mark.parametrize('media_id,stale,enabled', [(10,False,True),(40,False,True),(41,False,True),(42,False,True),(20,False,True),(30,False,True),(999,False,True),(40,True,True),(40,False,False)])
async def test_max_handle_update_journey(session_factory, media_setup, monkeypatch, media_id, stale, enabled):
    async with session_factory() as session:
        (await session.get(AIConfig, 1)).allow_image_edit = enabled
        await session.commit()
    import telegram_client
    monkeypatch.setattr(telegram_client,'create_telegram_bot',lambda token: Bot('123456:TEST_BOT_TOKEN',session=TelegramTransport()))
    client = MaxTransport()
    application = max_app.MaxBotApplication(client)
    layouts = []
    async def chat(config, layout, **kwargs):
        layouts.append(layout)
        return f'Вот вариант:\n[EDIT_IMG_MEDIA: {media_id} | make the sky blue]'
    monkeypatch.setattr(max_ai, '_dispatch_provider', chat)
    edits = []
    async def edit(key, base, upload_base, model, prompt, source):
        edits.append((model, prompt, source))
        if stale:
            async with session_factory() as session:
                user = await session.get(User, 100000000007)
                user.current_topic_id = 2
                await session.commit()
        return b'edited-output'
    monkeypatch.setattr(max_ai, '_edit_kie', edit)
    await application.handle_update({'update_type': 'message_created', 'update_id': 'media-edit', 'message': {'sender': {'user_id': 7, 'name': 'Max'}, 'recipient': {'chat_id': 100}, 'body': {'mid': 'incoming', 'text': 'Измени портрет'}}})
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))
    assert layouts
    prompt = '\n'.join(layouts[0].shared_instructions)
    assert ('[PHOTO id=40]' in prompt) is enabled and '[PHOTO id=20]' not in prompt
    assert ('EDIT_IMG_MEDIA' in prompt) is enabled
    assert all(command not in prompt for command in ('SEND_AUDIO','RANDOM_IMG','CHOICE_IMG','SHOW_IMG'))
    attachments = [kwargs['json_data']['attachments'] for method,path,kwargs in client.wire if method == 'POST' and kwargs.get('json_data', {}).get('attachments')]
    image_payloads = [item for group in attachments for item in group if item['type'] == 'image']
    assert image_payloads == ([{'type':'image','payload':{'token':'edited-token'}}] if media_id in (10,40,41,42) and not stale and enabled else [])
    expected_source = b'original-telegram-image' if media_id == 10 else b'original-max-image'
    assert edits == ([('seedream/4.5-edit','make the sky blue',expected_source)] if media_id in (10,40,41,42) and enabled else [])
    assert client.uploads == ([('image',b'edited-output')] if media_id in (10,40,41,42) and not stale and enabled else [])
    if not enabled:
        assert client.sources == []
    assert all('EDIT_IMG_MEDIA' not in (kwargs.get('json_data', {}).get('text') or '') for _,_,kwargs in client.wire)
    async with session_factory() as session:
        history = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 100000000007, DBMessage.role == 'assistant'))).all()
        assert bool(history) is (not stale)
        if history and media_id not in (10,40,41,42):
            assert 'недоступно' in history[0].content
        if not enabled:
            assert 'отключено' in history[0].content
    if stale:
        assert not any('Вот вариант' in (kwargs.get('json_data',{}).get('text') or '') for _,_,kwargs in client.wire)


@pytest.mark.asyncio
async def test_max_token_is_never_sent_to_telegram():
    media = SimpleNamespace(source_platform='max', source_url=None, file_id='max-token')
    bot = SimpleNamespace(get_file=AsyncMock())
    with pytest.raises(MediaSourceError):
        await load_media_image_bytes(media, telegram_bot=bot)
    bot.get_file.assert_not_called()


@pytest.mark.asyncio
async def test_disable_during_download_prevents_provider(session_factory, media_setup, monkeypatch):
    import media_image_edit
    async def download(*args, **kwargs):
        async with session_factory() as session:
            (await session.get(AIConfig, 1)).allow_image_edit = False
            await session.commit()
        return b'original-image'
    monkeypatch.setattr(media_image_edit, 'load_media_image_bytes', download)
    edit = AsyncMock()
    with pytest.raises(MediaSourceError, match='отключено'):
        await media_image_edit.edit_media_image(
            media_image_edit.MediaEditDirective(10, 'make the sky blue'),
            session_factory=session_factory, user_id=media_setup,
            dialogue_id=1, topic_id=1, edit_image=edit,
        )
    edit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('media_type', ['audio', 'video', 'document'])
async def test_non_photos_denied(session_factory, media_setup, media_type):
    async with session_factory() as session:
        media = await session.get(MediaLibrary, 10)
        media.media_type = media_type
        await session.commit()
        with pytest.raises(MediaSourceError):
            await resolve_editable_media_source(session, media_id=10, topic_id=1)


@pytest.mark.asyncio
async def test_uploaded_telegram_dispatcher_contract(session_factory, media_setup, monkeypatch):
    transport = TelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)
    analyze = AsyncMock(return_value='EDIT_IMG: make uploaded sky blue')
    monkeypatch.setattr(ai_integration, 'analyze_image_content', analyze)
    edit = AsyncMock(return_value=b'uploaded-edit-output')
    monkeypatch.setattr(ai_integration, '_call_kie_image_edit', edit)
    try:
        await dispatcher.feed_update(bot, Update(update_id=4, message=Message(message_id=4, date=datetime.now(timezone.utc), chat=Chat(id=media_setup,type='private'), from_user=TelegramUser(id=media_setup,is_bot=False,first_name='Alice'), caption='Измени это фото', photo=[PhotoSize(file_id='telegram-source',file_unique_id='unique',width=100,height=100)])))
        assert analyze.await_args.args[0] == b'original-telegram-image'
        assert edit.await_args.args[-3:] == ('seedream/4.5-edit','make uploaded sky blue',b'original-telegram-image')
        assert [call.photo.data for call in transport.calls if isinstance(call, SendPhoto)] == [b'uploaded-edit-output']
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None


@pytest.mark.asyncio
async def test_uploaded_max_handle_update_contract(session_factory, media_setup, monkeypatch):
    client = MaxTransport()
    monkeypatch.setattr(client,'download_attachment',AsyncMock(return_value=b'uploaded-max-image'))
    analyze = AsyncMock(return_value='EDIT_IMG: make uploaded sky blue')
    monkeypatch.setattr(max_ai,'analyze_image',analyze)
    edit = AsyncMock(return_value=b'uploaded-edit-output')
    monkeypatch.setattr(max_ai,'_edit_kie',edit)
    application = max_app.MaxBotApplication(client)
    await application.handle_update({'update_type':'message_created','update_id':'uploaded-photo','message':{'sender':{'user_id':7},'recipient':{'chat_id':100},'body':{'mid':'source','text':'Измени это фото','attachments':[{'type':'image','payload':{'token':'uploaded-token','url':'https://img.oneme.ru/upload.jpg'}}]}}})
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))
    assert analyze.await_args.args[1] == b'uploaded-max-image'
    assert edit.await_args.args[-3:] == ('seedream/4.5-edit','make uploaded sky blue',b'uploaded-max-image')
    assert client.uploads == [('image',b'uploaded-edit-output')]
    assert any(kwargs.get('json_data',{}).get('attachments') == [{'type':'image','payload':{'token':'edited-token'}}] for _,_,kwargs in client.wire)


@pytest.mark.asyncio
async def test_cached_max_source_survives_url_expiry(session_factory, media_setup):
    async with session_factory() as session:
        media = await session.get(MediaLibrary,40)
        media.source_image_bytes = b'persistent-original'
        await session.commit()
    async with session_factory() as session:
        media = await resolve_editable_media_source(session,media_id=40,topic_id=1,include_source_bytes=True)
    client = SimpleNamespace(download_attachment_url=AsyncMock(side_effect=RuntimeError('expired')))
    assert await load_media_image_bytes(media,max_client=client) == b'persistent-original'
    client.download_attachment_url.assert_not_called()


@pytest.mark.asyncio
async def test_max_cached_source_read_once(async_engine, session_factory, media_setup, monkeypatch):
    from sqlalchemy import event
    async with session_factory() as session:
        (await session.get(MediaLibrary, 40)).source_image_bytes = b'cached-original'
        await session.commit()
    statements = []
    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('SELECT'):
            statements.append(statement)
    event.listen(async_engine.sync_engine, 'before_cursor_execute', record)
    client = MaxTransport()
    application = max_app.MaxBotApplication(client)
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value='[EDIT_IMG_MEDIA: 40 | make the sky blue]'))
    edit = AsyncMock(return_value=b'edited-cached-image')
    monkeypatch.setattr(max_ai, '_edit_kie', edit)
    try:
        await application.handle_update({'update_type': 'message_created', 'update_id': 'cached-edit', 'message': {'sender': {'user_id': 7, 'name': 'Max'}, 'recipient': {'chat_id': 100}, 'body': {'mid': 'cached-incoming', 'text': 'Измени портрет'}}})
        if application.background_tasks:
            await asyncio.gather(*list(application.background_tasks))
    finally:
        event.remove(async_engine.sync_engine, 'before_cursor_execute', record)
    assert len([statement for statement in statements if 'media_library.source_image_bytes' in statement]) == 1
    assert edit.await_args.args[-1] == b'cached-original'
    assert client.sources == []
    assert client.uploads == [('image', b'edited-cached-image')]
    assert any({'type': 'image', 'payload': {'token': 'edited-token'}} in kwargs.get('json_data', {}).get('attachments', []) for _, _, kwargs in client.wire)


@pytest.mark.asyncio
@pytest.mark.parametrize('phase',['inference','download','edit','upload'])
async def test_max_stale_dialogue(session_factory, media_setup, monkeypatch, phase):
    client = MaxTransport()
    async def switch():
        async with session_factory() as session:
            user = await session.get(User,100000000007)
            user.current_dialogue_id = 2
            await session.commit()
    async def chat(*args,**kwargs):
        if phase == 'inference':
            await switch()
        return 'Вот вариант:\n[EDIT_IMG_MEDIA: 40 | blue sky]'
    async def download(url):
        if phase == 'download':
            await switch()
        return b'original'
    async def edit(*args):
        if phase == 'edit':
            await switch()
        return b'edited'
    async def upload(*args):
        if phase == 'upload':
            await switch()
        return {'token':'edited-token'}
    monkeypatch.setattr(max_ai,'_dispatch_provider',chat)
    monkeypatch.setattr(client,'download_attachment_url',download)
    monkeypatch.setattr(max_ai,'_edit_kie',edit)
    monkeypatch.setattr(client,'upload_file',upload)
    await common.run_ai_dialogue(client,100,100000000007,'Измени портрет')
    assert not any(item.get('type') == 'image' for _,_,kwargs in client.wire for item in kwargs.get('json_data',{}).get('attachments',[]))
    assert not any('Вот вариант' in (kwargs.get('json_data',{}).get('text') or '') for _,_,kwargs in client.wire)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['download','provider','malformed'])
async def test_max_failures_no_output(session_factory,media_setup,monkeypatch,failure):
    client = MaxTransport()
    response = '[EDIT_IMG_MEDIA: abc | x]' if failure == 'malformed' else '[EDIT_IMG_MEDIA: 40 | blue sky]'
    monkeypatch.setattr(max_ai,'_dispatch_provider',AsyncMock(return_value=response))
    provider_error = ai_integration.AIServiceError('provider unavailable')
    provider_error.classification = 'provider_unavailable'
    edit = AsyncMock(side_effect=provider_error)
    monkeypatch.setattr(max_ai,'_edit_kie',edit)
    if failure == 'download':
        monkeypatch.setattr(client,'download_attachment_url',AsyncMock(side_effect=RuntimeError('failed')))
    await common.run_ai_dialogue(client,100,100000000007,'Измени портрет')
    assert edit.await_count == int(failure == 'provider')
    assert not client.uploads
    assert any('Не удалось' in (kwargs.get('json_data',{}).get('text') or '') for _,_,kwargs in client.wire)
    async with session_factory() as session:
        history = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 100000000007,DBMessage.role == 'assistant'))).all()
        assert len(history) == 1
        assert history[0].topic_id == 1 and history[0].dialogue_id == 1
        assert history[0].content == next(kwargs['json_data']['text'] for _,_,kwargs in client.wire if 'Не удалось' in (kwargs.get('json_data',{}).get('text') or ''))


@pytest.mark.asyncio
async def test_max_admin_preserves_source_bytes(session_factory,media_setup):
    from max_messenger_bot.services import admin_topic_media
    from tests.test_max_media_admin import FakeState
    from unittest.mock import patch
    state = FakeState()
    state.state = 'admin_media_add_desc'
    state.data = {'topic_id':1,'collection_id':1,'token':'new-token','file_name':'new','media_type':'photo','source_url':'https://img.oneme.ru/source.jpg'}
    client = MaxTransport()
    with patch.object(admin_topic_media,'async_session_maker',session_factory):
        await admin_topic_media.save_add_description(client,state,100,100000000007,'Portrait')
    async with session_factory() as session:
        media_id = await session.scalar(select(MediaLibrary.id).where(MediaLibrary.file_name == 'new'))
        media = await resolve_editable_media_source(session,media_id=media_id,topic_id=1,include_source_bytes=True)
        assert media.source_platform == 'max'
        assert media.source_image_bytes == b'original-max-image'
        assert media.file_id == 'new-token'


@pytest.mark.asyncio
async def test_additive_migration_preserves_legacy_media(async_engine, media_setup):
    from database import _migrate_media_edit_sources
    async with async_engine.begin() as connection:
        for column in ('source_platform','source_url','source_image_bytes'):
            await connection.execute(text(f'ALTER TABLE media_library DROP COLUMN {column}'))
        await connection.run_sync(_migrate_media_edit_sources)
        await connection.run_sync(_migrate_media_edit_sources)
        rows = (await connection.execute(text('SELECT id, file_id, source_platform, source_url, source_image_bytes FROM media_library ORDER BY id'))).all()
        assert rows == [(10,'telegram-source',None,None,None),(20,'other-source',None,None,None),(30,'audio',None,None,None),(40,'max-source',None,None,None),(41,'max-token-only',None,None,None),(42,'legacy-max-source',None,None,None)]


@pytest.mark.asyncio
@pytest.mark.parametrize('url',['http://img.oneme.ru/file','https://localhost/file','https://127.0.0.1/file','https://img.oneme.ru.attacker.test/file','https://user:password@img.oneme.ru/file','https://img.oneme.ru:8443/file'])
async def test_max_download_rejects_non_platform_urls(url):
    from max_messenger_bot.api import MaxApiError
    client = MaxApiClient('test','https://max.test')
    with pytest.raises(MaxApiError):
        await client.download_attachment_url(url)


@pytest.mark.asyncio
async def test_edit_sources_preserve_text_history_dedup(session_factory,media_setup):
    from ai_request_builder import build_conversational_request_layout
    async with session_factory() as session:
        session.add(DBMessage(id=701,user_id=media_setup,dialogue_id=1,topic_id=1,role='user',content='Повтор'))
        session.add(DBMessage(id=702,user_id=media_setup,dialogue_id=1,topic_id=1,role='user',content='Повтор'))
        await session.commit()
        user = await session.get(User,media_setup)
        config = await session.get(AIConfig,1)
        layout = await build_conversational_request_layout(session,user=user,ai_config=config,dialogue_id=1,topic_id=1,current_user_content='Повтор',exclude_message_id=702,service_capabilities=MAX_CAPABILITIES,media_edit_sources='[PHOTO id=10] portrait')
        from ai_request_context import build_openai_chat_messages
        messages = build_openai_chat_messages(layout)
        assert [message['content'] for message in messages if message['role'] == 'user'] == ['Повтор']


@pytest.mark.asyncio
@pytest.mark.parametrize('platform,error_kind',[(None,'network'),(None,'invalid_id'),('telegram','invalid_id'),('max','unavailable')])
async def test_unavailable_source_never_guesses_success(platform,error_kind):
    media = SimpleNamespace(source_platform=platform,source_url=None,source_image_bytes=None,file_id='opaque-token')
    error = TelegramBadRequest(method=GetFile(file_id='opaque-token'),message='wrong file identifier') if error_kind == 'invalid_id' else RuntimeError('network unavailable')
    bot = SimpleNamespace(get_file=AsyncMock(side_effect=error))
    client = SimpleNamespace(download_attachment=AsyncMock(side_effect=RuntimeError('MAX cannot resolve token')))
    with pytest.raises(MediaSourceError):
        await load_media_image_bytes(media,telegram_bot=bot,max_client=client)
    assert client.download_attachment.await_count == int(platform == 'max' or (platform is None and error_kind == 'invalid_id'))


@pytest.mark.asyncio
@pytest.mark.parametrize('phase',['edit','upload'])
async def test_scope_revoked_during_media_work(session_factory,media_setup,monkeypatch,phase):
    from sqlalchemy import delete
    client = MaxTransport()
    async def revoke():
        async with session_factory() as session:
            await session.execute(delete(topic_collection_association).where(topic_collection_association.c.topic_id == 1))
            await session.commit()
    async def edit(*args):
        if phase == 'edit':
            await revoke()
        return b'edited'
    async def upload(*args):
        if phase == 'upload':
            await revoke()
        return {'token':'edited-token'}
    monkeypatch.setattr(max_ai,'_dispatch_provider',AsyncMock(return_value='Вот вариант:\n[EDIT_IMG_MEDIA: 40 | blue sky]'))
    monkeypatch.setattr(max_ai,'_edit_kie',edit)
    monkeypatch.setattr(client,'upload_file',upload)
    await common.run_ai_dialogue(client,100,100000000007,'Измени портрет')
    assert not any(item.get('type') == 'image' for _,_,kwargs in client.wire for item in kwargs.get('json_data',{}).get('attachments',[]))
    assert not any('Вот вариант' in (kwargs.get('json_data',{}).get('text') or '') for _,_,kwargs in client.wire)


@pytest.mark.asyncio
async def test_max_error_history_does_not_move_to_new_topic(session_factory,media_setup,monkeypatch):
    client = MaxTransport()
    original_request = client._request
    async def request(method,path,**kwargs):
        if 'Не удалось' in (kwargs.get('json_data',{}).get('text') or ''):
            async with session_factory() as session:
                user = await session.get(User,100000000007)
                user.current_topic_id = 2
                await session.commit()
        return await original_request(method,path,**kwargs)
    monkeypatch.setattr(client,'_request',request)
    monkeypatch.setattr(max_ai,'_dispatch_provider',AsyncMock(return_value='[EDIT_IMG_MEDIA: 40 | blue sky]'))
    monkeypatch.setattr(client,'download_attachment_url',AsyncMock(side_effect=RuntimeError('unavailable')))
    async with session_factory() as session:
        user = await session.get(User,100000000007)
        user.current_topic_id = None
        await session.commit()
    await common.run_ai_dialogue(client,100,100000000007,'Измени портрет')
    async with session_factory() as session:
        assert not (await session.scalars(select(DBMessage).where(DBMessage.user_id == 100000000007,DBMessage.role == 'assistant'))).all()
