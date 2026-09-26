from contextvars import ContextVar

from aiogram import BaseMiddleware


content_editing_locale = ContextVar("content_editing_locale", default=None)
content_admin_id = ContextVar("content_admin_id", default=None)


class AdminAuthoringMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        state = data.get("state")
        current_state = await state.get_state() if state else None
        state_data = await state.get_data() if state else {}
        locale = state_data.get("authoring_locale") if current_state else None
        if not locale:
            return await handler(event, data)
        token = content_editing_locale.set(locale)
        admin_token = content_admin_id.set(event.from_user.id)
        try:
            result = await handler(event, data)
            return result
        finally:
            content_editing_locale.reset(token)
            content_admin_id.reset(admin_token)


async def admin_language_request(make_request, bot, method):
    return await make_request(bot, method)
