from contextvars import ContextVar

from aiogram import BaseMiddleware


content_editing_locale = ContextVar("content_editing_locale", default=None)
content_admin_id = ContextVar("content_admin_id", default=None)


class AdminAuthoringMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        from content_authoring import editing_locale
        from database import async_session_maker
        from handlers import is_admin

        state = data.get("state")
        current_state = await state.get_state() if state else None
        callback_data = getattr(event, "data", "") or ""
        callback = getattr(data.get("handler"), "callback", None)
        callback_name = getattr(callback, "__name__", "")
        is_authoring = (
            (current_state or "").startswith(("Admin", "AutomationAdmin"))
            or callback_name.startswith("admin_")
            or callback_data.startswith(("admin_", "automation_", "followup_", "edit_topic_", "edit_content_", "edit_plan_", "create_topic", "mailing_", "ref_tpl_", "add_secret_", "edit_secret_"))
        )
        if not is_authoring or not await is_admin(event.from_user.id):
            return await handler(event, data)
        state_data = await state.get_data() if state else {}
        async with async_session_maker() as session:
            locale = state_data.get("authoring_locale") if current_state else None
            locale = locale or await editing_locale(session, event.bot.id, event.from_user.id)
        token = content_editing_locale.set(locale)
        admin_token = content_admin_id.set(event.from_user.id)
        try:
            result = await handler(event, data)
            if state and (await state.get_state() or "").startswith(("Admin", "AutomationAdmin")):
                await state.update_data(authoring_locale=locale)
            return result
        finally:
            content_editing_locale.reset(token)
            content_admin_id.reset(admin_token)


async def admin_language_request(make_request, bot, method):
    from translation_service import LOCALE_LABELS
    locale = content_editing_locale.get()
    if locale and getattr(method, "chat_id", None) == content_admin_id.get():
        text = getattr(method, "text", None)
        if isinstance(text, str) and "Язык контента:" not in text:
            heading = f"✍️ Язык контента: {LOCALE_LABELS.get(locale, locale)}\n\n"
            if len(heading + text) <= 4096:
                method = method.model_copy(update={"text": heading + text})
    return await make_request(bot, method)
