import re

from aiogram import BaseMiddleware

from database import SubscriptionPlan, async_session_maker
from translation_service import refresh_translation_cache, resolve_user_effective_locale, translate


def display_value(kind, resource, field, locale):
    identity = resource.key if kind == "content" else resource.id
    source = getattr(resource, field, None) or ""
    return translate(f"{kind}.{identity}.{field}", locale, source=source, fallback=source) or None


def question_available(question, locale):
    from universal_tests import get_answer_options
    if not display_value("test_question", question, "text", locale):
        return False
    for option in get_answer_options(question):
        if not translate(f"test_question.{question.id}.option.{option.translation_slot}.text", locale, fallback=option.text, source=option.text or ""):
            return False
    return True


class ContentRuntimeMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        await refresh_translation_cache(async_session_maker)
        message_text = getattr(event, "text", None)
        state = data.get("state")
        if message_text and (state is None or await state.get_state() is None):
            from content_menu import needs_menu_refresh
            if await needs_menu_refresh(event.from_user.id, message_text, session_maker=async_session_maker):
                from keyboards import main_client_keyboard
                async with async_session_maker() as session:
                    locale = await resolve_user_effective_locale(session, event.from_user.id)
                await event.answer(translate("ui.navigation.menu_hint", locale, fallback="Нажмите на кнопку или воспользуйтесь меню для навигации"), reply_markup=await main_client_keyboard(event.from_user.id))
                return
        callback_data = getattr(event, "data", "") or ""
        match = re.match(r"^(?:sub_pay|pay_yookassa|pay_robokassa|pay_tg)_(\d+)(?:_|$)", callback_data)
        if match:
            async with async_session_maker() as session:
                locale = await resolve_user_effective_locale(session, event.from_user.id)
                plan = await session.get(SubscriptionPlan, int(match[1]))
                if plan is None or not display_value("plan", plan, "name", locale):
                    await event.answer(translate("ui.subscription.plan_not_found", locale, fallback="Тариф не найден."), show_alert=True)
                    return
        return await handler(event, data)
