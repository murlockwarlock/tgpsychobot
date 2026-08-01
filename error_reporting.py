import html
import logging
import traceback
from typing import Any, Sequence

from aiogram import Bot

from database import get_all_admin_ids


def _user_ref(user_id: int | None, username: str | None = None, full_name: str | None = None) -> str:
    if user_id is None:
        return "неизвестно"
    link = f"<a href='tg://user?id={user_id}'>перейти в профиль</a>"
    if username:
        return f"@{html.escape(username)} ({link})"
    name = html.escape(full_name) if full_name else str(user_id)
    return f"{name} ({link})"


def _shorten(value: str, limit: int = 1400) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]} ... [truncated]"


def exception_summary(exception: Exception) -> str:
    message = str(exception).strip()
    return message or type(exception).__name__


async def notify_admins_about_error(
    bot: Bot,
    *,
    title: str,
    user_id: int | None = None,
    username: str | None = None,
    full_name: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    stage: str | None = None,
    details: str | None = None,
    extra: dict[str, Any] | None = None,
    provider_attempts: Sequence[dict[str, str | None]] | None = None,
    exception: Exception | None = None,
    include_traceback: bool = True,
    logger: logging.Logger | None = None,
    level: int = logging.ERROR,
) -> None:
    log = logger or logging.getLogger(__name__)
    trace = traceback.format_exc() if exception is not None and include_traceback else ""
    effective_details = details.strip() if details and details.strip() else None
    if not effective_details and exception is not None and not provider_attempts:
        effective_details = exception_summary(exception)

    log_lines = [title]
    if provider:
        log_lines.append(f"provider={provider}")
    if model:
        log_lines.append(f"model={model}")
    if stage:
        log_lines.append(f"stage={stage}")
    if effective_details:
        log_lines.append(f"details={effective_details}")
    if provider_attempts:
        attempts_log = "; ".join(
            f"{attempt.get('provider') or 'unknown'}={attempt.get('error') or 'unknown error'}"
            for attempt in provider_attempts
        )
        log_lines.append(f"attempts={attempts_log}")
    if extra:
        log_lines.append(f"extra={extra}")

    if exception is not None:
        log.log(level, " | ".join(log_lines), exc_info=exception)
    else:
        log.log(level, " | ".join(log_lines))

    message_lines = [f"⚠️ <b>{html.escape(title)}</b>"]
    if user_id is not None:
        message_lines.append(f"Пользователь: {_user_ref(user_id, username, full_name)}")
    if provider:
        message_lines.append(f"Провайдер: <code>{html.escape(provider)}</code>")
    if model:
        message_lines.append(f"Модель: <code>{html.escape(model)}</code>")
    if stage:
        message_lines.append(f"Этап: <code>{html.escape(stage)}</code>")
    if effective_details:
        message_lines.append(f"Ошибка: <code>{html.escape(_shorten(effective_details, 1800))}</code>")
    if provider_attempts:
        message_lines.append("Попытки провайдеров:")
        for index, attempt in enumerate(provider_attempts, start=1):
            attempt_provider = html.escape(attempt.get("provider") or "неизвестно")
            attempt_model = attempt.get("model")
            model_suffix = f" (<code>{html.escape(attempt_model)}</code>)" if attempt_model else ""
            attempt_error = html.escape(_shorten(attempt.get("error") or "неизвестная ошибка", 900))
            message_lines.append(
                f"{index}. <b>{attempt_provider}</b>{model_suffix}\n"
                f"Ошибка: <code>{attempt_error}</code>"
            )
    if extra:
        safe_extra = _shorten("\n".join(f"{k}={v}" for k, v in extra.items()), 1200)
        message_lines.append(f"Контекст: <code>{html.escape(safe_extra)}</code>")
    if trace and trace.strip() and trace.strip() != "NoneType: None":
        message_lines.append(f"Traceback: <code>{html.escape(_shorten(trace, 1800))}</code>")

    admin_text = "\n".join(message_lines)
    admin_ids = await get_all_admin_ids()
    for admin_id in admin_ids:
        try:
            await bot.send_message(admin_id, admin_text, parse_mode="HTML")
        except Exception as send_exc:
            log.error("Failed to deliver admin error notification admin_id=%s error=%s", admin_id, send_exc)
