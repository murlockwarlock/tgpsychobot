from __future__ import annotations

import html
import json
import math
import re
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select

from ..api import MaxApiClient, MaxApiError
from ..identity import is_max_user_id, max_public_name, max_username, raw_max_user_id
from ..keyboards import admin_ai_log_detail_keyboard, admin_ai_logs_keyboard
from ..legacy import AILog, User, async_session_maker
from ..logging_utils import get_bot_logger
from ..time_utils import format_msk
from ai_log_context import ai_log_context_label

log = get_bot_logger("admin_ai_logs")

PER_PAGE = 8
MAX_AI_LOG_DETAIL_TEXT_LIMIT = 3800
MAX_EXPORT_LOGS_LIMIT = 500
MAX_EXPORT_UNCOMPRESSED_BYTES = 20 * 1024 * 1024

VALID_PERIODS = frozenset({"all", "today", "7d", "30d"})
VALID_REQUEST_TYPES = frozenset({"all", "chat", "followup"})

AI_LOG_PERIOD_LABELS = {
    "all": "всё время",
    "today": "сегодня",
    "7d": "7 дней",
    "30d": "30 дней",
}

AI_LOG_TYPE_LABELS = {
    "all": "все типы",
    "chat": "обычные запросы",
    "followup": "догоняющие",
}


def validate_filters(period: str, request_type: str) -> bool:
    return period in VALID_PERIODS and request_type in VALID_REQUEST_TYPES


def _ai_log_platform(log_entry: AILog) -> str | None:
    platform = str(getattr(log_entry, "platform", "") or "").strip().lower()
    if platform in {"telegram", "max"}:
        return platform
    if log_entry.user_id is None:
        return None
    return "max" if is_max_user_id(log_entry.user_id) else "telegram"


def _ai_log_platform_label(log_entry: AILog) -> str:
    return {"telegram": "Telegram", "max": "MAX"}.get(_ai_log_platform(log_entry), "не зафиксирована")


def _ai_log_display_user_id(log_entry: AILog) -> int | None:
    if _ai_log_platform(log_entry) == "max" and log_entry.user_id is not None:
        return raw_max_user_id(log_entry.user_id)
    return log_entry.user_id


def _ai_log_user_filter_label(user_id: int) -> str:
    if is_max_user_id(user_id):
        return f"ID max: {raw_max_user_id(user_id)}"
    return f"ID {user_id}"


def _ai_log_period_start(period: str) -> datetime | None:
    now_utc = datetime.utcnow()
    if period == "today":
        now_msk = now_utc + timedelta(hours=3)
        return now_msk.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=3)
    if period == "7d":
        return now_utc - timedelta(days=7)
    if period == "30d":
        return now_utc - timedelta(days=30)
    return None


def _apply_ai_log_filters(query, *, filter_user_id: int | None, period: str, request_type: str = "all"):
    if filter_user_id:
        query = query.where(AILog.user_id == filter_user_id)
    period_start = _ai_log_period_start(period)
    if period_start is not None:
        query = query.where(AILog.created_at >= period_start)
    if request_type in AI_LOG_TYPE_LABELS and request_type != "all":
        query = query.where(AILog.request_type == request_type)
    return query


def _build_ai_log_file_content(log_entry: AILog) -> str:
    request_preview = log_entry.request_payload or "не зафиксирован"
    return (
        f"========================================\n"
        f"AI LOG RECORD #{log_entry.id}\n"
        f"========================================\n"
        f"Timestamp: {log_entry.created_at}\n"
        f"User ID: {_ai_log_display_user_id(log_entry)}\n"
        f"Request type: {getattr(log_entry, 'request_type', 'chat')}\n"
        f"Platform: {_ai_log_platform_label(log_entry)}\n"
        f"Context: {ai_log_context_label(log_entry)}\n"
        f"Provider: {log_entry.provider}\n"
        f"Model: {log_entry.model}\n"
        f"Latency: {log_entry.latency_ms} ms\n"
        f"========================================\n\n"
        f"📤 [1] FULL REQUEST PAYLOAD:\n"
        f"----------------------------------------\n"
        f"{request_preview}\n\n"
        f"========================================\n"
        f"🤖 [2] RAW RESPONSE FROM LLM:\n"
        f"----------------------------------------\n"
        f"{log_entry.raw_response or ''}\n\n"
        f"========================================\n"
        f"💬 [3] CLEAN TEXT SENT TO USER:\n"
        f"----------------------------------------\n"
        f"{log_entry.clean_text or ''}\n"
    )


def _safe_truncate_escaped(raw_text: str, max_escaped_len: int, truncation_suffix: str = "...") -> str:
    escaped_full = html.escape(raw_text)
    if len(escaped_full) <= max_escaped_len:
        return escaped_full

    budget = max(0, max_escaped_len - len(truncation_suffix))
    if budget == 0:
        return truncation_suffix[:max_escaped_len]

    low = 0
    high = min(len(raw_text), budget)
    best_k = 0
    while low <= high:
        mid = (low + high) // 2
        escaped_prefix = html.escape(raw_text[:mid])
        if len(escaped_prefix) <= budget:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1

    return html.escape(raw_text[:best_k]) + truncation_suffix


def _format_bounded_detail_text(log_entry: AILog, user: User | None) -> str:
    platform = _ai_log_platform(log_entry)

    def _esc_meta(val: str | None, max_len: int = 60, default: str = "не указан") -> str:
        if not val:
            return html.escape(default)
        return _safe_truncate_escaped(str(val), max_len, "…")

    if platform == "max":
        max_id = raw_max_user_id(log_entry.user_id) if log_entry.user_id is not None else "не указан"
        comm_name = _esc_meta(user.name if user else None, 60, "Не указано")
        pub_name = _esc_meta(max_public_name(user) if user else None, 60, "Не указано")
        u_name = _esc_meta(max_username(user) if user else None, 60, "не указан")
        identity_lines = [
            "📱 <b>Платформа:</b> MAX",
            f"🆔 <b>ID max:</b> <code>{max_id}</code>",
            f"💬 <b>Имя для общения:</b> {comm_name}",
            f"👤 <b>Имя в max:</b> {pub_name}",
            f"<b>Username:</b> {u_name}",
        ]
        if user and user.tg_user_id:
            identity_lines.append(f"🔗 <b>Telegram ID:</b> <code>{user.tg_user_id}</code>")
    elif platform == "telegram":
        if user:
            uname = f"@{user.username}" if user.username else (user.first_name or user.name or "")
            user_info = f"<b>{_esc_meta(uname, 60, '')}</b> (ID: {log_entry.user_id})"
        else:
            user_info = f"ID: {log_entry.user_id}"
        comm_name = _esc_meta(user.name if user else None, 60, "Не указано")
        identity_lines = [
            "📱 <b>Платформа:</b> Telegram",
            f"👤 <b>Пользователь:</b> {user_info}",
            f"🆔 <b>Telegram ID:</b> <code>{log_entry.user_id if log_entry.user_id is not None else 'не указан'}</code>",
            f"💬 <b>Имя для общения:</b> {comm_name}",
        ]
    else:
        if user:
            uname = f"@{user.username}" if user.username else (user.first_name or user.name or "")
            user_info = f"<b>{_esc_meta(uname, 60, '')}</b> (ID: {log_entry.user_id})"
        else:
            user_info = f"ID: {log_entry.user_id}"
        identity_lines = [
            "📱 <b>Платформа:</b> не зафиксирована",
            f"👤 <b>Пользователь:</b> {user_info}",
            f"🆔 <b>ID:</b> <code>{log_entry.user_id if log_entry.user_id is not None else 'не указан'}</code>",
        ]
        if user and user.name:
            identity_lines.append(f"💬 <b>Имя для общения:</b> {_esc_meta(user.name, 60, '')}")

    lat_text = f"{log_entry.latency_ms / 1000:.2f} сек" if log_entry.latency_ms else "не измерялось"
    dt_str = format_msk(log_entry.created_at, "%d-%m-%Y %H:%M:%S МСК")
    log_type = getattr(log_entry, "request_type", "chat") or "chat"

    context_str = _esc_meta(ai_log_context_label(log_entry), 80, "—")
    provider_str = _esc_meta(log_entry.provider, 50, "—")
    model_str = _esc_meta(log_entry.model, 60, "—")
    type_str = _esc_meta(AI_LOG_TYPE_LABELS.get(log_type, log_type), 40, log_type)

    header_text = (
        f"📄 <b>Детали лога ИИ #{log_entry.id}</b>\n\n"
        + "\n".join(identity_lines)
        + "\n"
        f"🧾 <b>Тип запроса:</b> {type_str}\n"
        f"📍 <b>Контекст:</b> {context_str}\n"
        f"🤖 <b>Провайдер:</b> <b>{provider_str}</b>\n"
        f"🧠 <b>Модель:</b> <code>{model_str}</code>\n"
        f"⏱ <b>Время ответа:</b> <code>{lat_text}</code>\n"
        f"📅 <b>Дата вызова:</b> {dt_str}\n\n"
    )

    section1_prefix = "📤 <b>Полный payload запроса (превью):</b>\n<code>"
    section1_suffix = "</code>\n\n"
    section2_prefix = "🤖 <b>Сырой ответ модели (Raw LLM Output):</b>\n<code>"
    section2_suffix = "</code>\n\n"
    section3_prefix = "💬 <b>Текст ответа пользователю (Clean Text):</b>\n<code>"
    section3_suffix = "</code>"

    overhead = (
        len(header_text)
        + len(section1_prefix) + len(section1_suffix)
        + len(section2_prefix) + len(section2_suffix)
        + len(section3_prefix) + len(section3_suffix)
    )
    available_budget = max(90, MAX_AI_LOG_DETAIL_TEXT_LIMIT - overhead)

    raw_payload = log_entry.request_payload
    raw_response = log_entry.raw_response or ""
    clean_text = log_entry.clean_text or ""

    if not raw_payload:
        payload_preview = "не зафиксирован"
        payload_cost = len(payload_preview)
    else:
        max_p = min(700, max(30, int(available_budget * 0.25)))
        payload_preview = _safe_truncate_escaped(raw_payload, max_p, "...")
        payload_cost = len(payload_preview)

    rem_after_payload = max(60, available_budget - payload_cost)

    clean_escaped = html.escape(clean_text)
    max_c = min(900, max(30, int(rem_after_payload * 0.35)))
    if len(clean_escaped) <= max_c:
        clean_preview = clean_escaped
    else:
        clean_preview = _safe_truncate_escaped(clean_text, max_c, "...")
    clean_cost = len(clean_preview)

    raw_budget = max(30, rem_after_payload - clean_cost)
    raw_preview = _safe_truncate_escaped(
        raw_response,
        raw_budget,
        "\n\n[...] (Полный сырой файл скачайте по кнопке ниже)",
    )

    final_text = (
        header_text
        + section1_prefix
        + payload_preview
        + section1_suffix
        + section2_prefix
        + raw_preview
        + section2_suffix
        + section3_prefix
        + clean_preview
        + section3_suffix
    )

    if len(final_text) > MAX_AI_LOG_DETAIL_TEXT_LIMIT:
        excess = len(final_text) - MAX_AI_LOG_DETAIL_TEXT_LIMIT
        tighter_raw_budget = max(10, raw_budget - excess)
        raw_preview = _safe_truncate_escaped(
            raw_response,
            tighter_raw_budget,
            "\n\n[...] (Полный сырой файл скачайте по кнопке ниже)",
        )
        final_text = (
            header_text
            + section1_prefix
            + payload_preview
            + section1_suffix
            + section2_prefix
            + raw_preview
            + section2_suffix
            + section3_prefix
            + clean_preview
            + section3_suffix
        )
        if len(final_text) > MAX_AI_LOG_DETAIL_TEXT_LIMIT:
            excess = len(final_text) - MAX_AI_LOG_DETAIL_TEXT_LIMIT
            tighter_clean_budget = max(10, clean_cost - excess)
            clean_preview = _safe_truncate_escaped(clean_text, tighter_clean_budget, "...")
            final_text = (
                header_text
                + section1_prefix
                + payload_preview
                + section1_suffix
                + section2_prefix
                + raw_preview
                + section2_suffix
                + section3_prefix
                + clean_preview
                + section3_suffix
            )

    return final_text


async def show_ai_logs_list(
    client: MaxApiClient,
    chat_id: int,
    page: int = 0,
    filter_user_id: int | None = None,
    period: str = "all",
    request_type: str = "all",
) -> None:
    if period not in VALID_PERIODS or request_type not in VALID_REQUEST_TYPES:
        raise ValueError(f"Invalid AI log filter period='{period}', request_type='{request_type}'")

    async with async_session_maker() as session:
        query = _apply_ai_log_filters(
            select(AILog),
            filter_user_id=filter_user_id,
            period=period,
            request_type=request_type,
        )

        count_query = select(func.count()).select_from(query.subquery())
        total_count = (await session.execute(count_query)).scalar() or 0
        total_pages = math.ceil(total_count / PER_PAGE) if total_count > 0 else 1
        page = max(0, min(page, total_pages - 1))

        stmt = query.order_by(AILog.created_at.desc()).offset(page * PER_PAGE).limit(PER_PAGE)
        logs = (await session.execute(stmt)).scalars().all()

    if not logs:
        text = "📜 <b>Логи вызовов ИИ пусты.</b>"
        markup = admin_ai_logs_keyboard(
            [],
            page=0,
            total_pages=1,
            filter_user_id=filter_user_id,
            period=period,
            request_type=request_type,
        )
        await client.send_message(chat_id=chat_id, text=text, attachments=markup)
        return

    filter_text = f" пользователя <code>{_ai_log_user_filter_label(filter_user_id)}</code>" if filter_user_id else ""
    header = (
        f"📜 <b>Логи вызовов ИИ{filter_text}</b> (Стр. {page + 1}/{total_pages})\n\n"
        f"Период: <b>{AI_LOG_PERIOD_LABELS[period]}</b>\n"
        f"Тип: <b>{AI_LOG_TYPE_LABELS[request_type]}</b>\n"
        f"Всего вызовов: <b>{total_count}</b>\nВыберите запись:"
    )

    markup = admin_ai_logs_keyboard(
        logs,
        page=page,
        total_pages=total_pages,
        filter_user_id=filter_user_id,
        period=period,
        request_type=request_type,
    )
    await client.send_message(chat_id=chat_id, text=header, attachments=markup)


async def show_ai_log_detail(
    client: MaxApiClient,
    chat_id: int,
    log_id: int,
    page: int = 0,
    filter_user_id: int | None = None,
    period: str = "all",
    request_type: str = "all",
) -> None:
    if period not in VALID_PERIODS or request_type not in VALID_REQUEST_TYPES:
        raise ValueError(f"Invalid AI log filter period='{period}', request_type='{request_type}'")

    async with async_session_maker() as session:
        log_entry = await session.get(AILog, log_id)
        if not log_entry:
            await client.send_message(chat_id=chat_id, text="Запись лога не найдена.")
            return

        user = await session.get(User, log_entry.user_id) if log_entry.user_id else None

    text = _format_bounded_detail_text(log_entry, user)
    markup = admin_ai_log_detail_keyboard(
        log_id,
        page=page,
        filter_user_id=filter_user_id,
        period=period,
        request_type=request_type,
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=markup)


async def download_ai_log_file(client: MaxApiClient, chat_id: int, log_id: int) -> None:
    async with async_session_maker() as session:
        log_entry = await session.get(AILog, log_id)
        if not log_entry:
            await client.send_message(chat_id=chat_id, text="Запись лога не найдена.")
            return

    content = _build_ai_log_file_content(log_entry)
    safe_provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_entry.provider or "provider")[:32]
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_entry.model or "model")[:32]
    filename = f"ai_log_{log_id}_{safe_provider}_{safe_model}.txt"
    try:
        await client.send_text_file(
            chat_id=chat_id,
            filename=filename,
            content=content,
            caption=f"📄 Полный сырой лог ИИ #{log_id}",
        )
    except Exception as exc:
        log.exception("Failed to send AI log text file: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"Не удалось отправить файл лога: {html.escape(str(exc))}")


async def export_ai_logs_package(
    client: MaxApiClient,
    chat_id: int,
    filter_user_id: int | None = None,
    period: str = "all",
    request_type: str = "all",
) -> None:
    if period not in VALID_PERIODS or request_type not in VALID_REQUEST_TYPES:
        raise ValueError(f"Invalid AI log filter period='{period}', request_type='{request_type}'")

    async with async_session_maker() as session:
        base_query = _apply_ai_log_filters(
            select(AILog),
            filter_user_id=filter_user_id,
            period=period,
            request_type=request_type,
        )
        count_query = select(func.count()).select_from(base_query.subquery())
        total_count = (await session.execute(count_query)).scalar() or 0
        if total_count == 0:
            await client.send_message(chat_id=chat_id, text="По выбранному фильтру логов нет.")
            return

        stmt = base_query.order_by(AILog.created_at.desc()).limit(MAX_EXPORT_LOGS_LIMIT)
        logs = (await session.execute(stmt)).scalars().all()

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        included_logs = []
        total_txt_bytes = 0
        byte_limit_reached = False
        manifest = []

        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for log_entry in logs:
                safe_provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_entry.provider or "provider")[:32]
                filename = f"ai_log_{log_entry.id}_{safe_provider}.txt"
                content_str = _build_ai_log_file_content(log_entry)
                content_bytes = content_str.encode("utf-8")

                platform = _ai_log_platform(log_entry)
                entry_data = {
                    "id": log_entry.id,
                    "created_at": log_entry.created_at.isoformat() if log_entry.created_at else None,
                    "platform": platform or "unknown",
                    "provider": log_entry.provider,
                    "model": log_entry.model,
                    "latency_ms": log_entry.latency_ms,
                    "file": filename,
                }
                if platform == "max":
                    entry_data["max_id"] = raw_max_user_id(log_entry.user_id) if log_entry.user_id is not None else None
                elif platform == "telegram":
                    entry_data["telegram_id"] = log_entry.user_id

                candidate_manifest = manifest + [entry_data]
                candidate_manifest_bytes = json.dumps(candidate_manifest, ensure_ascii=False, indent=2).encode("utf-8")
                candidate_total_uncompressed = total_txt_bytes + len(content_bytes) + len(candidate_manifest_bytes)

                if candidate_total_uncompressed > MAX_EXPORT_UNCOMPRESSED_BYTES:
                    byte_limit_reached = True
                    break

                archive.writestr(filename, content_bytes)
                total_txt_bytes += len(content_bytes)
                manifest.append(entry_data)
                included_logs.append(log_entry)

            if included_logs:
                manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
                archive.writestr("manifest.json", manifest_bytes)

        if not included_logs:
            if byte_limit_reached:
                await client.send_message(
                    chat_id=chat_id,
                    text="Не удалось сформировать архив: первая же запись лога превышает максимальный лимит объёма экспорта (20 МБ). Вы можете скачать эту запись отдельно через просмотр лога.",
                )
            else:
                await client.send_message(chat_id=chat_id, text="По выбранному фильтру логов нет.")
            return

        upload_res = await client.upload_file("file", tmp_path)
        token = upload_res.get("token") or upload_res.get("fileId") or upload_res.get("id")
        if not token:
            raise MaxApiError(f"MAX upload did not return file token: {upload_res}")

        user_suffix = (
            f"_user_{raw_max_user_id(filter_user_id)}"
            if filter_user_id and is_max_user_id(filter_user_id)
            else f"_user_{filter_user_id}"
            if filter_user_id
            else ""
        )
        zip_filename = f"ai_logs_{period}{user_suffix}_{datetime.utcnow():%Y%m%d_%H%M%S}.zip"

        if total_count > len(included_logs) and byte_limit_reached:
            limit_note = f" (лимит экспорта по объёму данных: {len(included_logs)} из {total_count})"
        elif total_count > len(included_logs):
            limit_note = f" (лимит экспорта по количеству: {len(included_logs)} из {total_count})"
        else:
            limit_note = ""

        caption = f"📦 Логи ИИ: {len(included_logs)} шт.{limit_note}, период — {AI_LOG_PERIOD_LABELS[period]}"

        await client.send_media_attachment(
            chat_id=chat_id,
            media_type="file",
            token=token,
            caption=caption,
        )
    except Exception as exc:
        log.exception("Failed to export AI logs package: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"Не удалось отправить архив логов: {html.escape(str(exc))}")
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
