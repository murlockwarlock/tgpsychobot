"""Persistent test-attempt history and admin-friendly rendering helpers."""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any

from sqlalchemy import select, text

from database import Message, TestAttempt
from response_buttons import extract_response_buttons
from time_helpers import format_msk
from user_metadata import extract_data_blocks


TEST_RESULT_ROLE = "test_result"
CONVERSATION_ROLES = ("user", "assistant")


def conversation_role_filter(message_model=Message):
    return message_model.role.in_(CONVERSATION_ROLES)


def _json_list(raw: str | None) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _json_dict(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


async def save_test_attempt(
    session,
    *,
    user_id: int,
    source_session_created_at: datetime | None,
    completed_at: datetime,
    platform: str | None,
    topic_id: int | None,
    dialogue_id: int | None,
    answers: list[dict[str, Any]],
    report_text: str,
    formula_results: dict[str, Any],
    interpretation_text: str,
) -> TestAttempt:
    attempt = None
    if source_session_created_at is not None:
        attempt = await session.scalar(
            select(TestAttempt).where(
                TestAttempt.user_id == user_id,
                TestAttempt.source_session_created_at == source_session_created_at,
            )
        )
    if attempt is None:
        attempt = TestAttempt(user_id=user_id, source_session_created_at=source_session_created_at)
        session.add(attempt)

    attempt.completed_at = completed_at
    attempt.platform = platform
    attempt.topic_id = topic_id
    attempt.dialogue_id = dialogue_id
    attempt.answers_json = json.dumps(answers, ensure_ascii=False, separators=(",", ":"))
    attempt.report_text = report_text
    attempt.formula_results_json = (
        json.dumps(formula_results, ensure_ascii=False, separators=(",", ":"))
        if formula_results
        else None
    )
    attempt.interpretation_text = interpretation_text
    return attempt


async def attach_secret_answers(session, user_id: int, text: str) -> None:
    attempt = await session.scalar(
        select(TestAttempt)
        .where(TestAttempt.user_id == user_id)
        .order_by(TestAttempt.completed_at.desc(), TestAttempt.id.desc())
        .limit(1)
    )
    if attempt is not None:
        attempt.secret_answers = text


def build_test_history_entry(
    answers: list[dict[str, Any]],
    formula_results: dict[str, Any],
    report_text: str | None = None,
) -> str:
    lines = ["🧪 Результаты теста"]
    if answers:
        lines.extend(["", "📝 Ответы"])
        for fallback_number, answer in enumerate(answers, start=1):
            number = answer.get("question_number") or fallback_number
            question = str(answer.get("question") or "Вопрос без текста").strip()
            value = str(answer.get("answer") or "—").strip()
            lines.extend(["", f"{number}. {question}", f"Ответ: {value}"])
            numeric = answer.get("numeric_value")
            if numeric is not None:
                lines.append(f"Числовое значение: {numeric}")
    elif report_text:
        lines.extend(["", "📝 Сохранённый результат", "", report_text.strip()])

    if formula_results:
        lines.extend(["", "📊 Расчётные показатели"])
        for name, value in formula_results.items():
            rendered = (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else str(value)
            )
            lines.append(f"• {name}: {rendered}")
    return "\n".join(lines).strip()


async def save_test_history_message(
    session,
    *,
    attempt: TestAttempt,
    user_id: int,
    dialogue_id: int,
    topic_id: int | None,
    answers: list[dict[str, Any]],
    formula_results: dict[str, Any],
    report_text: str,
) -> Message:
    await session.flush()
    message = None
    if attempt.id is not None:
        message = await session.scalar(
            select(Message).where(Message.test_attempt_id == attempt.id)
        )
    if message is None:
        message = Message(user_id=user_id, test_attempt_id=attempt.id)
        session.add(message)

    message.dialogue_id = dialogue_id
    message.topic_id = topic_id
    message.role = TEST_RESULT_ROLE
    message.content = build_test_history_entry(answers, formula_results, report_text)
    message.timestamp = attempt.completed_at
    return message


async def backfill_test_history_messages(session) -> int:
    """Add missing history blocks for attempts saved before this feature existed."""
    attempts = (
        await session.execute(
            select(TestAttempt)
            .outerjoin(Message, Message.test_attempt_id == TestAttempt.id)
            .where(Message.id.is_(None))
            .order_by(TestAttempt.id)
        )
    ).scalars().all()

    for attempt in attempts:
        content = build_test_history_entry(
            _json_list(attempt.answers_json),
            _json_dict(attempt.formula_results_json),
            attempt.report_text,
        )
        await session.execute(
            text(
                """
                INSERT INTO messages (
                    user_id, dialogue_id, role, content, timestamp, topic_id, test_attempt_id
                ) VALUES (
                    :user_id, :dialogue_id, :role, :content, :timestamp, :topic_id, :test_attempt_id
                )
                ON CONFLICT (test_attempt_id) DO NOTHING
                """
            ),
            {
                "user_id": attempt.user_id,
                "dialogue_id": attempt.dialogue_id or 1,
                "role": TEST_RESULT_ROLE,
                "content": content,
                "timestamp": attempt.completed_at,
                "topic_id": attempt.topic_id,
                "test_attempt_id": attempt.id,
            },
        )
    return len(attempts)


def attempt_to_dict(attempt: TestAttempt) -> dict[str, Any]:
    return {
        "id": attempt.id,
        "completed_at": attempt.completed_at.isoformat() if attempt.completed_at else None,
        "platform": attempt.platform,
        "topic_id": attempt.topic_id,
        "dialogue_id": attempt.dialogue_id,
        "answers": _json_list(attempt.answers_json),
        "report": attempt.report_text,
        "formula_results": _json_dict(attempt.formula_results_json),
        "interpretation": attempt.interpretation_text,
        "secret_answers": attempt.secret_answers,
    }


def _plain_interpretation(value: str | None) -> str:
    visible, _, _ = extract_data_blocks(value or "")
    visible, _ = extract_response_buttons(visible)
    return visible.strip()


def _escaped_chunks(value: str, max_escaped_size: int = 2400) -> list[str]:
    """Escape text while keeping each chunk safely below Telegram's HTML limit."""
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for char in value or "":
        escaped = html.escape(char)
        if current and current_size + len(escaped) > max_escaped_size:
            chunks.append("".join(current))
            current = []
            current_size = 0
        current.append(escaped)
        current_size += len(escaped)
    if current or not chunks:
        chunks.append("".join(current))
    return chunks


def _short_escaped(value: Any, max_escaped_size: int = 180) -> str:
    chunks = _escaped_chunks(str(value), max_escaped_size)
    return chunks[0] + ("…" if len(chunks) > 1 else "")


def build_test_attempt_pages(
    attempts: list[TestAttempt],
    *,
    client_name: str,
    topic_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    topic_names = topic_names or {}
    pages: list[dict[str, Any]] = []
    total_attempts = len(attempts)

    for attempt_index, attempt in enumerate(attempts):
        completed_label = (
            format_msk(attempt.completed_at, "%d.%m.%Y %H:%M:%S МСК")
            if attempt.completed_at
            else "дата неизвестна"
        )
        platform = {"telegram": "Telegram", "max": "MAX"}.get(
            (attempt.platform or "").lower(),
            attempt.platform or "не указана",
        )
        topic = topic_names.get(attempt.topic_id, "Основной диалог" if attempt.topic_id is None else f"ID {attempt.topic_id}")
        answers = _json_list(attempt.answers_json)
        formulas = _json_dict(attempt.formula_results_json)

        header = (
            f"<b>🧪 Результаты теста</b>\n"
            f"<b>Клиент:</b> {_short_escaped(client_name)}\n"
            f"<b>Пройден:</b> {html.escape(completed_label)}\n"
            f"<b>Платформа:</b> {_short_escaped(platform)}\n"
            f"<b>Тема:</b> {_short_escaped(topic)}\n"
            f"<b>Диалог:</b> {attempt.dialogue_id or '—'}\n"
            f"<b>Попытка:</b> {total_attempts - attempt_index}/{total_attempts}"
        )

        overview = (
            f"{header}\n\n"
            f"<b>📋 Ответов:</b> {len(answers) if answers else 'снимок старого формата'}\n"
            f"<b>📊 Расчётных показателей:</b> {len(formulas) if formulas else 'нет'}"
        )
        pages.append({"attempt_index": attempt_index, "part": "Обзор", "html": overview})

        if formulas:
            formula_blocks: list[str] = []
            for name, value in formulas.items():
                escaped_name = _short_escaped(name, 500)
                serialized_value = (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                value_chunks = _escaped_chunks(serialized_value, 1500)
                for chunk_index, value_chunk in enumerate(value_chunks):
                    continuation = " <i>(продолжение)</i>" if chunk_index else ""
                    formula_blocks.append(f"• <b>{escaped_name}</b>{continuation}: {value_chunk}")

            current_blocks: list[str] = []
            formula_page = 1
            for block in formula_blocks:
                projected = len("\n".join([*current_blocks, block]))
                if current_blocks and projected > 2500:
                    pages.append({
                        "attempt_index": attempt_index,
                        "part": f"Расчётные показатели, часть {formula_page}",
                        "html": f"{header}\n\n<b>📊 Расчётные показатели</b>\n" + "\n".join(current_blocks),
                    })
                    current_blocks = []
                    formula_page += 1
                current_blocks.append(block)
            if current_blocks:
                pages.append({
                    "attempt_index": attempt_index,
                    "part": f"Расчётные показатели, часть {formula_page}",
                    "html": f"{header}\n\n<b>📊 Расчётные показатели</b>\n" + "\n".join(current_blocks),
                })

        if answers:
            answer_blocks: list[tuple[int, str]] = []
            for fallback_number, answer in enumerate(answers, start=1):
                number = answer.get("question_number") or fallback_number
                question_raw = str(answer.get("question") or "Вопрос без текста")
                question_parts = _escaped_chunks(question_raw, 700)
                question = question_parts[0] + ("…" if len(question_parts) > 1 else "")
                answer_chunks = _escaped_chunks(str(answer.get("answer") or "—"), 1500)
                numeric = answer.get("numeric_value")
                numeric_line = f"\n<i>Числовое значение: {html.escape(str(numeric))}</i>" if numeric is not None else ""
                for chunk_index, answer_chunk in enumerate(answer_chunks):
                    continuation = " <i>(продолжение)</i>" if chunk_index else ""
                    block = f"<b>{number}. {question}</b>{continuation}\nОтвет: {answer_chunk}"
                    if chunk_index == len(answer_chunks) - 1:
                        block += numeric_line
                    answer_blocks.append((fallback_number, block))

            current_blocks: list[str] = []
            current_numbers: list[int] = []
            for answer_number, block in answer_blocks:
                projected = len("\n\n".join([*current_blocks, block]))
                if current_blocks and projected > 2500:
                    pages.append({
                        "attempt_index": attempt_index,
                        "part": f"Ответы {min(current_numbers)}–{max(current_numbers)} из {len(answers)}",
                        "html": f"{header}\n\n<b>📝 Ответы</b>\n\n" + "\n\n".join(current_blocks),
                    })
                    current_blocks = []
                    current_numbers = []
                current_blocks.append(block)
                current_numbers.append(answer_number)
            if current_blocks:
                pages.append({
                    "attempt_index": attempt_index,
                    "part": f"Ответы {min(current_numbers)}–{max(current_numbers)} из {len(answers)}",
                    "html": f"{header}\n\n<b>📝 Ответы</b>\n\n" + "\n\n".join(current_blocks),
                })
        elif attempt.report_text:
            for part_index, chunk in enumerate(_escaped_chunks(attempt.report_text), start=1):
                pages.append({
                    "attempt_index": attempt_index,
                    "part": f"Сохранённый результат, часть {part_index}",
                    "html": f"{header}\n\n<b>📝 Сохранённый результат</b>\n\n<code>{chunk}</code>",
                })

        interpretation = _plain_interpretation(attempt.interpretation_text)
        if interpretation:
            chunks = _escaped_chunks(interpretation)
            for part_index, chunk in enumerate(chunks, start=1):
                part = "Итоговая интерпретация" + (f", часть {part_index}/{len(chunks)}" if len(chunks) > 1 else "")
                pages.append({
                    "attempt_index": attempt_index,
                    "part": part,
                    "html": f"{header}\n\n<b>🤖 Итоговая интерпретация</b>\n\n{chunk}",
                })

        if attempt.secret_answers:
            for part_index, chunk in enumerate(_escaped_chunks(attempt.secret_answers), start=1):
                pages.append({
                    "attempt_index": attempt_index,
                    "part": f"Секретный блок, часть {part_index}",
                    "html": f"{header}\n\n<b>🔐 Ответы секретного блока</b>\n\n{chunk}",
                })

    return pages
