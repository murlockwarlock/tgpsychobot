"""Deterministic, idempotent execution of event-handler actions."""

from __future__ import annotations

import json
import logging
import re
import html
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

from database import (
    AutomationAction,
    AutomationActionExecution,
    AutomationCondition,
    AutomationConversationState,
    AutomationEvent,
    AutomationHandler,
    User,
    async_session_maker,
    get_all_admin_ids,
)
from user_metadata import append_metadata_records, merge_metadata


log = logging.getLogger(__name__)
_METADATA_PLACEHOLDER_RE = re.compile(r"\{metadata\.([A-Za-z0-9_.-]+)\}")


def _load_object(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_path(data: dict[str, Any], path: str | None) -> Any:
    current: Any = data
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def condition_matches(condition: AutomationCondition, event: AutomationEvent) -> bool:
    state = _load_object(event.state_json)
    metadata = _load_object(event.metadata_json)
    condition_type = condition.condition_type
    actual: Any
    if condition_type == "event":
        actual = event.name
    elif condition_type == "current_step":
        actual = state.get("current_step")
    elif condition_type == "metadata":
        actual = _resolve_path(metadata, condition.field_path)
    else:
        return False

    expected = condition.expected_value
    operator = condition.operator or "equals"
    if operator == "exists":
        return actual is not None
    if operator == "not_equals":
        return str(actual) != expected
    if operator == "contains":
        if isinstance(actual, (list, tuple, set)):
            return expected in {str(item) for item in actual}
        return expected in str(actual or "")
    return str(actual) == expected


def handler_matches(handler: AutomationHandler, event: AutomationEvent) -> bool:
    if not handler.is_active or not handler.conditions or not handler.actions:
        return False
    if not handler.all_topics:
        if event.topic_id == 0:
            if not handler.include_main_dialogue:
                return False
        elif event.topic_id not in {topic.id for topic in handler.topics}:
            return False
    return all(condition_matches(condition, event) for condition in handler.conditions)


def render_message_template(
    template: str,
    *,
    event: AutomationEvent,
    user: User,
    escape_values: bool = True,
) -> str:
    metadata = _load_object(event.metadata_json)
    state = _load_object(event.state_json)

    def replace_metadata(match: re.Match) -> str:
        value = _resolve_path(metadata, match.group(1))
        if value is None:
            return ""
        return html.escape(str(value)) if escape_values else str(value)

    def clean(value: Any) -> str:
        raw = str(value or "")
        return html.escape(raw) if escape_values else raw

    rendered = _METADATA_PLACEHOLDER_RE.sub(replace_metadata, template or "")
    display_name = clean(user.name or user.first_name)
    replacements = {
        "{event}": clean(event.name),
        "{current_step}": clean(state.get("current_step")),
        "{user_id}": str(user.id),
        "{username}": clean(f"@{user.username}") if user.username else "",
        "{name}": display_name,
        "{user}": display_name,
        "{dialogue_id}": str(event.dialogue_id),
        "{topic_id}": str(event.topic_id),
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered.strip()


def _render_metadata_value(value: Any, *, event: AutomationEvent, user: User) -> Any:
    if isinstance(value, str):
        return render_message_template(value, event=event, user=user, escape_values=False)
    if isinstance(value, list):
        return [_render_metadata_value(item, event=event, user=user) for item in value]
    if isinstance(value, dict):
        return {
            key: _render_metadata_value(item, event=event, user=user)
            for key, item in value.items()
        }
    return value


async def _execute_action(session, bot, event, handler, action, user) -> None:
    if action.action_type == "send_message":
        text = render_message_template(action.message_template or "", event=event, user=user)
        if not text:
            raise ValueError("Пустой шаблон сообщения")
        if action.recipient_type == "all_admins":
            recipients = sorted(await get_all_admin_ids())
        elif action.recipient_type == "selected_user" and action.recipient_user_id:
            recipients = [action.recipient_user_id]
        else:
            raise ValueError("Не выбран получатель сообщения")
        errors = []
        for recipient_id in recipients:
            try:
                await bot.send_message(recipient_id, text)
            except Exception as exc:
                errors.append(f"{recipient_id}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    elif action.action_type == "save_metadata":
        configured = _load_object(action.metadata_json)
        payload = _render_metadata_value(configured, event=event, user=user)
        if not payload:
            payload = _load_object(event.metadata_json)
        state_row = await session.scalar(
            select(AutomationConversationState).where(
                AutomationConversationState.user_id == event.user_id,
                AutomationConversationState.dialogue_id == event.dialogue_id,
                AutomationConversationState.topic_id == event.topic_id,
            )
        )
        if state_row is not None and payload:
            current = _load_object(state_row.metadata_json)
            state_row.metadata_json = json.dumps(
                merge_metadata(current, payload),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            user.metadata_json = append_metadata_records(
                user.metadata_json,
                [{"data": payload, "raw_json": json.dumps(payload, ensure_ascii=False)}],
            )
    else:
        raise ValueError(f"Неизвестный тип действия: {action.action_type}")


async def process_pending_events(bot, *, limit: int = 100, user_id: int | None = None) -> int:
    """Process pending events once; successful actions are never repeated."""
    processed_count = 0
    async with async_session_maker() as session:
        stale_claim_before = datetime.utcnow() - timedelta(minutes=10)
        stmt = (
            select(AutomationEvent)
            .where(
                AutomationEvent.processed_at.is_(None),
                or_(
                    AutomationEvent.processing_started_at.is_(None),
                    AutomationEvent.processing_started_at < stale_claim_before,
                ),
            )
            .order_by(AutomationEvent.created_at.asc(), AutomationEvent.id.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        if user_id is not None:
            stmt = stmt.where(AutomationEvent.user_id == user_id)
        claimed_events = (await session.execute(stmt)).scalars().all()
        claimed_ids = [event.id for event in claimed_events]
        for event in claimed_events:
            event.processing_started_at = datetime.utcnow()
        await session.commit()
        if not claimed_ids:
            return 0

        events = (
            await session.execute(
                select(AutomationEvent)
                .where(AutomationEvent.id.in_(claimed_ids))
                .order_by(AutomationEvent.created_at.asc(), AutomationEvent.id.asc())
            )
        ).scalars().all()
        handlers = (
            await session.execute(
                select(AutomationHandler)
                .where(AutomationHandler.is_active.is_(True))
                .options(
                    selectinload(AutomationHandler.topics),
                    selectinload(AutomationHandler.conditions),
                    selectinload(AutomationHandler.actions),
                )
            )
        ).scalars().all()

        for event in events:
            event_id = event.id
            user = await session.get(User, event.user_id)
            if user is None:
                event.processed_at = datetime.utcnow()
                continue
            failed = False
            for handler in handlers:
                if not handler_matches(handler, event):
                    continue
                for action in handler.actions:
                    handler_id = handler.id
                    action_id = action.id
                    already_done = await session.scalar(
                        select(AutomationActionExecution.id).where(
                            AutomationActionExecution.event_id == event_id,
                            AutomationActionExecution.action_id == action_id,
                        )
                    )
                    if already_done:
                        continue
                    try:
                        await _execute_action(session, bot, event, handler, action, user)
                        session.add(AutomationActionExecution(
                            event_id=event_id,
                            handler_id=handler_id,
                            action_id=action_id,
                        ))
                        await session.commit()
                    except Exception:
                        failed = True
                        await session.rollback()
                        log.exception(
                            "Automation action failed: event=%s handler=%s action=%s",
                            event_id,
                            handler_id,
                            action_id,
                        )
                        break
                if failed:
                    break
            if not failed:
                event.processed_at = datetime.utcnow()
                await session.commit()
                processed_count += 1
            else:
                retry_event = await session.get(AutomationEvent, event_id)
                if retry_event is not None:
                    retry_event.processing_started_at = None
                    await session.commit()
    return processed_count
