"""Conversation automation state, metadata and event persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    AutomationConversationState,
    AutomationEvent,
    AutomationMetadataRecord,
    AutomationStepTransition,
    User,
)
from user_metadata import ServiceDataBlock, append_metadata_records, merge_metadata


@dataclass(frozen=True)
class AutomationApplyResult:
    state_changed: bool
    previous_step: str | None
    current_step: str | None
    event_names: tuple[str, ...]


async def get_automation_summary(session: AsyncSession) -> dict[str, int]:
    """Return the compact algorithm counters used by the general statistics screen."""
    return {
        "users": await session.scalar(
            select(func.count(distinct(AutomationStepTransition.user_id)))
        ) or 0,
        "current_users": await session.scalar(
            select(func.count(distinct(AutomationConversationState.user_id))).where(
                AutomationConversationState.current_step.is_not(None)
            )
        ) or 0,
        "transitions": await session.scalar(
            select(func.count(AutomationStepTransition.id))
        ) or 0,
    }


def _load_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _dump_object(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _step_from_state(state: dict[str, Any]) -> str | None:
    value = state.get("current_step")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


async def get_conversation_automation_state(
    session: AsyncSession,
    *,
    user_id: int,
    dialogue_id: int,
    topic_id: int | None,
) -> AutomationConversationState | None:
    return await session.scalar(
        select(AutomationConversationState).where(
            AutomationConversationState.user_id == user_id,
            AutomationConversationState.dialogue_id == dialogue_id,
            AutomationConversationState.topic_id == (topic_id or 0),
        )
    )


async def build_runtime_automation_context(
    session: AsyncSession,
    *,
    user_id: int,
    dialogue_id: int,
    topic_id: int | None,
) -> str:
    """Return dynamic state as a separate runtime message for the LLM."""
    row = await get_conversation_automation_state(
        session,
        user_id=user_id,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
    )
    if row is None:
        payload = {"current_state": {}, "metadata": {}}
    else:
        payload = {
            "current_state": _load_object(row.current_state_json),
            "metadata": _load_object(row.metadata_json),
        }
    return (
        "СЛУЖЕБНЫЕ ДАННЫЕ ТЕКУЩЕГО ДИАЛОГА. Не показывай их пользователю:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


async def apply_service_data_blocks(
    session: AsyncSession,
    *,
    user: User,
    dialogue_id: int,
    topic_id: int | None,
    blocks: Iterable[ServiceDataBlock],
) -> AutomationApplyResult:
    """Atomically persist a model response in the current dialogue scope."""
    blocks = list(blocks)
    metadata_blocks = [
        {"data": block.metadata, "raw_json": block.raw_json}
        for block in blocks
        if block.metadata
    ]
    if metadata_blocks:
        # Keep the existing admin export/history fully backward compatible.
        user.metadata_json = append_metadata_records(user.metadata_json, metadata_blocks)

    structured_blocks = [block for block in blocks if not block.legacy]
    if not structured_blocks:
        return AutomationApplyResult(False, None, None, ())

    scope_topic_id = topic_id or 0
    scope_dialogue_id = dialogue_id or 1
    row = await get_conversation_automation_state(
        session,
        user_id=user.id,
        dialogue_id=scope_dialogue_id,
        topic_id=scope_topic_id,
    )
    if row is None:
        row = AutomationConversationState(
            user_id=user.id,
            dialogue_id=scope_dialogue_id,
            topic_id=scope_topic_id,
        )
        session.add(row)

    previous_step = row.current_step
    current_state = _load_object(row.current_state_json)
    current_metadata = _load_object(row.metadata_json)
    event_names: list[str] = []

    for block in structured_blocks:
        if block.current_state:
            current_state = merge_metadata(current_state, block.current_state)

        if block.metadata:
            session.add(AutomationMetadataRecord(
                user_id=user.id,
                dialogue_id=scope_dialogue_id,
                topic_id=scope_topic_id,
                save_mode=block.save_mode,
                data_json=_dump_object(block.metadata),
            ))
            if block.save_mode == "merge":
                current_metadata = merge_metadata(current_metadata, block.metadata)

        for event_name in block.events:
            event_names.append(event_name)
            session.add(AutomationEvent(
                user_id=user.id,
                dialogue_id=scope_dialogue_id,
                topic_id=scope_topic_id,
                name=event_name,
                state_json=_dump_object(current_state),
                metadata_json=_dump_object(current_metadata),
            ))

    current_step = _step_from_state(current_state)
    row.current_state_json = _dump_object(current_state)
    row.metadata_json = _dump_object(current_metadata)
    row.current_step = current_step

    state_changed = bool(current_step and current_step != previous_step)
    if state_changed:
        session.add(AutomationStepTransition(
            user_id=user.id,
            dialogue_id=scope_dialogue_id,
            topic_id=scope_topic_id,
            previous_step=previous_step,
            current_step=current_step,
            state_json=row.current_state_json,
        ))

    return AutomationApplyResult(
        state_changed=state_changed,
        previous_step=previous_step,
        current_step=current_step,
        event_names=tuple(event_names),
    )
