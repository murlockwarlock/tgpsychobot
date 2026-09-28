from __future__ import annotations

import json

from sqlalchemy import select, delete

from database import (
    AutomationConversationState,
    AutomationDialogueState,
    CardSpreadState,
    TestSession,
    UserTopicState,
)
from memory_mode import (
    MEMORY_MODE_GLOBAL,
    MEMORY_MODE_TOPIC,
    METADATA_RESET_MODE_PRESERVE,
    get_metadata_reset_mode,
)


def _metadata_json(value: str | None) -> str:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        payload = {}
    return json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=False, separators=(",", ":"))


async def _current_metadata(session, user_id: int, dialogue_id: int, topic_id: int, memory_mode: str) -> str:
    if memory_mode == MEMORY_MODE_GLOBAL:
        row = await session.scalar(
            select(AutomationDialogueState).where(
                AutomationDialogueState.user_id == user_id,
                AutomationDialogueState.dialogue_id == dialogue_id,
            )
        )
        if row is not None:
            return _metadata_json(row.metadata_json)
    row = await session.scalar(
        select(AutomationConversationState).where(
            AutomationConversationState.user_id == user_id,
            AutomationConversationState.dialogue_id == dialogue_id,
            AutomationConversationState.topic_id == topic_id,
        )
    )
    return _metadata_json(row.metadata_json if row is not None else None)


async def _reset_new_scope_state(session, user_id: int, dialogue_id: int, topic_id: int, memory_mode: str, metadata_json: str) -> None:
    conversation = await session.scalar(
        select(AutomationConversationState).where(
            AutomationConversationState.user_id == user_id,
            AutomationConversationState.dialogue_id == dialogue_id,
            AutomationConversationState.topic_id == topic_id,
        )
    )
    if conversation is not None:
        conversation.current_step = None
        conversation.current_state_json = "{}"
        conversation.metadata_json = metadata_json if memory_mode != MEMORY_MODE_GLOBAL else "{}"
    elif metadata_json != "{}" and memory_mode != MEMORY_MODE_GLOBAL:
        session.add(
            AutomationConversationState(
                user_id=user_id,
                dialogue_id=dialogue_id,
                topic_id=topic_id,
                current_state_json="{}",
                metadata_json=metadata_json,
            )
        )

    dialogue = await session.scalar(
        select(AutomationDialogueState).where(
            AutomationDialogueState.user_id == user_id,
            AutomationDialogueState.dialogue_id == dialogue_id,
        )
    )
    if memory_mode == MEMORY_MODE_GLOBAL:
        if dialogue is None:
            session.add(
                AutomationDialogueState(
                    user_id=user_id,
                    dialogue_id=dialogue_id,
                    metadata_json=metadata_json,
                )
            )
        else:
            dialogue.metadata_json = metadata_json
    elif dialogue is not None:
        dialogue.metadata_json = "{}"


async def start_new_dialogue_scope(
    session,
    user,
    topic_id: int | None,
    memory_mode: str,
    metadata_reset_mode: str | None = None,
    *,
    update_topic_state: bool = False,
) -> tuple[int, int]:
    old_dialogue_id = user.current_dialogue_id or 1
    scope_topic_id = topic_id or 0
    metadata_reset_mode = metadata_reset_mode if metadata_reset_mode in {"reset", "preserve"} else "reset"
    metadata_json = "{}"
    if metadata_reset_mode == METADATA_RESET_MODE_PRESERVE:
        metadata_json = await _current_metadata(
            session,
            user.id,
            old_dialogue_id,
            scope_topic_id,
            memory_mode,
        )

    user.current_dialogue_id = old_dialogue_id + 1
    new_dialogue_id = user.current_dialogue_id

    if update_topic_state and memory_mode == MEMORY_MODE_TOPIC:
        state = await session.get(UserTopicState, (user.id, scope_topic_id))
        if state is None:
            session.add(UserTopicState(user_id=user.id, topic_id=scope_topic_id, dialogue_id=new_dialogue_id))
        else:
            state.dialogue_id = new_dialogue_id

    await _reset_new_scope_state(
        session,
        user.id,
        new_dialogue_id,
        scope_topic_id,
        memory_mode,
        metadata_json,
    )
    await session.execute(delete(CardSpreadState).where(CardSpreadState.user_id == user.id))
    await session.execute(delete(TestSession).where(TestSession.user_id == user.id))
    return old_dialogue_id, new_dialogue_id


def configured_metadata_reset_mode(config) -> str:
    return get_metadata_reset_mode(config)
