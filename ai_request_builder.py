"""Unified conversational and isolated AI request builder for Telegram and MAX."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ai_request_context import (
    AIRequestLayout,
    AIRequestMessage,
    neutralize_stable_prompt,
    normalize_request_messages,
)
import automation_engine
from database import (
    AIConfig,
    Message,
    SubscriptionConfig,
    Topic,
    User,
    UserAIActivity,
)
from memory_mode import build_history_scope, get_memory_mode
from prompt_blocks import (
    DEFAULT_SERVICE_PROMPT_TEMPLATE,
    DEFAULT_SHORT_RESPONSE_INSTRUCTION,
    MAX_CAPABILITIES,
    ServiceCapabilities,
    TELEGRAM_CAPABILITIES,
    render_prompt_block,
    render_service_prompt,
)
from result_history import ai_history_role_filter, select_ai_history_messages
from subscription_context import active_subscription_flag

log = logging.getLogger(__name__)




def build_temporal_activity_context(
    minutes_since_last_visit: int,
    minutes_since_last_message: int,
) -> str:
    """Format canonical temporal activity variables block."""
    return (
        "ВРЕМЕННОЙ КОНТЕКСТ:\n"
        f"minutes_since_last_visit: {minutes_since_last_visit}\n"
        f"minutes_since_last_message: {minutes_since_last_message}"
    )


def build_client_runtime_context(
    user: User,
    subscription_config: SubscriptionConfig | None = None,
    include_subscription_status: bool = True,
) -> str:
    """Format canonical client profile information block."""
    user_name = getattr(user, "name", None) or getattr(user, "first_name", None) or "Не указано"
    user_gender = getattr(user, "gender", None) or "Не указан"
    lines = ["ДАННЫЕ КЛИЕНТА:", f"ИМЯ: {user_name}", f"ПОЛ: {user_gender}"]
    if getattr(user, "age", None):
        lines.append(f"ВОЗРАСТ: {user.age}")
    if include_subscription_status:
        state = getattr(user, "_sa_instance_state", None)
        user_sub = state.dict.get("subscription") if state is not None else getattr(user, "subscription", None)
        subscription_flag = active_subscription_flag(subscription_config, user_sub)
        if subscription_flag:
            lines.append(subscription_flag)
    return "\n".join(lines)


async def get_user_ai_activity_gaps(
    session: AsyncSession,
    *,
    user_id: int,
    topic_id: int | None,
    now: datetime | None = None,
) -> tuple[int, int]:
    """Calculate minutes since previous AI activity for topic and global scopes.

    Returns (minutes_since_last_visit, minutes_since_last_message).
    Zero-speculative legacy backfill: returns 0 if no prior activity exists.
    """
    if now is None:
        now = datetime.utcnow()

    topic_scope = f"topic:{topic_id}" if topic_id else "main"
    try:
        res = await session.execute(
            select(UserAIActivity).where(
                UserAIActivity.user_id == user_id,
                UserAIActivity.scope_key.in_(("global", topic_scope)),
            )
        )
        rows = res.scalars().all() if hasattr(res, "scalars") else []
    except (Exception, AssertionError) as exc:
        log.debug("UserAIActivity query bypassed or failed: %s", exc)
        return 0, 0

    scope_map = {row.scope_key: row.last_request_at for row in rows if hasattr(row, "scope_key")}

    visit_at = scope_map.get(topic_scope)
    if visit_at:
        minutes_since_last_visit = max(0, int((now - visit_at).total_seconds() // 60))
    else:
        minutes_since_last_visit = 0

    global_at = scope_map.get("global")
    if global_at:
        minutes_since_last_message = max(0, int((now - global_at).total_seconds() // 60))
    else:
        minutes_since_last_message = 0

    return minutes_since_last_visit, minutes_since_last_message


async def upsert_user_ai_activity(
    session: AsyncSession,
    *,
    user_id: int,
    scope_key: str,
    request_time: datetime,
) -> None:
    """Monotonically update or insert UserAIActivity using an isolated nested savepoint."""
    stmt = (
        update(UserAIActivity)
        .where(
            UserAIActivity.user_id == user_id,
            UserAIActivity.scope_key == scope_key,
            UserAIActivity.last_request_at < request_time,
        )
        .values(last_request_at=request_time, updated_at=request_time)
    )
    res = await session.execute(stmt)
    if getattr(res, "rowcount", 0) > 0:
        return

    if hasattr(session, "scalar"):
        existing = await session.scalar(
            select(UserAIActivity.last_request_at).where(
                UserAIActivity.user_id == user_id,
                UserAIActivity.scope_key == scope_key,
            )
        )
        if existing is not None:
            return

    try:
        if hasattr(session, "begin_nested"):
            async with session.begin_nested():
                session.add(UserAIActivity(
                    user_id=user_id,
                    scope_key=scope_key,
                    last_request_at=request_time,
                    created_at=request_time,
                    updated_at=request_time,
                ))
                if hasattr(session, "flush"):
                    await session.flush()
        else:
            session.add(UserAIActivity(
                user_id=user_id,
                scope_key=scope_key,
                last_request_at=request_time,
                created_at=request_time,
                updated_at=request_time,
            ))
            if hasattr(session, "flush"):
                await session.flush()
    except IntegrityError:
        if hasattr(session, "execute"):
            res_up = await session.execute(
                update(UserAIActivity)
                .where(
                    UserAIActivity.user_id == user_id,
                    UserAIActivity.scope_key == scope_key,
                    UserAIActivity.last_request_at < request_time,
                )
                .values(last_request_at=request_time, updated_at=request_time)
            )
            if getattr(res_up, "rowcount", 0) <= 0:
                if hasattr(session, "scalar"):
                    existing_after = await session.scalar(
                        select(UserAIActivity.last_request_at).where(
                            UserAIActivity.user_id == user_id,
                            UserAIActivity.scope_key == scope_key,
                        )
                    )
                    if existing_after is None or existing_after < request_time:
                        raise
                else:
                    raise



class ActivityTracker:
    """Tracks logical conversational request activity with mark-once semantics."""

    def __init__(
        self,
        session_factory,
        *,
        user_id: int,
        topic_id: int | None,
        request_time: datetime | None = None,
        track_user_activity: bool = True,
    ):
        self.session_factory = session_factory
        self.user_id = user_id
        self.topic_id = topic_id
        self.request_time = request_time or datetime.utcnow()
        self.track_user_activity = track_user_activity
        self._marked = False
        self._lock = asyncio.Lock()

    async def mark_outbound_attempt_once(self) -> None:
        """Mark activity atomically on the first real outbound network attempt."""
        if not self.track_user_activity or self._marked:
            return
        async with self._lock:
            if not self.track_user_activity or self._marked:
                return
            try:
                async with self.session_factory() as session:
                    await upsert_user_ai_activity(
                        session,
                        user_id=self.user_id,
                        scope_key="global",
                        request_time=self.request_time,
                    )
                    topic_scope = f"topic:{self.topic_id}" if self.topic_id else "main"
                    await upsert_user_ai_activity(
                        session,
                        user_id=self.user_id,
                        scope_key=topic_scope,
                        request_time=self.request_time,
                    )
                    if hasattr(session, "commit"):
                        await session.commit()
                self._marked = True
            except Exception as exc:
                self._marked = False
                log.warning("Failed to record outbound user activity for user_id=%s: %s", self.user_id, exc)
                raise



async def load_conversational_ai_history(
    session: AsyncSession,
    *,
    user_id: int,
    dialogue_id: int,
    topic_id: int | None,
    memory_mode: str,
    limit_first: int = 2,
    limit_recent: int = 10,
    exclude_message_id: int | None = None,
) -> tuple[AIRequestMessage, ...]:
    """Load, filter, and turn-group conversational messages deterministically."""
    stmt = select(Message).where(
        build_history_scope(Message, user_id, dialogue_id, topic_id, memory_mode),
        ai_history_role_filter(Message),
    )
    if exclude_message_id is not None:
        stmt = stmt.where(Message.id != exclude_message_id)

    stmt = stmt.options(selectinload(Message.topic)).order_by(Message.timestamp.asc(), Message.id.asc())
    raw_messages = (await session.execute(stmt)).scalars().all()
    if exclude_message_id is not None:
        raw_messages = [m for m in raw_messages if getattr(m, "id", None) != exclude_message_id]

    selected = select_ai_history_messages(raw_messages, limit_first, limit_recent)
    history_items = [
        {"role": item.role, "content": item.content}
        for item in selected
        if item.content
    ]
    return normalize_request_messages(history_items)


async def build_conversational_request_layout(
    session: AsyncSession,
    *,
    user: User,
    ai_config: AIConfig,
    dialogue_id: int,
    topic_id: int | None = None,
    current_user_content: Any = None,
    exclude_message_id: int | None = None,
    stable_system_prompt: str | None = None,
    shared_instructions: Iterable[str] | None = None,
    minutes_since_last_visit: int = 0,
    minutes_since_last_message: int = 0,
    test_context: str = "",
    short_response_instruction: str = "",
    knowledge_context: str = "",
    available_media_text: str = "",
    media_instruction_block: str = "",
    subscription_config: SubscriptionConfig | None = None,
    load_subscription_config: bool = True,
    include_subscription_status: bool = True,
    memory_mode: str | None = None,
    limit_first: int | None = None,
    limit_recent: int | None = None,
    scenario_context: str | None = None,
    history: Iterable[Any] | None = None,
    modality_instructions: Iterable[str] | None = None,
    service_capabilities: ServiceCapabilities | None = None,
) -> AIRequestLayout:
    """Build the single canonical AIRequestLayout for conversational turns (TG & MAX)."""
    effective_memory_mode = memory_mode or get_memory_mode(ai_config)

    if stable_system_prompt is None:
        if topic_id:
            topic = await session.get(Topic, topic_id)
            if topic and topic.system_prompt:
                stable_system_prompt = topic.system_prompt
        if not stable_system_prompt:
            stable_system_prompt = (
                getattr(ai_config, "system_prompt", None)
                or getattr(ai_config, "general_system_prompt", None)
                or "Ты полезный ИИ-помощник."
            )

    if shared_instructions is None:
        shared_prompt_block = (getattr(ai_config, "shared_prompt_block", "") or "").strip()
        service_prompt_template = getattr(ai_config, "service_prompt_block", None) or DEFAULT_SERVICE_PROMPT_TEMPLATE
        service_prompt_block = render_service_prompt(
            service_prompt_template,
            capabilities=service_capabilities,
            available_media_text=available_media_text,
            media_instruction_block=media_instruction_block,
            test_context_injection="",
            short_response_instruction="",
        )
        modality_parts = [part.strip() for part in (modality_instructions or ()) if part and str(part).strip()]
        shared_parts = [part for part in (shared_prompt_block, service_prompt_block, *modality_parts) if part]
        effective_shared_instructions = tuple(shared_parts)
    else:
        effective_shared_instructions = tuple(shared_instructions)

    if subscription_config is None and load_subscription_config and include_subscription_status:
        subscription_config = await session.get(SubscriptionConfig, 1)

    client_context = build_client_runtime_context(
        user,
        subscription_config=subscription_config,
        include_subscription_status=include_subscription_status,
    )
    temporal_context = build_temporal_activity_context(
        minutes_since_last_visit=minutes_since_last_visit,
        minutes_since_last_message=minutes_since_last_message,
    )
    runtime_parts = [client_context, temporal_context]
    if short_response_instruction and short_response_instruction.strip():
        runtime_parts.append(short_response_instruction.strip())
    elif getattr(user, "response_length", "normal") == "short":
        runtime_parts.append(DEFAULT_SHORT_RESPONSE_INSTRUCTION)

    if scenario_context is None:
        scenario_text = await automation_engine.build_runtime_automation_context(
            session,
            user_id=user.id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            memory_mode=effective_memory_mode,
        )
    else:
        scenario_text = scenario_context
    scenario_parts = (scenario_text.strip(),) if scenario_text and scenario_text.strip() else ()

    request_parts = []
    if test_context and test_context.strip():
        request_parts.append(test_context.strip())
    if knowledge_context and knowledge_context.strip():
        request_parts.append("РЕЛЕВАНТНЫЕ ДАННЫЕ ИЗ БАЗЫ ЗНАНИЙ:\n" + knowledge_context.strip())

    first_limit = limit_first if limit_first is not None else (getattr(ai_config, "context_limit_first", 2) or 2)
    recent_limit = limit_recent if limit_recent is not None else (getattr(ai_config, "context_limit_recent", 10) or 10)
    if history is None:
        canonical_history = await load_conversational_ai_history(
            session,
            user_id=user.id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            memory_mode=effective_memory_mode,
            limit_first=first_limit,
            limit_recent=recent_limit,
            exclude_message_id=exclude_message_id,
        )
    else:
        canonical_history = list(history)

    return AIRequestLayout(
        stable_system_prompt=neutralize_stable_prompt(stable_system_prompt),
        shared_instructions=effective_shared_instructions,
        runtime_context=tuple(runtime_parts),
        scenario_context=scenario_parts,
        request_context=tuple(request_parts),
        history=tuple(canonical_history),
        current_user_content=current_user_content,
    )


async def build_isolated_request_layout(
    session: AsyncSession,
    *,
    user: User,
    ai_config: AIConfig,
    system_prompt: str,
    user_prompt: str,
    dialogue_id: int,
    topic_id: int | None = None,
    minutes_since_last_visit: int = 0,
    minutes_since_last_message: int = 0,
    available_media_text: str = "",
    media_instruction_block: str = "",
    memory_mode: str | None = None,
    service_capabilities: ServiceCapabilities | None = None,
) -> AIRequestLayout:
    """Build canonical AIRequestLayout for isolated direct calls (tests, single prompts, etc.)."""
    effective_memory_mode = memory_mode or get_memory_mode(ai_config)

    shared_prompt_block = (getattr(ai_config, "shared_prompt_block", "") or "").strip()
    service_prompt_template = getattr(ai_config, "service_prompt_block", None) or DEFAULT_SERVICE_PROMPT_TEMPLATE
    service_prompt_block = render_service_prompt(
        service_prompt_template,
        capabilities=service_capabilities,
        available_media_text=available_media_text,
        media_instruction_block=media_instruction_block,
        test_context_injection="",
        short_response_instruction="",
    )
    shared_instructions = tuple(part for part in (shared_prompt_block, service_prompt_block) if part)

    client_context = build_client_runtime_context(user, include_subscription_status=False)
    temporal_context = build_temporal_activity_context(
        minutes_since_last_visit=minutes_since_last_visit,
        minutes_since_last_message=minutes_since_last_message,
    )
    runtime_parts = [client_context, temporal_context]
    if getattr(user, "response_length", "normal") == "short":
        runtime_parts.append(DEFAULT_SHORT_RESPONSE_INSTRUCTION)

    scenario_text = await automation_engine.build_runtime_automation_context(
        session,
        user_id=user.id,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
        memory_mode=effective_memory_mode,
    )
    scenario_parts = (scenario_text.strip(),) if scenario_text and scenario_text.strip() else ()

    return AIRequestLayout(
        stable_system_prompt=neutralize_stable_prompt(system_prompt),
        shared_instructions=shared_instructions,
        runtime_context=tuple(runtime_parts),
        scenario_context=scenario_parts,
        request_context=(),
        history=(),
        current_user_content=user_prompt,
    )
