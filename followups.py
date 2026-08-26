"""Topic-scoped inactivity follow-ups with quiet hours and delivery idempotency."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import BaseMiddleware
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from database import (
    AutomationConversationState,
    AutomationEvent,
    FollowupCampaign,
    FollowupDelivery,
    FollowupRun,
    Message as DBMessage,
    User,
    async_session_maker,
)
from automation_events import condition_value_matches, resolve_condition_path
from response_buttons import extract_response_buttons
from user_metadata import extract_service_data, load_metadata


log = logging.getLogger(__name__)


FOLLOWUP_STAGE_MODE_LABELS = {
    "all": "На всех этапах",
    "selected": "На выбранных этапах",
    "all_except": "На всех этапах кроме",
    "not_set": "Этап не задан",
}
FOLLOWUP_STAGE_MODES = tuple(FOLLOWUP_STAGE_MODE_LABELS)
FOLLOWUP_METADATA_OPERATOR_LABELS = {
    "equals": "=",
    "not_equals": "!=",
    "contains": "содержит",
}
_FOLLOWUP_STAGE_MODE_ALIASES = {
    "all_stages": "all",
    "selected_stages": "selected",
    "step_not_set": "not_set",
}


@dataclass(frozen=True)
class FollowupEligibility:
    eligible: bool
    reason: str
    current_step: str | None = None
    stage_matches: bool = True
    metadata_configured: bool = False
    metadata_matches: bool = True
    matched_stop_event: str | None = None


@dataclass(frozen=True)
class FollowupStepSendResult:
    text: str
    history_text: str
    telegram_message_id: int | None


def parse_followup_csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _normalize_stage_mode(value: str | None) -> str:
    mode = str(value or "all").strip().lower()
    return _FOLLOWUP_STAGE_MODE_ALIASES.get(mode, mode)


def _normalize_current_step(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def evaluate_followup_eligibility(
    campaign: FollowupCampaign,
    *,
    current_step: str | None,
    metadata: dict,
    stop_event_names: list[str] | tuple[str, ...] | set[str] = (),
) -> FollowupEligibility:
    configured_stop_events = parse_followup_csv(getattr(campaign, "stop_events", ""))
    observed_stop_events = set(stop_event_names or ())
    matched_stop_event = next(
        (event_name for event_name in configured_stop_events if event_name in observed_stop_events),
        None,
    )

    normalized_step = _normalize_current_step(current_step)
    stage_mode = _normalize_stage_mode(getattr(campaign, "stage_mode", "all"))
    configured_steps = parse_followup_csv(getattr(campaign, "stage_values", ""))
    if stage_mode == "all":
        stage_matches = True
    elif stage_mode == "selected":
        stage_matches = normalized_step in configured_steps
    elif stage_mode == "all_except":
        stage_matches = normalized_step not in configured_steps
    elif stage_mode == "not_set":
        stage_matches = normalized_step is None
    else:
        stage_matches = False

    field_path = str(getattr(campaign, "metadata_field_path", "") or "").strip()
    metadata_configured = bool(field_path)
    expected = getattr(campaign, "metadata_expected_value", None)
    if not metadata_configured:
        metadata_matches = True
    elif expected is None:
        metadata_matches = False
    else:
        actual = resolve_condition_path(metadata or {}, field_path)
        metadata_matches = condition_value_matches(
            actual,
            str(expected),
            getattr(campaign, "metadata_operator", None) or "equals",
        )

    if matched_stop_event is not None:
        return FollowupEligibility(
            eligible=False,
            reason="stop_event_found",
            current_step=normalized_step,
            stage_matches=stage_matches,
            metadata_configured=metadata_configured,
            metadata_matches=metadata_matches,
            matched_stop_event=matched_stop_event,
        )
    if not stage_matches:
        return FollowupEligibility(
            eligible=False,
            reason="stage_not_allowed",
            current_step=normalized_step,
            stage_matches=False,
            metadata_configured=metadata_configured,
            metadata_matches=metadata_matches,
        )
    if not metadata_matches:
        return FollowupEligibility(
            eligible=False,
            reason="metadata_mismatch",
            current_step=normalized_step,
            stage_matches=True,
            metadata_configured=metadata_configured,
            metadata_matches=False,
        )
    return FollowupEligibility(
        eligible=True,
        reason="eligible",
        current_step=normalized_step,
        stage_matches=True,
        metadata_configured=metadata_configured,
        metadata_matches=True,
    )


async def check_campaign_eligibility(
    session,
    campaign: FollowupCampaign,
    *,
    user_id: int,
    dialogue_id: int,
    topic_id: int | None,
) -> FollowupEligibility:
    scope_topic_id = topic_id or 0
    state = await session.scalar(
        select(AutomationConversationState).where(
            AutomationConversationState.user_id == user_id,
            AutomationConversationState.dialogue_id == dialogue_id,
            AutomationConversationState.topic_id == scope_topic_id,
        )
    )
    configured_stop_events = parse_followup_csv(getattr(campaign, "stop_events", ""))
    stop_event_names: tuple[str, ...] = ()
    if configured_stop_events:
        stop_event_name = await session.scalar(
            select(AutomationEvent.name)
            .where(
                AutomationEvent.user_id == user_id,
                AutomationEvent.dialogue_id == dialogue_id,
                AutomationEvent.topic_id == scope_topic_id,
                AutomationEvent.name.in_(configured_stop_events),
            )
            .limit(1)
        )
        if stop_event_name is not None:
            stop_event_names = (stop_event_name,)
    return evaluate_followup_eligibility(
        campaign,
        current_step=state.current_step if state is not None else None,
        metadata=load_metadata(state.metadata_json if state is not None else None),
        stop_event_names=stop_event_names,
    )


def _campaign_matches_scope(campaign: FollowupCampaign, topic_id: int) -> bool:
    if campaign.all_topics:
        return True
    if topic_id == 0:
        return campaign.include_main_dialogue
    return topic_id in {topic.id for topic in campaign.topics}


def _jitter_seconds(campaign: FollowupCampaign) -> int:
    low = max(0, int(campaign.jitter_min_seconds or 0))
    high = max(low, int(campaign.jitter_max_seconds or 0))
    return random.randint(low, high) if high else 0


def _outside_quiet_hours(moment_utc: datetime, campaign: FollowupCampaign) -> datetime:
    """Move a UTC timestamp to the end of local quiet hours when necessary."""
    try:
        tz = ZoneInfo(campaign.timezone or "Europe/Moscow")
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("Europe/Moscow")
    aware_utc = moment_utc.replace(tzinfo=timezone.utc) if moment_utc.tzinfo is None else moment_utc
    local = aware_utc.astimezone(tz)
    minute = local.hour * 60 + local.minute
    start = int(campaign.quiet_start_minute)
    end = int(campaign.quiet_end_minute)
    if start == end:
        return aware_utc.astimezone(timezone.utc).replace(tzinfo=None)
    in_quiet = start <= minute < end if start < end else minute >= start or minute < end
    if not in_quiet:
        return aware_utc.astimezone(timezone.utc).replace(tzinfo=None)

    target_day = local.date()
    if start >= end and minute >= start:
        target_day += timedelta(days=1)
    target_local = datetime.combine(target_day, datetime.min.time(), tzinfo=tz) + timedelta(minutes=end)
    return target_local.astimezone(timezone.utc).replace(tzinfo=None)


def _due_at(activity_at: datetime, campaign: FollowupCampaign, delay_minutes: int) -> datetime:
    raw_due = activity_at + timedelta(
        minutes=max(1, int(delay_minutes)),
        seconds=_jitter_seconds(campaign),
    )
    return _outside_quiet_hours(raw_due, campaign)


async def send_followup_step(
    bot,
    *,
    user: User,
    step,
    dialogue_id: int,
    topic_id: int,
) -> FollowupStepSendResult:
    if step.message_type == "ai":
        from ai_integration import get_ai_response

        instruction = (
            "[Служебная команда системы]: Пользователь замолчал.\n"
            "Сформируй одно догоняющее сообщение после паузы пользователя. "
            "Не упоминай автоматизацию и не добавляй блок DATA.\n\n"
            + (step.ai_instruction or "Мягко верни пользователя к текущему диалогу.")
        )
        text = await get_ai_response(
            user.id,
            instruction,
            user.name or user.first_name or "Не указано",
            user.gender or "Не указан",
            bot=None,
            topic_id_override=None if topic_id == 0 else topic_id,
            dialogue_id_override=dialogue_id,
            persist_service_data=False,
            request_type="followup",
        )
        if not text:
            raise ValueError("AI вернул пустое догоняющее сообщение")
        from handlers import _send_generated_response

        visible_text, _, _ = extract_service_data(text)
        history_text, _ = extract_response_buttons(visible_text)
        await _send_generated_response(bot, user.id, text)
        return FollowupStepSendResult(text, history_text, None)

    if step.message_type != "static":
        raise ValueError("Неизвестный тип догоняющего шага")
    text = (step.message_text or "").strip()
    if not text:
        raise ValueError("Пустой текст догоняющего сообщения")
    sent = await bot.send_message(user.id, text)
    return FollowupStepSendResult(text, text, getattr(sent, "message_id", None))


async def record_user_activity(
    user_id: int,
    *,
    dialogue_id: int,
    topic_id: int | None,
    activity_at: datetime | None = None,
    _allow_conflict_retry: bool = True,
) -> None:
    """Restart matching follow-up chains from the latest real user activity."""
    activity_at = activity_at or datetime.utcnow()
    scope_topic_id = topic_id or 0
    async with async_session_maker() as session:
        campaigns = (
            await session.execute(
                select(FollowupCampaign)
                .where(FollowupCampaign.is_active.is_(True))
                .options(selectinload(FollowupCampaign.topics), selectinload(FollowupCampaign.steps))
            )
        ).scalars().all()
        for campaign in campaigns:
            if not campaign.steps or not _campaign_matches_scope(campaign, scope_topic_id):
                continue
            run = await session.scalar(
                select(FollowupRun).where(
                    FollowupRun.campaign_id == campaign.id,
                    FollowupRun.user_id == user_id,
                    FollowupRun.dialogue_id == dialogue_id,
                    FollowupRun.topic_id == scope_topic_id,
                ).with_for_update()
            )
            eligibility = await check_campaign_eligibility(
                session,
                campaign,
                user_id=user_id,
                dialogue_id=dialogue_id,
                topic_id=scope_topic_id,
            )
            if not eligibility.eligible:
                if run is not None and run.status == "active":
                    run.status = "cancelled"
                continue
            first_step = campaign.steps[0]
            if run is None:
                run = FollowupRun(
                    campaign_id=campaign.id,
                    user_id=user_id,
                    dialogue_id=dialogue_id,
                    topic_id=scope_topic_id,
                    generation=1,
                    next_step_index=0,
                    last_activity_at=activity_at,
                    due_at=_due_at(activity_at, campaign, first_step.delay_minutes),
                    status="active",
                )
                session.add(run)
            else:
                run.generation += 1
                run.next_step_index = 0
                run.last_activity_at = activity_at
                run.due_at = _due_at(activity_at, campaign, first_step.delay_minutes)
                run.status = "active"
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            if not _allow_conflict_retry:
                raise
            await record_user_activity(
                user_id,
                dialogue_id=dialogue_id,
                topic_id=topic_id,
                activity_at=activity_at,
                _allow_conflict_retry=False,
            )


class FollowupActivityMiddleware(BaseMiddleware):
    """Record messages and button clicks without coupling handlers to follow-ups."""

    async def __call__(self, handler, event, data):
        from_user = getattr(event, "from_user", None)
        callback_data = getattr(event, "data", "") or ""
        fsm_context = data.get("state")
        active_state = await fsm_context.get_state() if fsm_context is not None else None
        is_admin_interaction = callback_data.startswith(("admin_", "automation_", "followup_")) or (
            isinstance(active_state, str)
            and active_state.startswith(("AdminStates:", "AutomationAdminStates:"))
        )
        result = await handler(event, data)
        if from_user is not None and not getattr(from_user, "is_bot", False) and not is_admin_interaction:
            try:
                async with async_session_maker() as session:
                    user = await session.get(User, from_user.id)
                    scope = (
                        user.current_dialogue_id,
                        user.current_topic_id,
                    ) if user else None
                if scope:
                    await record_user_activity(
                        from_user.id,
                        dialogue_id=scope[0],
                        topic_id=scope[1],
                    )
            except Exception:
                log.exception("Could not record follow-up activity for user %s", from_user.id)
        return result


async def process_due_followups(bot, *, limit: int = 100) -> int:
    """Deliver due steps and advance each chain exactly once per generation."""
    now = datetime.utcnow()
    delivered = 0
    async with async_session_maker() as session:
        runs = (
            await session.execute(
                select(FollowupRun)
                .where(FollowupRun.status == "active", FollowupRun.due_at <= now)
                .order_by(FollowupRun.due_at.asc(), FollowupRun.id.asc())
                .limit(limit)
                .options(
                    selectinload(FollowupRun.campaign).selectinload(FollowupCampaign.topics),
                    selectinload(FollowupRun.campaign).selectinload(FollowupCampaign.steps),
                )
            )
        ).scalars().all()

        for run in runs:
            campaign = run.campaign
            user = await session.get(User, run.user_id)
            if (
                user is None
                or not campaign.is_active
                or user.current_dialogue_id != run.dialogue_id
                or (user.current_topic_id or 0) != run.topic_id
                or not _campaign_matches_scope(campaign, run.topic_id)
            ):
                run.status = "cancelled"
                continue
            eligibility = await check_campaign_eligibility(
                session,
                campaign,
                user_id=run.user_id,
                dialogue_id=run.dialogue_id,
                topic_id=run.topic_id,
            )
            if not eligibility.eligible:
                run.status = "cancelled"
                continue
            if run.next_step_index >= len(campaign.steps):
                run.status = "completed"
                continue

            step = campaign.steps[run.next_step_index]
            already_sent = await session.scalar(
                select(FollowupDelivery.id).where(
                    FollowupDelivery.run_id == run.id,
                    FollowupDelivery.generation == run.generation,
                    FollowupDelivery.step_id == step.id,
                )
            )
            if already_sent:
                run.next_step_index += 1
                continue

            try:
                send_result = await send_followup_step(
                    bot,
                    user=user,
                    step=step,
                    dialogue_id=run.dialogue_id,
                    topic_id=run.topic_id,
                )
                session.add(DBMessage(
                    user_id=run.user_id,
                    role="assistant",
                    content=send_result.history_text or "Выберите действие:",
                    ai_context_content=send_result.text if step.message_type == "ai" else None,
                    dialogue_id=run.dialogue_id,
                    topic_id=None if run.topic_id == 0 else run.topic_id,
                ))
                session.add(FollowupDelivery(
                    run_id=run.id,
                    step_id=step.id,
                    generation=run.generation,
                    telegram_message_id=send_result.telegram_message_id,
                ))
                run.next_step_index += 1
                if run.next_step_index >= len(campaign.steps):
                    run.status = "completed"
                else:
                    next_step = campaign.steps[run.next_step_index]
                    run.due_at = _due_at(datetime.utcnow(), campaign, next_step.delay_minutes)
                await session.commit()
                delivered += 1
            except Exception:
                log.exception("Follow-up delivery failed: run=%s step=%s", run.id, step.id)
                await session.rollback()
                retry_run = await session.get(FollowupRun, run.id)
                if retry_run is not None:
                    retry_run.due_at = datetime.utcnow() + timedelta(minutes=5)
                    await session.commit()
        await session.commit()
    return delivered
