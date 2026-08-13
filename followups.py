"""Topic-scoped inactivity follow-ups with quiet hours and delivery idempotency."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import BaseMiddleware
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from database import (
    FollowupCampaign,
    FollowupDelivery,
    FollowupRun,
    Message as DBMessage,
    User,
    async_session_maker,
)
from response_buttons import extract_response_buttons
from user_metadata import extract_service_data


log = logging.getLogger(__name__)


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
            # Another update for the same user created the unique run first.
            # Re-read it under a row lock and apply the latest activity once.
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
                        topic_id_override=None if run.topic_id == 0 else run.topic_id,
                        dialogue_id_override=run.dialogue_id,
                        persist_service_data=False,
                        request_type="followup",
                    )
                    if not text:
                        raise ValueError("AI вернул пустое догоняющее сообщение")
                    from handlers import _send_generated_response
                    visible_text, _, _ = extract_service_data(text)
                    history_text, _ = extract_response_buttons(visible_text)
                    await _send_generated_response(bot, run.user_id, text)
                    session.add(DBMessage(
                        user_id=run.user_id,
                        role="assistant",
                        content=history_text or "Выберите действие:",
                        ai_context_content=text,
                        dialogue_id=run.dialogue_id,
                        topic_id=None if run.topic_id == 0 else run.topic_id,
                    ))
                    telegram_message_id = None
                else:
                    text = (step.message_text or "").strip()
                    if not text:
                        raise ValueError("Пустой текст догоняющего сообщения")
                    sent = await bot.send_message(run.user_id, text)
                    telegram_message_id = getattr(sent, "message_id", None)
                    session.add(DBMessage(
                        user_id=run.user_id,
                        role="assistant",
                        content=text,
                        dialogue_id=run.dialogue_id,
                        topic_id=None if run.topic_id == 0 else run.topic_id,
                    ))
                session.add(FollowupDelivery(
                    run_id=run.id,
                    step_id=step.id,
                    generation=run.generation,
                    telegram_message_id=telegram_message_id,
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
