"""Topic-scoped inactivity follow-ups with quiet hours and delivery idempotency."""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import BaseMiddleware
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import selectinload

from database import (
    AutomationConversationState,
    AutomationEvent,
    FollowupCampaign,
    FollowupDelivery,
    FollowupDeliveryAttempt,
    FollowupRun,
    FollowupStep,
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
    "all_explicit": "all",
    "all_except_explicit": "all_except",
    "all_legacy": "all",
    "all_except_legacy": "all_except",
}
_FOLLOWUP_LEGACY_STAGE_MODES = {
    "all_stages",
    "all_legacy",
    "all_except_legacy",
}
_FOLLOWUP_LEGACY_UNSET_MODES = {
    "not_set",
    "step_not_set",
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


class FollowupPreparationError(Exception):
    pass


class FollowupStepConfigurationError(FollowupPreparationError):
    pass


class FollowupStepExecutionError(FollowupPreparationError):
    pass


@dataclass(frozen=True)
class FollowupActivityIngress:
    dialogue_id: int
    topic_id: int
    activity_at: datetime
    run_ids: dict[int, int]
    run_generations: dict[int, int]


@dataclass(frozen=True)
class FollowupDeliveryClaim:
    attempt_id: int
    claim_token: str
    run_id: int
    campaign_id: int
    generation: int
    step_index: int
    step_id: int
    dialogue_id: int
    topic_id: int
    user: User
    campaign: FollowupCampaign
    step: object


FOLLOWUP_ATTEMPT_CLAIMED = "claimed"
FOLLOWUP_ATTEMPT_RETRYABLE = "retryable"
FOLLOWUP_ATTEMPT_CANCELLED = "cancelled"
FOLLOWUP_ATTEMPT_DELIVERED = "delivered"
FOLLOWUP_ATTEMPT_UNCERTAIN = "uncertain"
FOLLOWUP_ATTEMPT_RETRY_EXHAUSTED = "retry_exhausted"
FOLLOWUP_RUN_UNCERTAIN = "uncertain"
FOLLOWUP_RUN_RETRY_EXHAUSTED = "retry_exhausted"
FOLLOWUP_ATTEMPT_STALE_AFTER = timedelta(minutes=15)
FOLLOWUP_PREPARATION_RETRY_DELAY = timedelta(minutes=5)
FOLLOWUP_PREPARATION_RETRY_MAX_ATTEMPTS = 3


def parse_followup_csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _normalize_stage_mode(value: str | None) -> str:
    mode = str(value or "all").strip().lower()
    return _FOLLOWUP_STAGE_MODE_ALIASES.get(mode, mode)


def _stage_include_unset(campaign: FollowupCampaign, *, mode: str | None = None) -> bool:
    raw_mode = str(getattr(campaign, "stage_mode", "all") or "all").strip().lower()
    if raw_mode in _FOLLOWUP_LEGACY_STAGE_MODES:
        return True
    if raw_mode in {"all_explicit", "all_except_explicit"}:
        return False
    value = getattr(campaign, "stage_include_unset", None)
    if value is not None:
        return bool(value)
    normalized_mode = _normalize_stage_mode(mode or raw_mode)
    return normalized_mode in {"all", "all_except"}


def _stage_mode_for_storage(mode: str, include_unset: bool) -> str:
    if mode not in FOLLOWUP_STAGE_MODES:
        raise ValueError(f"Unknown follow-up stage mode: {mode}")
    if include_unset or mode == "selected":
        return mode
    return f"{mode}_explicit"


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
    raw_stage_mode = str(getattr(campaign, "stage_mode", "all") or "all").strip().lower()
    stage_mode = _normalize_stage_mode(raw_stage_mode)
    configured_steps = parse_followup_csv(getattr(campaign, "stage_values", ""))
    if normalized_step is None:
        if raw_stage_mode in _FOLLOWUP_LEGACY_STAGE_MODES or raw_stage_mode in _FOLLOWUP_LEGACY_UNSET_MODES:
            stage_matches = True
        elif stage_mode in FOLLOWUP_STAGE_MODES:
            stage_matches = _stage_include_unset(campaign, mode=stage_mode)
        else:
            stage_matches = False
    elif stage_mode == "all":
        stage_matches = True
    elif stage_mode == "selected":
        stage_matches = normalized_step in configured_steps
    elif stage_mode == "all_except":
        stage_matches = normalized_step not in configured_steps
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


def _followup_ai_instruction(step) -> str:
    return (
        "[Служебная команда системы]: Пользователь замолчал.\n"
        "Сформируй одно догоняющее сообщение после паузы пользователя. "
        "Не упоминай автоматизацию и не добавляй блок DATA.\n\n"
        + (getattr(step, "ai_instruction", None) or "Мягко верни пользователя к текущему диалогу.")
    )


def validate_followup_step(step) -> None:
    if step is None:
        raise FollowupStepConfigurationError("Следующий шаг отсутствует")
    message_type = getattr(step, "message_type", None)
    if message_type not in {"static", "ai"}:
        raise FollowupStepConfigurationError("Неизвестный тип догоняющего шага")
    if hasattr(step, "delay_minutes"):
        delay_minutes = getattr(step, "delay_minutes")
        if isinstance(delay_minutes, bool) or not isinstance(delay_minutes, int) or not 1 <= delay_minutes <= 525600:
            raise FollowupStepConfigurationError("Некорректная задержка догоняющего шага")
    if message_type == "static":
        message_text = getattr(step, "message_text", None)
        if not isinstance(message_text, str) or not message_text.strip():
            raise FollowupStepConfigurationError("Пустой текст догоняющего сообщения")
    else:
        ai_instruction = getattr(step, "ai_instruction", None)
        if ai_instruction is not None and not isinstance(ai_instruction, str):
            raise FollowupStepConfigurationError("Некорректная инструкция AI")


async def prepare_followup_step(
    *,
    user: User,
    step,
    dialogue_id: int,
    topic_id: int,
) -> FollowupStepSendResult:
    validate_followup_step(step)
    if step.message_type == "ai":
        try:
            from ai_integration import get_ai_response

            text = await get_ai_response(
                user.id,
                _followup_ai_instruction(step),
                user.name or user.first_name or "Не указано",
                user.gender or "Не указан",
                bot=None,
                topic_id_override=None if topic_id == 0 else topic_id,
                dialogue_id_override=dialogue_id,
                persist_service_data=False,
                request_type="followup",
            )
            if not isinstance(text, str) or not text.strip():
                raise FollowupStepExecutionError("AI вернул пустое догоняющее сообщение")
            visible_text, _, _ = extract_service_data(text)
            history_text, _ = extract_response_buttons(visible_text)
            if not visible_text or not visible_text.strip():
                raise FollowupStepExecutionError("AI не подготовил видимое догоняющее сообщение")
            return FollowupStepSendResult(text, history_text, None)
        except FollowupStepExecutionError:
            raise
        except Exception as exc:
            raise FollowupStepExecutionError("Не удалось подготовить AI-догоняющее сообщение") from exc

    text = step.message_text.strip()
    return FollowupStepSendResult(text, text, None)


async def emit_followup_step(
    bot,
    *,
    user: User,
    step,
    send_result: FollowupStepSendResult,
) -> FollowupStepSendResult:
    validate_followup_step(step)
    if step.message_type == "ai":
        from handlers import _send_generated_response

        await _send_generated_response(bot, user.id, send_result.text)
        return send_result
    sent = await bot.send_message(user.id, send_result.text)
    return FollowupStepSendResult(
        send_result.text,
        send_result.history_text,
        getattr(sent, "message_id", None),
    )


async def send_followup_step(
    bot,
    *,
    user: User,
    step,
    dialogue_id: int,
    topic_id: int,
) -> FollowupStepSendResult:
    send_result = await prepare_followup_step(
        user=user,
        step=step,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
    )
    return await emit_followup_step(
        bot,
        user=user,
        step=step,
        send_result=send_result,
    )


def _scope(dialogue_id: int, topic_id: int | None) -> tuple[int, int]:
    return dialogue_id if dialogue_id is not None else 1, topic_id or 0


async def begin_user_activity(
    user_id: int,
    *,
    dialogue_id: int,
    topic_id: int | None,
    activity_at: datetime | None = None,
) -> FollowupActivityIngress:
    activity_at = activity_at or datetime.utcnow()
    scope_dialogue_id, scope_topic_id = _scope(dialogue_id, topic_id)
    run_ids: dict[int, int] = {}
    run_generations: dict[int, int] = {}
    async with async_session_maker() as session:
        runs = (
            await session.execute(
                select(FollowupRun)
                .where(
                    FollowupRun.user_id == user_id,
                    FollowupRun.dialogue_id == scope_dialogue_id,
                    FollowupRun.topic_id == scope_topic_id,
                )
                .with_for_update()
            )
        ).scalars().all()
        for run in runs:
            run.generation = (run.generation or 1) + 1
            run.next_step_index = 0
            run.last_activity_at = activity_at
            run.due_at = activity_at
            run.status = "pending"
            run_ids[run.campaign_id] = run.id
            run_generations[run.campaign_id] = run.generation
        await session.commit()
    return FollowupActivityIngress(
        dialogue_id=scope_dialogue_id,
        topic_id=scope_topic_id,
        activity_at=activity_at,
        run_ids=run_ids,
        run_generations=run_generations,
    )


def _reset_run_schedule(run: FollowupRun, campaign: FollowupCampaign, activity_at: datetime) -> None:
    first_step = campaign.steps[0]
    run.next_step_index = 0
    run.last_activity_at = activity_at
    run.due_at = _due_at(activity_at, campaign, first_step.delay_minutes)
    run.status = "active"


async def finalize_user_activity(
    user_id: int,
    ingress: FollowupActivityIngress,
    *,
    dialogue_id: int,
    topic_id: int | None,
) -> None:
    final_dialogue_id, final_topic_id = _scope(dialogue_id, topic_id)
    same_scope = (
        final_dialogue_id == ingress.dialogue_id
        and final_topic_id == ingress.topic_id
    )
    async with async_session_maker() as session:
        campaigns = (
            await session.execute(
                select(FollowupCampaign)
                .options(selectinload(FollowupCampaign.topics), selectinload(FollowupCampaign.steps))
            )
        ).scalars().all()
        runs = (
            await session.execute(
                select(FollowupRun)
                .where(FollowupRun.user_id == user_id)
                .with_for_update()
            )
        ).scalars().all()
        runs_by_id = {run.id: run for run in runs}
        runs_by_campaign_scope = {
            (run.campaign_id, run.dialogue_id, run.topic_id): run
            for run in runs
        }

        scheduled_campaign_ids: set[int] = set()
        for campaign in campaigns:
            if not campaign.steps or not _campaign_matches_scope(campaign, final_topic_id):
                continue
            if not campaign.is_active:
                continue
            run = runs_by_campaign_scope.get(
                (campaign.id, final_dialogue_id, final_topic_id)
            )
            eligibility = await check_campaign_eligibility(
                session,
                campaign,
                user_id=user_id,
                dialogue_id=final_dialogue_id,
                topic_id=final_topic_id,
            )
            if not eligibility.eligible:
                if run is not None and (
                    (
                        same_scope
                        and ingress.run_generations.get(campaign.id) == run.generation
                    )
                    or (
                        not same_scope
                        and (
                            run.last_activity_at is None
                            or run.last_activity_at < ingress.activity_at
                        )
                    )
                ):
                    run.status = "cancelled"
                scheduled_campaign_ids.add(campaign.id)
                continue
            expected_generation = (
                ingress.run_generations.get(campaign.id)
                if same_scope
                else None
            )
            if expected_generation is not None:
                if run is None or run.generation != expected_generation or run.status != "pending":
                    continue
            elif run is not None:
                if (
                    run.last_activity_at is not None
                    and run.last_activity_at >= ingress.activity_at
                ):
                    continue
                run.generation = (run.generation or 1) + 1
            else:
                run = FollowupRun(
                    campaign_id=campaign.id,
                    user_id=user_id,
                    dialogue_id=final_dialogue_id,
                    topic_id=final_topic_id,
                    generation=1,
                    next_step_index=0,
                    last_activity_at=ingress.activity_at,
                    due_at=ingress.activity_at,
                    status="pending",
                )
                session.add(run)
                runs_by_campaign_scope[(campaign.id, final_dialogue_id, final_topic_id)] = run
            _reset_run_schedule(run, campaign, ingress.activity_at)
            scheduled_campaign_ids.add(campaign.id)

        for campaign_id, expected_generation in ingress.run_generations.items():
            run_id = ingress.run_ids[campaign_id]
            run = runs_by_id.get(run_id)
            if run is None or run.generation != expected_generation or run.status != "pending":
                continue
            if not same_scope or campaign_id not in scheduled_campaign_ids:
                run.status = "cancelled"
        await session.commit()


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
    try:
        ingress = await begin_user_activity(
            user_id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            activity_at=activity_at,
        )
        await finalize_user_activity(
            user_id,
            ingress,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
        )
    except IntegrityError:
        if not _allow_conflict_retry:
            raise
        await record_user_activity(
            user_id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            activity_at=activity_at,
            _allow_conflict_retry=False,
        )


async def _user_scope_for_activity(user_id: int) -> tuple[int, int] | None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        if user is None:
            return None
        return _scope(user.current_dialogue_id, user.current_topic_id)


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
        ingress = None
        if from_user is not None and not getattr(from_user, "is_bot", False) and not is_admin_interaction:
            try:
                scope = await _user_scope_for_activity(from_user.id)
                if scope is not None:
                    ingress = await begin_user_activity(
                        from_user.id,
                        dialogue_id=scope[0],
                        topic_id=scope[1],
                    )
            except Exception:
                log.exception("Could not invalidate follow-up activity for user %s", from_user.id)
        try:
            return await handler(event, data)
        finally:
            if ingress is not None:
                try:
                    scope = await _user_scope_for_activity(from_user.id)
                    if scope is not None:
                        await finalize_user_activity(
                            from_user.id,
                            ingress,
                            dialogue_id=scope[0],
                            topic_id=scope[1],
                        )
                except Exception:
                    log.exception("Could not finalize follow-up activity for user %s", from_user.id)


async def _due_followup_ids(now: datetime, limit: int) -> list[int]:
    async with async_session_maker() as session:
        uncertain_attempt = select(FollowupDeliveryAttempt.id).where(
            FollowupDeliveryAttempt.run_id == FollowupRun.id,
            FollowupDeliveryAttempt.generation == FollowupRun.generation,
            FollowupDeliveryAttempt.step_index == FollowupRun.next_step_index,
            FollowupDeliveryAttempt.status == FOLLOWUP_ATTEMPT_UNCERTAIN,
        ).exists()
        return list(
            (
                await session.scalars(
                    select(FollowupRun.id)
                    .where(
                        FollowupRun.status == "active",
                        FollowupRun.due_at <= now,
                        ~uncertain_attempt,
                    )
                    .order_by(FollowupRun.due_at.asc(), FollowupRun.id.asc())
                    .limit(limit)
                )
            ).all()
        )


def _claim_snapshot(
    attempt: FollowupDeliveryAttempt,
    run: FollowupRun,
    user: User,
    campaign: FollowupCampaign,
    step,
) -> FollowupDeliveryClaim:
    return FollowupDeliveryClaim(
        attempt_id=attempt.id,
        claim_token=attempt.claim_token,
        run_id=run.id,
        campaign_id=run.campaign_id,
        generation=run.generation,
        step_index=run.next_step_index,
        step_id=step.id,
        dialogue_id=run.dialogue_id,
        topic_id=run.topic_id,
        user=user,
        campaign=campaign,
        step=step,
    )


def _advance_run_values(run: FollowupRun, campaign: FollowupCampaign) -> dict:
    next_step_index = run.next_step_index + 1
    values = {
        "next_step_index": next_step_index,
        "updated_at": datetime.utcnow(),
    }
    if next_step_index >= len(campaign.steps):
        values["status"] = "completed"
    else:
        next_step = campaign.steps[next_step_index]
        values["due_at"] = _due_at(datetime.utcnow(), campaign, next_step.delay_minutes)
        values["status"] = "active"
    return values


async def _claim_due_followup(run_id: int, now: datetime) -> FollowupDeliveryClaim | None:
    async with async_session_maker() as session:
        run = await session.scalar(
            select(FollowupRun)
            .where(
                FollowupRun.id == run_id,
                FollowupRun.status == "active",
                FollowupRun.due_at <= now,
            )
            .options(
                selectinload(FollowupRun.campaign).selectinload(FollowupCampaign.topics),
                selectinload(FollowupRun.campaign).selectinload(FollowupCampaign.steps),
            )
            .with_for_update()
        )
        if run is None:
            return None
        campaign = run.campaign
        user = await session.get(User, run.user_id)
        if (
            user is None
            or not campaign.is_active
            or user.current_dialogue_id != run.dialogue_id
            or (user.current_topic_id or 0) != run.topic_id
            or not _campaign_matches_scope(campaign, run.topic_id)
        ):
            await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == run.id,
                    FollowupRun.generation == run.generation,
                    FollowupRun.next_step_index == run.next_step_index,
                    FollowupRun.status == "active",
                )
                .values(status="cancelled", updated_at=datetime.utcnow())
            )
            await session.commit()
            return None
        eligibility = await check_campaign_eligibility(
            session,
            campaign,
            user_id=run.user_id,
            dialogue_id=run.dialogue_id,
            topic_id=run.topic_id,
        )
        if not eligibility.eligible:
            await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == run.id,
                    FollowupRun.generation == run.generation,
                    FollowupRun.next_step_index == run.next_step_index,
                    FollowupRun.status == "active",
                )
                .values(status="cancelled", updated_at=datetime.utcnow())
            )
            await session.commit()
            return None
        if run.next_step_index >= len(campaign.steps):
            await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == run.id,
                    FollowupRun.generation == run.generation,
                    FollowupRun.next_step_index == run.next_step_index,
                    FollowupRun.status == "active",
                )
                .values(status="completed", updated_at=datetime.utcnow())
            )
            await session.commit()
            return None

        step = campaign.steps[run.next_step_index]
        step = await session.scalar(
            select(FollowupStep)
            .where(
                FollowupStep.id == step.id,
                FollowupStep.campaign_id == run.campaign_id,
            )
            .with_for_update()
        )
        if step is None:
            await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == run.id,
                    FollowupRun.generation == run.generation,
                    FollowupRun.next_step_index == run.next_step_index,
                    FollowupRun.status == "active",
                )
                .values(status="cancelled", updated_at=datetime.utcnow())
            )
            await session.commit()
            return None
        already_sent = await session.scalar(
            select(FollowupDelivery.id).where(
                FollowupDelivery.run_id == run.id,
                FollowupDelivery.generation == run.generation,
                FollowupDelivery.step_id == step.id,
            )
        )
        if already_sent:
            await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == run.id,
                    FollowupRun.generation == run.generation,
                    FollowupRun.next_step_index == run.next_step_index,
                    FollowupRun.status == "active",
                )
                .values(_advance_run_values(run, campaign))
            )
            await session.commit()
            return None

        attempt = await session.scalar(
            select(FollowupDeliveryAttempt)
            .where(
                FollowupDeliveryAttempt.run_id == run.id,
                FollowupDeliveryAttempt.generation == run.generation,
                FollowupDeliveryAttempt.step_index == run.next_step_index,
            )
            .with_for_update()
        )
        if attempt is not None:
            if attempt.status == FOLLOWUP_ATTEMPT_RETRYABLE and attempt.step_id != step.id:
                attempt.status = FOLLOWUP_ATTEMPT_CANCELLED
                attempt.finished_at = now
                attempt.error_text = "Delivery attempt no longer matches the current step."
                run.status = "cancelled"
                run.updated_at = now
                await session.commit()
            elif attempt.status == FOLLOWUP_ATTEMPT_RETRYABLE:
                attempt_count = max(1, int(attempt.attempt_count or 1))
                if attempt_count >= FOLLOWUP_PREPARATION_RETRY_MAX_ATTEMPTS:
                    attempt.status = FOLLOWUP_ATTEMPT_RETRY_EXHAUSTED
                    attempt.finished_at = now
                    attempt.error_text = attempt.error_text or "Preparation retry limit reached."
                    run.status = FOLLOWUP_RUN_RETRY_EXHAUSTED
                    run.updated_at = now
                    await session.commit()
                else:
                    attempt.status = FOLLOWUP_ATTEMPT_CLAIMED
                    attempt.claim_token = uuid.uuid4().hex
                    attempt.claimed_at = now
                    attempt.finished_at = None
                    attempt.attempt_count = attempt_count + 1
                    await session.commit()
                    return _claim_snapshot(attempt, run, user, campaign, step)
            elif (
                attempt.status == FOLLOWUP_ATTEMPT_CLAIMED
                and now - (attempt.claimed_at or now) >= FOLLOWUP_ATTEMPT_STALE_AFTER
            ):
                attempt.status = FOLLOWUP_ATTEMPT_UNCERTAIN
                attempt.finished_at = now
                attempt.error_text = "Delivery claim became stale before completion."
                run.status = FOLLOWUP_RUN_UNCERTAIN
                run.updated_at = now
                await session.commit()
            elif attempt.status == FOLLOWUP_ATTEMPT_UNCERTAIN:
                run.status = FOLLOWUP_RUN_UNCERTAIN
                run.updated_at = now
                await session.commit()
            else:
                await session.rollback()
            return None

        attempt = FollowupDeliveryAttempt(
            run_id=run.id,
            step_id=step.id,
            step_index=run.next_step_index,
            generation=run.generation,
            claim_token=uuid.uuid4().hex,
            status=FOLLOWUP_ATTEMPT_CLAIMED,
            claimed_at=now,
            attempt_count=1,
        )
        session.add(attempt)
        try:
            await session.commit()
        except (IntegrityError, OperationalError):
            await session.rollback()
            return None
        return _claim_snapshot(attempt, run, user, campaign, step)


async def _release_claim(
    claim: FollowupDeliveryClaim,
    reason: str,
    *,
    cancel_run: bool = False,
) -> None:
    async with async_session_maker() as session:
        run = None
        if cancel_run:
            run = await session.scalar(
                select(FollowupRun)
                .where(
                    FollowupRun.id == claim.run_id,
                    FollowupRun.generation == claim.generation,
                    FollowupRun.next_step_index == claim.step_index,
                    FollowupRun.status == "active",
                )
                .with_for_update()
            )
        attempt = await session.scalar(
            select(FollowupDeliveryAttempt)
            .where(
                FollowupDeliveryAttempt.id == claim.attempt_id,
                FollowupDeliveryAttempt.run_id == claim.run_id,
                FollowupDeliveryAttempt.generation == claim.generation,
                FollowupDeliveryAttempt.step_index == claim.step_index,
                FollowupDeliveryAttempt.claim_token == claim.claim_token,
                FollowupDeliveryAttempt.status == FOLLOWUP_ATTEMPT_CLAIMED,
            )
            .with_for_update()
        )
        if attempt is not None:
            attempt.status = FOLLOWUP_ATTEMPT_CANCELLED
            attempt.finished_at = datetime.utcnow()
            attempt.error_text = reason
            if run is not None:
                run.status = "cancelled"
                run.updated_at = datetime.utcnow()
            await session.commit()
        else:
            await session.rollback()


async def _schedule_preparation_retry(
    claim: FollowupDeliveryClaim,
    error: Exception | str,
) -> None:
    now = datetime.utcnow()
    async with async_session_maker() as session:
        run = await session.scalar(
            select(FollowupRun)
            .where(FollowupRun.id == claim.run_id)
            .with_for_update()
        )
        attempt = await session.scalar(
            select(FollowupDeliveryAttempt)
            .where(
                FollowupDeliveryAttempt.id == claim.attempt_id,
                FollowupDeliveryAttempt.run_id == claim.run_id,
                FollowupDeliveryAttempt.step_id == claim.step_id,
                FollowupDeliveryAttempt.generation == claim.generation,
                FollowupDeliveryAttempt.step_index == claim.step_index,
                FollowupDeliveryAttempt.claim_token == claim.claim_token,
                FollowupDeliveryAttempt.status == FOLLOWUP_ATTEMPT_CLAIMED,
            )
            .with_for_update()
        )
        if attempt is None:
            await session.rollback()
            return

        attempt.finished_at = now
        attempt.error_text = str(error)
        current_generation = (
            run is not None
            and run.generation == claim.generation
            and run.next_step_index == claim.step_index
            and run.status == "active"
        )
        if not current_generation:
            attempt.status = FOLLOWUP_ATTEMPT_CANCELLED
        elif max(1, int(attempt.attempt_count or 1)) >= FOLLOWUP_PREPARATION_RETRY_MAX_ATTEMPTS:
            attempt.status = FOLLOWUP_ATTEMPT_RETRY_EXHAUSTED
            run.status = FOLLOWUP_RUN_RETRY_EXHAUSTED
            run.updated_at = now
        else:
            attempt.status = FOLLOWUP_ATTEMPT_RETRYABLE
            run.due_at = now + FOLLOWUP_PREPARATION_RETRY_DELAY
            run.updated_at = now
        await session.commit()


async def _refresh_delivery_claim(
    claim: FollowupDeliveryClaim,
    now: datetime,
) -> FollowupDeliveryClaim | None:
    async with async_session_maker() as session:
        run = await session.scalar(
            select(FollowupRun)
            .where(
                FollowupRun.id == claim.run_id,
                FollowupRun.campaign_id == claim.campaign_id,
                FollowupRun.generation == claim.generation,
                FollowupRun.next_step_index == claim.step_index,
                FollowupRun.status == "active",
                FollowupRun.due_at <= now,
            )
            .options(
                selectinload(FollowupRun.campaign).selectinload(FollowupCampaign.topics),
                selectinload(FollowupRun.campaign).selectinload(FollowupCampaign.steps),
            )
            .with_for_update()
        )
        if run is None:
            await session.rollback()
            return None
        attempt = await session.scalar(
            select(FollowupDeliveryAttempt)
            .where(
                FollowupDeliveryAttempt.id == claim.attempt_id,
                FollowupDeliveryAttempt.run_id == claim.run_id,
                FollowupDeliveryAttempt.step_id == claim.step_id,
                FollowupDeliveryAttempt.generation == claim.generation,
                FollowupDeliveryAttempt.step_index == claim.step_index,
                FollowupDeliveryAttempt.claim_token == claim.claim_token,
                FollowupDeliveryAttempt.status == FOLLOWUP_ATTEMPT_CLAIMED,
            )
            .with_for_update()
        )
        if attempt is None:
            await session.rollback()
            return None
        if now - (attempt.claimed_at or now) >= FOLLOWUP_ATTEMPT_STALE_AFTER:
            attempt.status = FOLLOWUP_ATTEMPT_UNCERTAIN
            attempt.finished_at = now
            attempt.error_text = "Delivery claim became stale before external delivery."
            run.status = FOLLOWUP_RUN_UNCERTAIN
            run.updated_at = now
            await session.commit()
            return None

        campaign = run.campaign
        user = await session.get(User, run.user_id)
        valid = (
            user is not None
            and campaign.is_active
            and user.current_dialogue_id == run.dialogue_id
            and (user.current_topic_id or 0) == run.topic_id
            and _campaign_matches_scope(campaign, run.topic_id)
            and 0 <= run.next_step_index < len(campaign.steps)
            and campaign.steps[run.next_step_index].id == claim.step_id
        )
        eligibility = None
        if valid:
            eligibility = await check_campaign_eligibility(
                session,
                campaign,
                user_id=run.user_id,
                dialogue_id=run.dialogue_id,
                topic_id=run.topic_id,
            )
            valid = eligibility.eligible
        if not valid:
            attempt.status = FOLLOWUP_ATTEMPT_CANCELLED
            attempt.finished_at = datetime.utcnow()
            attempt.error_text = "Delivery claim was invalidated before external delivery."
            await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == run.id,
                    FollowupRun.generation == run.generation,
                    FollowupRun.next_step_index == run.next_step_index,
                    FollowupRun.status == "active",
                )
                .values(status="cancelled", updated_at=datetime.utcnow())
            )
            await session.commit()
            return None
        await session.commit()
        return _claim_snapshot(attempt, run, user, campaign, campaign.steps[run.next_step_index])


async def _mark_claim_uncertain(claim: FollowupDeliveryClaim, error: Exception | str) -> None:
    now = datetime.utcnow()
    async with async_session_maker() as session:
        run = await session.scalar(
            select(FollowupRun)
            .where(FollowupRun.id == claim.run_id)
            .with_for_update()
        )
        attempt = await session.scalar(
            select(FollowupDeliveryAttempt)
            .where(
                FollowupDeliveryAttempt.id == claim.attempt_id,
                FollowupDeliveryAttempt.run_id == claim.run_id,
                FollowupDeliveryAttempt.generation == claim.generation,
                FollowupDeliveryAttempt.step_index == claim.step_index,
                FollowupDeliveryAttempt.claim_token == claim.claim_token,
                FollowupDeliveryAttempt.status == FOLLOWUP_ATTEMPT_CLAIMED,
            )
            .with_for_update()
        )
        if attempt is None:
            await session.rollback()
            return
        attempt.status = FOLLOWUP_ATTEMPT_UNCERTAIN
        attempt.finished_at = now
        attempt.error_text = str(error)
        if (
            run is not None
            and run.generation == claim.generation
            and run.next_step_index == claim.step_index
            and run.status == "active"
        ):
            run.status = FOLLOWUP_RUN_UNCERTAIN
            run.updated_at = now
        await session.commit()


async def _complete_delivery_claim(
    claim: FollowupDeliveryClaim,
    send_result: FollowupStepSendResult,
) -> bool:
    now = datetime.utcnow()
    async with async_session_maker() as session:
        run = await session.scalar(
            select(FollowupRun)
            .where(FollowupRun.id == claim.run_id)
            .with_for_update()
        )
        attempt = await session.scalar(
            select(FollowupDeliveryAttempt)
            .where(
                FollowupDeliveryAttempt.id == claim.attempt_id,
                FollowupDeliveryAttempt.run_id == claim.run_id,
                FollowupDeliveryAttempt.generation == claim.generation,
                FollowupDeliveryAttempt.step_index == claim.step_index,
                FollowupDeliveryAttempt.claim_token == claim.claim_token,
                FollowupDeliveryAttempt.status == FOLLOWUP_ATTEMPT_CLAIMED,
            )
            .with_for_update()
        )
        if attempt is None:
            await session.rollback()
            return False
        delivery = await session.scalar(
            select(FollowupDelivery).where(
                FollowupDelivery.run_id == claim.run_id,
                FollowupDelivery.generation == claim.generation,
                FollowupDelivery.step_id == claim.step_id,
            )
        )
        if delivery is None:
            session.add(DBMessage(
                user_id=claim.user.id,
                role="assistant",
                content=send_result.history_text or "Выберите действие:",
                ai_context_content=send_result.text if claim.step.message_type == "ai" else None,
                dialogue_id=claim.dialogue_id,
                topic_id=None if claim.topic_id == 0 else claim.topic_id,
            ))
            session.add(FollowupDelivery(
                run_id=claim.run_id,
                step_id=claim.step_id,
                generation=claim.generation,
                telegram_message_id=send_result.telegram_message_id,
            ))
        attempt.status = FOLLOWUP_ATTEMPT_DELIVERED
        attempt.finished_at = now
        campaign = None
        user = None
        current_step = None
        if run is not None and (
            run.generation == claim.generation
            and run.next_step_index == claim.step_index
            and run.status == "active"
        ):
            campaign = await session.scalar(
                select(FollowupCampaign)
                .where(FollowupCampaign.id == run.campaign_id)
                .options(selectinload(FollowupCampaign.topics), selectinload(FollowupCampaign.steps))
            )
            user = await session.get(User, run.user_id)
            current_step = (
                campaign.steps[run.next_step_index]
                if campaign is not None and 0 <= run.next_step_index < len(campaign.steps)
                else None
            )
        current = (
            run is not None
            and campaign is not None
            and user is not None
            and campaign.is_active
            and user.current_dialogue_id == run.dialogue_id
            and (user.current_topic_id or 0) == run.topic_id
            and _campaign_matches_scope(campaign, run.topic_id)
            and current_step is not None
            and current_step.id == claim.step_id
        )
        if current:
            eligibility = await check_campaign_eligibility(
                session,
                campaign,
                user_id=run.user_id,
                dialogue_id=run.dialogue_id,
                topic_id=run.topic_id,
            )
            current = eligibility.eligible
        if current:
            values = _advance_run_values(run, campaign)
            result = await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == claim.run_id,
                    FollowupRun.generation == claim.generation,
                    FollowupRun.next_step_index == claim.step_index,
                    FollowupRun.status == "active",
                )
                .values(values)
            )
            if result.rowcount != 1:
                current = False
        elif run is not None and (
            run.generation == claim.generation
            and run.next_step_index == claim.step_index
            and run.status == "active"
        ):
            await session.execute(
                update(FollowupRun)
                .where(
                    FollowupRun.id == claim.run_id,
                    FollowupRun.generation == claim.generation,
                    FollowupRun.next_step_index == claim.step_index,
                    FollowupRun.status == "active",
                )
                .values(status="cancelled", updated_at=now)
            )
        await session.commit()
        return True


async def process_due_followups(bot, *, limit: int = 100) -> int:
    """Deliver due steps and advance each chain exactly once per generation."""
    now = datetime.utcnow()
    delivered = 0
    run_ids = await _due_followup_ids(now, limit)
    for run_id in run_ids:
        claim = await _claim_due_followup(run_id, now)
        if claim is None:
            continue
        refreshed_claim = await _refresh_delivery_claim(claim, datetime.utcnow())
        if refreshed_claim is None:
            await _release_claim(claim, "Delivery claim was invalidated before external delivery.")
            continue
        claim = refreshed_claim
        try:
            prepared_result = await prepare_followup_step(
                user=claim.user,
                step=claim.step,
                dialogue_id=claim.dialogue_id,
                topic_id=claim.topic_id,
            )
        except Exception as exc:
            log.exception(
                "Follow-up preparation failed: run=%s step=%s",
                claim.run_id,
                claim.step_id,
            )
            if isinstance(exc, FollowupStepConfigurationError):
                await _release_claim(
                    claim,
                    f"Follow-up step configuration is invalid: {exc}",
                    cancel_run=True,
                )
            else:
                await _schedule_preparation_retry(
                    claim,
                    f"Follow-up preparation failed before external delivery: {exc}",
                )
            continue
        refreshed_claim = await _refresh_delivery_claim(claim, datetime.utcnow())
        if refreshed_claim is None:
            await _release_claim(claim, "Delivery claim was invalidated before external delivery.")
            continue
        claim = refreshed_claim
        try:
            validate_followup_step(claim.step)
            send_result = await emit_followup_step(
                bot,
                user=claim.user,
                step=claim.step,
                send_result=prepared_result,
            )
        except FollowupStepConfigurationError as exc:
            log.exception(
                "Follow-up step configuration changed before delivery: run=%s step=%s",
                claim.run_id,
                claim.step_id,
            )
            await _release_claim(
                claim,
                f"Follow-up step configuration is invalid: {exc}",
                cancel_run=True,
            )
            continue
        except Exception as exc:
            log.exception(
                "Follow-up delivery failed: run=%s step=%s",
                claim.run_id,
                claim.step_id,
            )
            await _mark_claim_uncertain(claim, exc)
            continue
        try:
            completed = await _complete_delivery_claim(claim, send_result)
        except Exception as exc:
            log.exception(
                "Follow-up delivery finalization failed: run=%s step=%s",
                claim.run_id,
                claim.step_id,
            )
            await _mark_claim_uncertain(claim, f"finalization failed: {exc}")
            continue
        if completed:
            delivered += 1
    return delivered
