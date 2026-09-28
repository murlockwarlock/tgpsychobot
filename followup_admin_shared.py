from contextlib import asynccontextmanager

from sqlalchemy import func, select

from database import FollowupCampaign, FollowupStep
from translation_pack_manager import commit_readiness_critical_mutation, translation_coordination_lock


@asynccontextmanager
async def followup_step_mutation_lock(session):
    async with translation_coordination_lock(session):
        yield


async def allocate_followup_step_sort_order(session, campaign_id: int) -> int:
    campaign = await session.scalar(
        select(FollowupCampaign)
        .where(FollowupCampaign.id == campaign_id)
        .with_for_update()
    )
    if campaign is None:
        raise ValueError("Цепочка не найдена.")
    current_max = await session.scalar(
        select(func.max(FollowupStep.sort_order))
        .where(FollowupStep.campaign_id == campaign_id)
    )
    return int(current_max if current_max is not None else -1) + 1


async def save_static_followup_text(session, step, locale: str, value: str) -> None:
    from content_authoring import save_content_value

    await save_content_value(session, "followup_step", step, "message_text", locale, value)


async def commit_followup_step_mutation(session) -> None:
    await session.flush()
    revision_bumped = bool(session.info.get("authoring_revision_bumped"))
    await commit_readiness_critical_mutation(session, bump_revision=not revision_bumped)
