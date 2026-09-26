from __future__ import annotations

import asyncio
import html
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from followup_admin_contract import (
    FOLLOWUP_CAMPAIGN_EXPLANATION,
    FOLLOWUP_CAMPAIGN_DETAIL_INTRO,
    FOLLOWUP_METADATA_LABELS,
    FOLLOWUP_STAGE_LABELS,
    FOLLOWUP_STEPS_EXPLANATION,
    parse_followup_step_input,
)
from followups import (
    FOLLOWUP_STAGE_MODES,
    FOLLOWUP_STEP_DELETE_PROTECTED_ATTEMPT_STATUSES,
    FollowupTransportRegistry,
    UNSET_STAGE_TOKEN,
    _canonical_followup_stage_condition,
    check_campaign_eligibility,
    emit_followup_step,
    parse_followup_csv,
    prepare_followup_step,
)
from translation_service import resolve_user_effective_locale, translate
from translation_pack_manager import translation_coordination_lock

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..identity import max_communication_name, raw_max_user_id
from ..legacy import (
    FollowupCampaign,
    FollowupDelivery,
    FollowupDeliveryAttempt,
    FollowupStep,
    Topic,
    User,
    async_session_maker,
    followup_campaign_topic_association,
)
from ..storage import StateStore


def _back(payload: str) -> list[dict]:
    return [callback_button("⬅️ Назад", payload)]


_max_followup_tests_inflight: dict[tuple[int, int], asyncio.Event] = {}


async def _campaign(session, campaign_id: int):
    return await session.scalar(
        select(FollowupCampaign)
        .where(FollowupCampaign.id == campaign_id)
        .options(
            selectinload(FollowupCampaign.topics),
            selectinload(FollowupCampaign.steps),
        )
    )


def _scope_text(item: FollowupCampaign) -> str:
    if item.all_topics:
        return "все темы"
    names = ["основной диалог"] if item.include_main_dialogue else []
    names.extend(topic.name for topic in item.topics)
    return ", ".join(names) or "не выбраны"


def _campaign_text(item: FollowupCampaign) -> str:
    valid = bool(item.steps and (item.all_topics or item.include_main_dialogue or item.topics))
    warning = "" if valid else "\n\n⚠️ Для включения выберите область и добавьте хотя бы один шаг."
    return (
        f"💬 <b>{html.escape(item.name)}</b>\n\n"
        f"{FOLLOWUP_CAMPAIGN_DETAIL_INTRO}\n"
        f"Статус: {'✅ включена' if item.is_active else '⏸ выключена'}\n"
        f"Область: {html.escape(_scope_text(item))}\n"
        f"Шагов: {len(item.steps)}\n"
        f"Тихие часы: {item.quiet_start_minute // 60:02d}:{item.quiet_start_minute % 60:02d}–"
        f"{item.quiet_end_minute // 60:02d}:{item.quiet_end_minute % 60:02d} ({html.escape(item.timezone)})\n"
        f"Случайная задержка: {item.jitter_min_seconds}–{item.jitter_max_seconds} сек.{warning}"
    )


def _campaign_keyboard(item: FollowupCampaign, back: str) -> list[dict]:
    return inline_keyboard([
        [callback_button(f"Статус: {'✅ включена' if item.is_active else '⏸ выключена'}", f"admin_fu_toggle_{item.id}")],
        [callback_button("✏️ Переименовать", f"admin_fu_rename_{item.id}")],
        [callback_button("💬 Темы", f"admin_fu_topics_{item.id}")],
        [callback_button(f"🪜 Шаги ({len(item.steps)})", f"admin_fu_steps_{item.id}")],
        [callback_button("⚙️ Условия", f"admin_fu_conditions_{item.id}")],
        [callback_button("🧪 Проверить на себе", f"admin_fu_self_test_{item.id}")],
        [callback_button("🌙 Тихие часы", f"admin_fu_quiet_{item.id}")],
        [callback_button("🎲 Случайная задержка", f"admin_fu_jitter_{item.id}")],
        [callback_button("🗑 Удалить", f"admin_fu_delete_ask_{item.id}")],
        _back(back),
    ])


async def show_campaigns(client: MaxApiClient, chat_id: int, *, back: str = "admin_panel") -> None:
    async with async_session_maker() as session:
        campaigns = (await session.scalars(select(FollowupCampaign).order_by(FollowupCampaign.id))).all()
    rows = []
    for item in campaigns:
        icon = "🌐" if item.all_topics else ("✅" if item.is_active else "⏸")
        rows.append([callback_button(f"{icon} {item.name}", f"admin_fu_campaign_{item.id}")])
    rows.append([callback_button("➕ Новая цепочка", "admin_fu_add")])
    rows.append(_back(back))
    await client.send_message(
        chat_id=chat_id,
        text=f"💬 <b>Догоняющие сообщения ({len(campaigns)})</b>\n\n{FOLLOWUP_CAMPAIGN_EXPLANATION}",
        attachments=inline_keyboard(rows),
    )


async def show_campaign(client: MaxApiClient, chat_id: int, campaign_id: int, *, back: str = "admin_fu_list") -> None:
    async with async_session_maker() as session:
        item = await _campaign(session, campaign_id)
    if item is None:
        await client.send_message(chat_id=chat_id, text="Цепочка не найдена.")
        return
    await client.send_message(chat_id=chat_id, text=_campaign_text(item), attachments=_campaign_keyboard(item, back))


async def start_campaign_add(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "max_followup_campaign_name", {"back": "admin_fu_list"})
    await client.send_message(
        chat_id=chat_id,
        text="Введите название цепочки. После создания она будет выключена, пока вы не добавите шаги.",
        attachments=inline_keyboard([_back("admin_fu_list")]),
    )


async def receive_campaign_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    name = value.strip()
    if not 2 <= len(name) <= 100:
        await client.send_message(chat_id=chat_id, text="Название должно содержать от 2 до 100 символов.")
        return
    async with async_session_maker() as session:
        item = FollowupCampaign(name=name, include_main_dialogue=True, is_active=False)
        session.add(item)
        await session.commit()
        await session.refresh(item)
        campaign_id = item.id
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Цепочка создана выключенной.")
    await show_campaign(client, chat_id, campaign_id)


async def toggle_campaign(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await _campaign(session, campaign_id)
        if item is None:
            return
        if not item.is_active and not (item.steps and (item.all_topics or item.include_main_dialogue or item.topics)):
            await client.send_message(chat_id=chat_id, text="Сначала выберите область и добавьте шаг.")
            return
        item.is_active = not item.is_active
        await session.commit()
    await show_campaign(client, chat_id, campaign_id)


async def start_rename(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
    if item is None:
        await client.send_message(chat_id=chat_id, text="Цепочка не найдена.")
        return
    await states.set(user_id, chat_id, "max_followup_campaign_rename", {"campaign_id": campaign_id})
    await client.send_message(
        chat_id=chat_id,
        text=f"✏️ <b>Переименовать цепочку</b>\n\nТекущее название: <code>{html.escape(item.name)}</code>\n\nВведите новое название от 2 до 100 символов.",
        attachments=inline_keyboard([_back(f"admin_fu_campaign_{campaign_id}")]),
    )


async def receive_rename(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    campaign_id = snapshot.data.get("campaign_id") if snapshot else None
    name = value.strip()
    if campaign_id is None or not 2 <= len(name) <= 100:
        await client.send_message(chat_id=chat_id, text="Название должно содержать от 2 до 100 символов.")
        return
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, int(campaign_id))
        if item is None:
            await states.clear(user_id)
            await client.send_message(chat_id=chat_id, text="Цепочка не найдена.")
            return
        item.name = name
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Название цепочки изменено.")
    await show_campaign(client, chat_id, int(campaign_id))


async def show_topics(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await _campaign(session, campaign_id)
        topics = (await session.scalars(select(Topic).order_by(Topic.name))).all()
    if item is None:
        return
    selected = {topic.id for topic in item.topics}
    rows = [
        [callback_button(f"{'✅' if item.all_topics else '❌'} Все темы", f"admin_fu_scope_all_{campaign_id}")],
        [callback_button(f"{'✅' if item.include_main_dialogue else '❌'} Основной диалог", f"admin_fu_scope_main_{campaign_id}")],
    ]
    rows.extend([[callback_button(f"{'✅' if topic.id in selected else '❌'} {topic.name}", f"admin_fu_scope_topic_{campaign_id}_{topic.id}")] for topic in topics])
    rows.append(_back(f"admin_fu_campaign_{campaign_id}"))
    await client.send_message(
        chat_id=chat_id,
        text="💬 <b>Темы цепочки</b>\n\n«Все темы» автоматически включает основной диалог и будущие темы. Иначе выберите нужные области отдельно.",
        attachments=inline_keyboard(rows),
    )


async def toggle_scope(client: MaxApiClient, chat_id: int, campaign_id: int, scope: str, topic_id: int | None = None) -> None:
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item is None:
            return
        if scope == "all":
            item.all_topics = not item.all_topics
        elif scope == "main":
            item.include_main_dialogue = not item.include_main_dialogue
        elif topic_id is not None:
            clause = (followup_campaign_topic_association.c.campaign_id == campaign_id, followup_campaign_topic_association.c.topic_id == topic_id)
            exists = await session.scalar(select(followup_campaign_topic_association.c.campaign_id).where(*clause))
            if exists:
                await session.execute(delete(followup_campaign_topic_association).where(*clause))
            else:
                await session.execute(followup_campaign_topic_association.insert().values(campaign_id=campaign_id, topic_id=topic_id))
        await session.commit()
    await show_topics(client, chat_id, campaign_id)


def _conditions_text(item: FollowupCampaign) -> str:
    condition = _canonical_followup_stage_condition(item)
    stage = FOLLOWUP_STAGE_LABELS.get(condition.mode, FOLLOWUP_STAGE_LABELS["all"])
    if condition.mode in {"selected", "all_except"}:
        stage = f"{stage}: {', '.join(condition.values) or 'не заданы'}"
    metadata = "не заданы"
    if item.metadata_field_path:
        metadata = f"{item.metadata_field_path} {FOLLOWUP_METADATA_LABELS.get(item.metadata_operator or 'equals', item.metadata_operator)} {item.metadata_expected_value}"
    stops = ", ".join(parse_followup_csv(item.stop_events)) or "не заданы"
    return f"⚙️ <b>Условия цепочки</b>\n\nЭтапы:\n{html.escape(stage)}\n\nМетаданные:\n{html.escape(metadata)}\n\nСобытия остановки:\n{html.escape(stops)}"


async def show_conditions(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
    if item is None:
        return
    rows = [
        [callback_button("✏️ Изменить этапы", f"admin_fu_stage_{campaign_id}")],
        [callback_button("✏️ Изменить метаданные", f"admin_fu_metadata_{campaign_id}")],
        [callback_button("✏️ Изменить события остановки", f"admin_fu_stops_{campaign_id}")],
        _back(f"admin_fu_campaign_{campaign_id}"),
    ]
    if item.metadata_field_path:
        rows.insert(2, [callback_button("🧹 Очистить метаданные", f"admin_fu_metadata_clear_{campaign_id}")])
    if parse_followup_csv(item.stop_events):
        rows.insert(4 if item.metadata_field_path else 3, [callback_button("🧹 Очистить события", f"admin_fu_stops_clear_{campaign_id}")])
    await client.send_message(chat_id=chat_id, text=_conditions_text(item), attachments=inline_keyboard(rows))


async def clear_metadata(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item:
            item.metadata_field_path = None
            item.metadata_operator = None
            item.metadata_expected_value = None
            await session.commit()
    await show_conditions(client, chat_id, campaign_id)


async def clear_stops(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item:
            item.stop_events = ""
            await session.commit()
    await show_conditions(client, chat_id, campaign_id)


async def show_stage_picker(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
    current_mode = _canonical_followup_stage_condition(item).mode if item is not None else "all"
    rows = [
        [callback_button(("✅ " if mode == current_mode else "") + FOLLOWUP_STAGE_LABELS[mode], f"admin_fu_stage_mode_{campaign_id}_{mode}")]
        for mode in FOLLOWUP_STAGE_MODES
    ]
    rows.append(_back(f"admin_fu_conditions_{campaign_id}"))
    await client.send_message(chat_id=chat_id, text="🪜 <b>Этапы запуска</b>\n\nВыберите режим проверки текущего этапа.", attachments=inline_keyboard(rows))


async def select_stage_mode(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int, mode: str) -> None:
    if mode == "all":
        async with async_session_maker() as session:
            item = await session.get(FollowupCampaign, campaign_id)
            if item:
                item.stage_mode = "all"
                item.stage_values = ""
                item.stage_include_unset = True
                await session.commit()
        await show_conditions(client, chat_id, campaign_id)
        return
    await states.set(user_id, chat_id, "max_followup_stage_values", {"campaign_id": campaign_id, "mode": mode})
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"🪜 <b>{FOLLOWUP_STAGE_LABELS[mode]}</b>\n\n"
            "Введите точные названия этапов через запятую. Регистр сохраняется.\n"
            f"Для незаданного этапа используйте <code>{UNSET_STAGE_TOKEN}</code>.\n"
            "Примеры: <code>guide_choice, [не задан]</code>; "
            "<code>guide_choice, thinking, child_words, completed, [не задан]</code>."
        ),
        attachments=inline_keyboard([_back(f"admin_fu_stage_{campaign_id}")]),
    )


async def receive_stage_values(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    values = parse_followup_csv(value)
    if not snapshot or not values:
        await client.send_message(chat_id=chat_id, text="Укажите хотя бы один этап через запятую.")
        return
    campaign_id = int(snapshot.data["campaign_id"])
    mode = snapshot.data["mode"]
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item is None:
            await states.clear(user_id)
            return
        item.stage_mode = mode
        item.stage_values = ", ".join(values)
        item.stage_include_unset = UNSET_STAGE_TOKEN in values if mode == "selected" else UNSET_STAGE_TOKEN not in values
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Условия этапов сохранены.")
    await show_conditions(client, chat_id, campaign_id)


async def start_metadata(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int) -> None:
    await states.set(user_id, chat_id, "max_followup_metadata_field", {"campaign_id": campaign_id})
    await client.send_message(chat_id=chat_id, text="🧩 <b>Метаданные</b>\n\nВведите путь поля, например <code>profile.outcome</code>.", attachments=inline_keyboard([_back(f"admin_fu_conditions_{campaign_id}")]))


async def receive_metadata_field(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    field = value.strip()
    if not snapshot or not field or any(char.isspace() for char in field) or len(field) > 200:
        await client.send_message(chat_id=chat_id, text="Введите непустой путь без пробелов, например <code>profile.outcome</code>.")
        return
    await states.set(user_id, chat_id, "max_followup_metadata_operator", {"campaign_id": snapshot.data["campaign_id"], "field": field})
    rows = [[callback_button(label, f"admin_fu_metadata_op_{snapshot.data['campaign_id']}_{operator}")] for operator, label in FOLLOWUP_METADATA_LABELS.items()]
    rows.append(_back(f"admin_fu_metadata_{snapshot.data['campaign_id']}"))
    await client.send_message(chat_id=chat_id, text=f"Поле: <code>{html.escape(field)}</code>\n\nВыберите оператор:", attachments=inline_keyboard(rows))


async def select_metadata_operator(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int, operator: str) -> None:
    snapshot = await states.get(user_id)
    field = snapshot.data.get("field") if snapshot else None
    if not field:
        await show_conditions(client, chat_id, campaign_id)
        return
    await states.set(user_id, chat_id, "max_followup_metadata_value", {"campaign_id": campaign_id, "field": field, "operator": operator})
    await client.send_message(
        chat_id=chat_id,
        text=f"🧩 Поле: <code>{html.escape(field)}</code>\nОператор: <b>{FOLLOWUP_METADATA_LABELS[operator]}</b>\n\nВведите значение:",
        attachments=inline_keyboard([_back(f"admin_fu_metadata_operator_edit_{campaign_id}")]),
    )


async def show_metadata_operator(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int) -> None:
    snapshot = await states.get(user_id)
    field = snapshot.data.get("field") if snapshot else None
    if not field or snapshot.data.get("campaign_id") != campaign_id:
        await states.clear(user_id)
        await show_conditions(client, chat_id, campaign_id)
        return
    await states.set(user_id, chat_id, "max_followup_metadata_operator", {"campaign_id": campaign_id, "field": field})
    rows = [[callback_button(label, f"admin_fu_metadata_op_{campaign_id}_{operator}")] for operator, label in FOLLOWUP_METADATA_LABELS.items()]
    rows.append(_back(f"admin_fu_metadata_{campaign_id}"))
    await client.send_message(
        chat_id=chat_id,
        text=f"Поле: <code>{html.escape(field)}</code>\n\nВыберите оператор:",
        attachments=inline_keyboard(rows),
    )


async def receive_metadata_value(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    if not snapshot or not value.strip():
        await client.send_message(chat_id=chat_id, text="Значение не должно быть пустым.")
        return
    data = snapshot.data
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, int(data["campaign_id"]))
        if item is None:
            await states.clear(user_id)
            return
        item.metadata_field_path = data["field"]
        item.metadata_operator = data["operator"]
        item.metadata_expected_value = value.strip()
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Условие метаданных сохранено.")
    await show_conditions(client, chat_id, int(data["campaign_id"]))


async def start_stops(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int) -> None:
    await states.set(user_id, chat_id, "max_followup_stop_events", {"campaign_id": campaign_id})
    rows = []
    rows.append([callback_button("🧹 Очистить список", f"admin_fu_stops_clear_{campaign_id}")])
    rows.append(_back(f"admin_fu_conditions_{campaign_id}"))
    await client.send_message(
        chat_id=chat_id,
        text="🛑 <b>События остановки</b>\n\nВведите точные имена событий через запятую. Пустой список не останавливает цепочку.",
        attachments=inline_keyboard(rows),
    )


async def receive_stops(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    if not snapshot:
        return
    campaign_id = int(snapshot.data["campaign_id"])
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item:
            item.stop_events = ", ".join(parse_followup_csv(value))
            await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ События остановки сохранены.")
    await show_conditions(client, chat_id, campaign_id)


async def show_steps(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        item = await _campaign(session, campaign_id)
    if item is None:
        return
    rows = []
    for index, step in enumerate(item.steps, 1):
        kind = "AI" if step.message_type == "ai" else "текст"
        rows.append([callback_button(f"{index}. через {step.delay_minutes} мин. — {kind}", f"admin_fu_step_{campaign_id}_{step.id}")])
    rows.extend([
        [callback_button("➕ Обычный текст", f"admin_fu_step_add_{campaign_id}_static")],
        [callback_button("➕ Сгенерировать через AI", f"admin_fu_step_add_{campaign_id}_ai")],
        _back(f"admin_fu_campaign_{campaign_id}"),
    ])
    await client.send_message(chat_id=chat_id, text=f"🪜 <b>Шаги цепочки ({len(item.steps)})</b>\n\n{FOLLOWUP_STEPS_EXPLANATION}", attachments=inline_keyboard(rows))


async def show_step(client: MaxApiClient, chat_id: int, campaign_id: int, step_id: int) -> None:
    async with async_session_maker() as session:
        item = await _campaign(session, campaign_id)
    if item is None:
        return
    step = next((candidate for candidate in item.steps if candidate.id == step_id), None)
    if step is None:
        return
    index = next(index for index, candidate in enumerate(item.steps, 1) if candidate.id == step_id)
    kind = "AI" if step.message_type == "ai" else "static"
    content_label = "Инструкция" if kind == "AI" else "Текст"
    if kind == "AI":
        content = step.ai_instruction or "не задано"
    else:
        from content_authoring import read_content_value

        async with async_session_maker() as session:
            content = (await read_content_value(session, "followup_step", step, "message_text", "ru")).admin_label()
    content = content[:3000] + ("…" if len(content) > 3000 else "")
    text = f"🪜 <b>Шаг {index}</b>\n\nИндекс: <b>{index}</b>\nЗадержка: <b>{step.delay_minutes} мин.</b>\nТип: <b>{kind}</b>\n{content_label}:\n<code>{html.escape(content)}</code>"
    rows = [[callback_button("✏️ Редактировать", f"admin_fu_step_edit_{campaign_id}_{step_id}")]]
    if kind == "static":
        rows.append([callback_button("Текст сообщения", f"admin_fu_step_text_{step_id}")])
    rows.extend([[callback_button("🗑 Удалить", f"admin_fu_step_delete_{campaign_id}_{step_id}")], _back(f"admin_fu_steps_{campaign_id}")])
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(rows))


async def show_step_text(client: MaxApiClient, chat_id: int, step_id: int, locale: str = "ru") -> None:
    locale = "ru"
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, step_id)
        if step is None or step.message_type != "static":
            return
        from content_authoring import read_content_value

        value = await read_content_value(session, "followup_step", step, "message_text", locale)
        campaign_id = step.campaign_id
    rows = [
        [callback_button("Изменить: Сообщение", f"admin_fu_step_text_edit_{step_id}_ru")],
        _back(f"admin_fu_step_{campaign_id}_{step_id}"),
    ]
    preview = value.admin_label()
    if len(preview) > 250:
        preview = preview[:247] + "…"
    preview = html.escape(preview)
    await client.send_message(
        chat_id=chat_id,
        text=f"<b>Сообщения follow-up #{step_id}</b>\n\n<b>Сообщение:</b>\n{preview}",
        attachments=inline_keyboard(rows),
    )


async def start_step_text_edit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, step_id: int, locale: str) -> None:
    locale = "ru"
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, step_id)
        if step is None or step.message_type != "static":
            return
        from content_authoring import read_content_value

        value = await read_content_value(session, "followup_step", step, "message_text", locale)
    await states.set(user_id, chat_id, "max_followup_step_text", {"step_id": step_id, "locale": locale})
    current = value.text or "Не задано"
    await client.send_message(
        chat_id=chat_id,
        text=f"Сообщение:\n{html.escape(current[:1500])}\n\nВведите новое значение.",
        attachments=inline_keyboard([_back(f"admin_fu_step_text_{step_id}_{locale}")]),
    )


async def receive_step_text(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    text = value.strip()
    if not snapshot or not text:
        await client.send_message(chat_id=chat_id, text="Значение не может быть пустым.")
        return
    data = snapshot.data
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, int(data["step_id"]))
        if step is None or step.message_type != "static":
            await states.clear(user_id)
            return
        from content_authoring import save_content_value

        try:
            await save_content_value(session, "followup_step", step, "message_text", data["locale"], text)
        except ValueError as exc:
            await client.send_message(chat_id=chat_id, text=str(exc))
            return
        await session.commit()
        step_id = step.id
    from translation_service import refresh_translation_cache

    await refresh_translation_cache(async_session_maker, force=True)
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Сообщение сохранено.")
    await show_step_text(client, chat_id, step_id, data["locale"])


async def start_step_add(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int, message_type: str) -> None:
    await states.set(user_id, chat_id, "max_followup_step_add", {"campaign_id": campaign_id, "message_type": message_type})
    field = "текст сообщения" if message_type == "static" else "инструкцию для AI"
    extra = "AI сам видит текущий диалог. Напишите только, что нужно сказать; DATA добавлять не нужно." if message_type == "ai" else ""
    await client.send_message(chat_id=chat_id, text=f"<b>Новый шаг</b>\n\nВ первой строке укажите задержку в минутах, ниже — {field}.\n\nПример:\n<code>60\nМягко напомни пользователю о незавершённом упражнении.</code>\n\n{extra}", attachments=inline_keyboard([_back(f"admin_fu_steps_{campaign_id}")]))


async def receive_step_add(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    parsed = parse_followup_step_input(value)
    if not snapshot:
        await client.send_message(chat_id=chat_id, text="Нужны минуты в первой строке и текст ниже.")
        return
    if parsed is None:
        first, separator, body = (value or "").partition("\n")
        if not separator or not first.strip().isdigit() or not body.strip():
            await client.send_message(chat_id=chat_id, text="Нужны минуты в первой строке и текст ниже.")
            return
        await client.send_message(chat_id=chat_id, text="Задержка должна быть от 1 минуты до 365 дней.")
        return
    delay, content = parsed
    if not content:
        await client.send_message(chat_id=chat_id, text="Нужны минуты в первой строке и текст ниже.")
        return
    data = snapshot.data
    async with async_session_maker() as session:
        order = await session.scalar(
            select(func.count(FollowupStep.id)).where(FollowupStep.campaign_id == int(data["campaign_id"]))
        ) or 0
        step = FollowupStep(campaign_id=int(data["campaign_id"]), sort_order=order, delay_minutes=delay, message_type=data["message_type"])
        if data["message_type"] == "ai":
            step.ai_instruction = content
        else:
            step.message_text = content
        session.add(step)
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Шаг добавлен.")
    await show_steps(client, chat_id, int(data["campaign_id"]))


async def start_step_edit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int, step_id: int) -> None:
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, step_id)
    if step is None or step.campaign_id != campaign_id:
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="Шаг не найден.")
        return
    content = step.ai_instruction if step.message_type == "ai" else step.message_text
    await states.set(user_id, chat_id, "max_followup_step_edit", {"campaign_id": campaign_id, "step_id": step_id})
    label = "текст сообщения" if step.message_type == "static" else "инструкцию для AI"
    await client.send_message(chat_id=chat_id, text=f"<b>Редактирование шага</b>\n\nВ первой строке укажите задержку в минутах, ниже — {label}.", attachments=inline_keyboard([_back(f"admin_fu_step_{campaign_id}_{step_id}")]))


async def receive_step_edit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    parsed = parse_followup_step_input(value)
    if not snapshot:
        await client.send_message(chat_id=chat_id, text="Нужны минуты в первой строке и текст ниже.")
        return
    if parsed is None:
        first, separator, body = (value or "").partition("\n")
        if not separator or not first.strip().isdigit() or not body.strip():
            await client.send_message(chat_id=chat_id, text="Нужны минуты в первой строке и текст ниже.")
            return
        await client.send_message(chat_id=chat_id, text="Задержка должна быть от 1 минуты до 365 дней.")
        return
    delay, content = parsed
    if not content:
        await client.send_message(chat_id=chat_id, text="Нужны минуты в первой строке и текст ниже.")
        return
    data = snapshot.data
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, int(data["step_id"]))
        if step is None or step.campaign_id != int(data["campaign_id"]):
            await states.clear(user_id)
            await client.send_message(chat_id=chat_id, text="Шаг не найден.")
            return
        step.delay_minutes = delay
        if step.message_type == "ai":
            step.ai_instruction = content
        else:
            step.message_text = content
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Шаг изменён.")
    await show_step(client, chat_id, int(data["campaign_id"]), int(data["step_id"]))


async def ask_delete_step(client: MaxApiClient, chat_id: int, campaign_id: int, step_id: int) -> None:
    await delete_step(client, chat_id, campaign_id, step_id)


async def delete_step(client: MaxApiClient, chat_id: int, campaign_id: int, step_id: int) -> None:
    async with async_session_maker() as session:
        async with translation_coordination_lock(session):
            step = await session.scalar(
                select(FollowupStep)
                .where(FollowupStep.id == step_id, FollowupStep.campaign_id == campaign_id)
                .with_for_update()
            )
            if step is None:
                return
            sent_count = await session.scalar(select(func.count(FollowupDelivery.id)).where(FollowupDelivery.step_id == step_id)) or 0
            protected_count = await session.scalar(
                select(func.count(FollowupDeliveryAttempt.id)).where(
                    FollowupDeliveryAttempt.step_id == step_id,
                    FollowupDeliveryAttempt.status.in_(FOLLOWUP_STEP_DELETE_PROTECTED_ATTEMPT_STATUSES),
                )
            ) or 0
            if sent_count or protected_count:
                await client.send_message(chat_id=chat_id, text="Шаг уже отправлялся или находится в процессе отправки, его нельзя удалить без потери истории.")
                await show_steps(client, chat_id, campaign_id)
                return
            await session.delete(step)
            await session.flush()
            remaining = (
                await session.scalars(
                    select(FollowupStep)
                    .where(FollowupStep.campaign_id == campaign_id)
                    .order_by(FollowupStep.sort_order, FollowupStep.id)
                )
            ).all()
            for index, remaining_step in enumerate(remaining):
                remaining_step.sort_order = index
            await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Шаг удалён.")
    await show_steps(client, chat_id, campaign_id)


async def start_quiet(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int) -> None:
    await states.set(user_id, chat_id, "max_followup_quiet", {"campaign_id": campaign_id})
    await client.send_message(chat_id=chat_id, text="🌙 <b>Тихие часы</b>\n\nВведите интервал и часовую зону в формате:\n<code>22:00-09:00 Europe/Moscow</code>\n\nСообщение, попавшее в этот интервал, переносится на его окончание.", attachments=inline_keyboard([_back(f"admin_fu_campaign_{campaign_id}")]))


async def receive_quiet(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\s+([A-Za-z_]+/[A-Za-z_]+)\s*", value)
    if not snapshot or not match:
        await client.send_message(chat_id=chat_id, text="Формат: <code>22:00-09:00 Europe/Moscow</code>")
        return
    try:
        start_h, start_m, end_h, end_m = map(int, match.groups()[:4])
        if not (0 <= start_h < 24 and 0 <= start_m < 60 and 0 <= end_h < 24 and 0 <= end_m < 60):
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Проверьте время: часы 0–23, минуты 0–59.")
        return
    try:
        ZoneInfo(match.group(5))
    except ZoneInfoNotFoundError:
        await client.send_message(chat_id=chat_id, text="Неизвестная часовая зона. Пример: <code>Europe/Moscow</code>.")
        return
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, int(snapshot.data["campaign_id"]))
        if item:
            item.quiet_start_minute = start_h * 60 + start_m
            item.quiet_end_minute = end_h * 60 + end_m
            item.timezone = match.group(5)
            await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Тихие часы сохранены.")
    await show_campaign(client, chat_id, int(snapshot.data["campaign_id"]))


async def start_jitter(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, campaign_id: int) -> None:
    await states.set(user_id, chat_id, "max_followup_jitter", {"campaign_id": campaign_id})
    await client.send_message(chat_id=chat_id, text="🎲 <b>Случайная задержка</b>\n\nВведите диапазон в секундах, например <code>30-180</code>. Для точного времени отправки укажите <code>0-0</code>.", attachments=inline_keyboard([_back(f"admin_fu_campaign_{campaign_id}")]))


async def receive_jitter(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, value: str) -> None:
    snapshot = await states.get(user_id)
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if not snapshot or not match:
        await client.send_message(chat_id=chat_id, text="Формат диапазона: <code>30-180</code>.")
        return
    try:
        low, high = map(int, match.groups())
        if low < 0 or high < low or high > 86400:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Минимум не должен превышать максимум; максимум — 86400 секунд.")
        return
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, int(snapshot.data["campaign_id"]))
        if item:
            item.jitter_min_seconds = low
            item.jitter_max_seconds = high
            await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Случайная задержка сохранена.")
    await show_campaign(client, chat_id, int(snapshot.data["campaign_id"]))


async def delete_campaign(client: MaxApiClient, chat_id: int, campaign_id: int) -> None:
    async with async_session_maker() as session:
        async with translation_coordination_lock(session):
            item = await session.get(FollowupCampaign, campaign_id)
            if item:
                await session.delete(item)
                await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Цепочка удалена.")
    await show_campaigns(client, chat_id)


def _self_test_stage_text(item: FollowupCampaign) -> str:
    condition = _canonical_followup_stage_condition(item)
    label = FOLLOWUP_STAGE_LABELS.get(condition.mode, FOLLOWUP_STAGE_LABELS["all"])
    if condition.mode in {"selected", "all_except"}:
        return f"{html.escape(label)}: {html.escape(', '.join(condition.values) or 'не заданы')}"
    return html.escape(label)


def _self_test_metadata_text(item: FollowupCampaign) -> str:
    field_path = (getattr(item, "metadata_field_path", None) or "").strip()
    if not field_path:
        return "не заданы"
    operator = getattr(item, "metadata_operator", None) or "equals"
    value = "" if getattr(item, "metadata_expected_value", None) is None else str(item.metadata_expected_value)
    return (
        f"{html.escape(field_path)} "
        f"{html.escape(FOLLOWUP_METADATA_LABELS.get(operator, operator))} "
        f"{html.escape(value)}"
    )


async def _self_test_snapshot(session, campaign_id: int, user_id: int, step_index: int = 0) -> dict:
    item = await _campaign(session, campaign_id)
    user = await session.get(User, user_id, options=[selectinload(User.current_topic)])
    snapshot = {
        "campaign": item,
        "user": user,
        "dialogue_id": None,
        "topic_id": 0,
        "eligibility": None,
        "step": None,
        "step_index": max(0, int(step_index or 0)),
        "campaign_valid": False,
        "can_send": False,
        "reason": "campaign_not_found" if item is None else "user_not_found",
    }
    if item is None or user is None:
        return snapshot

    dialogue_id = user.current_dialogue_id or 1
    topic_id = user.current_topic_id or 0
    eligibility = await check_campaign_eligibility(
        session,
        item,
        user_id=user_id,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
    )
    step_index = max(0, int(step_index or 0))
    step = item.steps[step_index] if step_index < len(item.steps) else None
    campaign_valid = bool(item.is_active and item.steps and _campaign_scope_matches(item, topic_id))
    if not item.is_active:
        reason = "campaign_inactive"
    elif not _campaign_scope_matches(item, topic_id):
        reason = "scope_not_allowed"
    elif not eligibility.eligible:
        reason = eligibility.reason
    elif not item.steps:
        reason = "step_missing"
    elif step is None:
        reason = "manual_test_completed"
    elif step.message_type not in {"static", "ai"}:
        reason = "step_invalid"
    elif step.message_type == "static":
        locale = await resolve_user_effective_locale(session, user)
        text = translate(
            f"followup_step.{step.id}.message_text",
            locale,
            source=step.message_text or "",
            fallback=step.message_text or "",
        )
        reason = "eligible" if text and text.strip() else "step_invalid"
    else:
        reason = "eligible"
    snapshot.update(
        {
            "dialogue_id": dialogue_id,
            "topic_id": topic_id,
            "eligibility": eligibility,
            "step": step,
            "step_index": step_index,
            "campaign_valid": campaign_valid,
            "can_send": reason == "eligible" and campaign_valid,
            "reason": reason,
        }
    )
    return snapshot


def _campaign_scope_matches(item: FollowupCampaign, topic_id: int) -> bool:
    if item.all_topics:
        return True
    if topic_id == 0:
        return item.include_main_dialogue
    return topic_id in {topic.id for topic in item.topics}


def _self_test_text(snapshot: dict, user_id: int) -> str:
    item = snapshot["campaign"]
    user = snapshot["user"]
    if item is None:
        return "Цепочка не найдена."
    identity = max_communication_name(user) if user is not None else "Не найден"
    identity = html.escape(identity)
    display_id = raw_max_user_id(user.id) if user is not None else user_id
    if user is None:
        return (
            "🧪 <b>Проверка цепочки на себе</b>\n\n"
            f"Пользователь: {identity} / ID {display_id}\n\n"
            "Итог:\n❌ Пользователь не найден в базе бота"
        )
    eligibility = snapshot["eligibility"]
    current_step = html.escape(eligibility.current_step or "не задан")
    topic = user.current_topic
    topic_text = "Основной диалог" if snapshot["topic_id"] == 0 else (
        topic.name if topic is not None else f"ID {snapshot['topic_id']}"
    )
    stage_status = "✅" if eligibility.stage_matches else "❌"
    metadata_status = "✅" if eligibility.metadata_matches else "❌"
    if eligibility.metadata_configured:
        metadata_text = _self_test_metadata_text(item)
    else:
        metadata_text = "не заданы"
    if eligibility.matched_stop_event:
        stop_text = f"❌ Событие остановки: {html.escape(eligibility.matched_stop_event)}"
    elif parse_followup_csv(item.stop_events):
        stop_text = "✅ События остановки: совпадений нет"
    else:
        stop_text = "✅ События остановки: не заданы"
    reason_labels = {
        "campaign_inactive": "цепочка выключена",
        "scope_not_allowed": "текущая тема не входит в область цепочки",
        "step_missing": "следующий шаг отсутствует",
        "step_invalid": "следующий шаг заполнен некорректно",
        "manual_test_completed": "ручная проверка завершена",
        "stage_not_allowed": "этап не подходит",
        "metadata_mismatch": "метаданные не подходят",
        "stop_event_found": "найдено событие остановки",
    }
    result = "✅ Цепочка сейчас может запуститься" if snapshot["can_send"] else (
        "❌ Цепочка сейчас не запустится\n"
        f"Причина: {reason_labels.get(snapshot['reason'], snapshot['reason'])}"
    )
    next_step = snapshot["step"]
    if next_step is None:
        if snapshot["reason"] == "manual_test_completed":
            next_text = "Ручная проверка завершена.\nВсе шаги уже отправлены."
        else:
            next_text = "Следующий шаг:\nнет доступного шага"
    else:
        kind = "AI" if next_step.message_type == "ai" else "static"
        preview = (next_step.ai_instruction if next_step.message_type == "ai" else next_step.message_text) or ""
        preview = " ".join(preview.split())
        if len(preview) > 120:
            preview = preview[:119] + "…"
        next_text = (
            f"Следующий шаг:\n#{snapshot['step_index'] + 1} · через {next_step.delay_minutes} мин · {kind}\n"
            f"{html.escape(preview or 'не задано')}"
        )
    return (
        "🧪 <b>Проверка цепочки на себе</b>\n\n"
        f"Пользователь: {identity} / ID {display_id}\n"
        f"Диалог: {snapshot['dialogue_id']}\n"
        f"Тема: {html.escape(topic_text)}\n"
        f"Этап: {current_step}\n\n"
        "Условия:\n"
        f"{stage_status} Этап: {_self_test_stage_text(item)}\n"
        f"{metadata_status} Метаданные: {metadata_text}\n"
        f"{stop_text}\n\n"
        f"Итог:\n{result}\n\n"
        f"{next_text}"
    )


def _self_test_index(snapshot, campaign_id: int) -> int:
    if snapshot is None or snapshot.state != "max_followup_self_test":
        return 0
    if snapshot.data.get("campaign_id") != campaign_id:
        return 0
    try:
        return max(0, int(snapshot.data.get("step_index", 0)))
    except (TypeError, ValueError):
        return 0


async def show_self_test(
    client: MaxApiClient,
    chat_id: int,
    user_id: int,
    campaign_id: int,
    states: StateStore | None = None,
) -> None:
    step_index = 0
    if states is not None:
        step_index = _self_test_index(await states.get(user_id), campaign_id)
        await states.set(user_id, chat_id, "max_followup_self_test", {"campaign_id": campaign_id, "step_index": step_index})
    async with async_session_maker() as session:
        snapshot = await _self_test_snapshot(session, campaign_id, user_id, step_index)
    can_send = snapshot["can_send"]
    rows = []
    if can_send:
        rows.append([callback_button("▶️ Отправить следующий шаг сейчас", f"admin_fu_self_test_send_{campaign_id}")])
    rows.extend([[callback_button("🔄 Проверить условия заново", f"admin_fu_self_test_{campaign_id}")], _back(f"admin_fu_campaign_{campaign_id}")])
    await client.send_message(
        chat_id=chat_id,
        text=_self_test_text(snapshot, user_id),
        attachments=inline_keyboard(rows),
    )


async def send_self_test(
    client: MaxApiClient,
    chat_id: int,
    user_id: int,
    campaign_id: int,
    states: StateStore | None = None,
) -> None:
    key = (user_id, campaign_id)
    existing = _max_followup_tests_inflight.get(key)
    if existing is not None:
        await existing.wait()
        await show_self_test(client, chat_id, user_id, campaign_id, states)
        return
    completed = asyncio.Event()
    _max_followup_tests_inflight[key] = completed
    try:
        step_index = 0
        if states is not None:
            step_index = _self_test_index(await states.get(user_id), campaign_id)
        async with async_session_maker() as session:
            snapshot = await _self_test_snapshot(session, campaign_id, user_id, step_index)
        item = snapshot["campaign"]
        user = snapshot["user"]
        step = snapshot["step"]
        if item is not None and user is not None and step is not None and snapshot["can_send"]:
            prepared = await prepare_followup_step(
                user=user,
                step=step,
                dialogue_id=snapshot["dialogue_id"],
                topic_id=snapshot["topic_id"],
            )
            await emit_followup_step(FollowupTransportRegistry(max_client=client), user=user, step=step, send_result=prepared)
            if states is not None:
                await states.set(user_id, chat_id, "max_followup_self_test", {"campaign_id": campaign_id, "step_index": step_index + 1})
    except Exception:
        await client.send_message(chat_id=chat_id, text="Не удалось отправить шаг. Проверьте условия и настройки транспорта.")
    finally:
        _max_followup_tests_inflight.pop(key, None)
        completed.set()
    await show_self_test(client, chat_id, user_id, campaign_id, states)
