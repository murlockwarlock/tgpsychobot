from __future__ import annotations

import re

FOLLOWUP_CAMPAIGN_EXPLANATION = (
    "Цепочка начинается после действия пользователя и запускается заново после его нового сообщения или нажатия кнопки. "
    "При смене темы или создании нового диалога старая цепочка отменяется. "
    "Сообщения не отправляются в тихие часы."
)
FOLLOWUP_CAMPAIGN_DETAIL_INTRO = "Цепочка отправляет несколько напоминаний, пока пользователь молчит."
FOLLOWUP_STEPS_EXPLANATION = (
    "Первое время считается от последнего действия пользователя, следующие — от предыдущего сообщения. "
    "Новое действие пользователя начинает цепочку заново.\n\n"
    "Шаг «Обычный текст» отправляет ваш текст. Шаг «Сгенерировать через AI» передаёт AI вашу инструкцию "
    "и текущий диалог. Нажмите на шаг, чтобы открыть его детали."
)
FOLLOWUP_STAGE_LABELS = {
    "all": "На всех этапах",
    "selected": "На выбранных этапах",
    "all_except": "На всех этапах кроме",
}
FOLLOWUP_METADATA_LABELS = {
    "equals": "=",
    "not_equals": "!=",
    "contains": "содержит",
}


def parse_followup_step_input(raw_text: str | None) -> tuple[int, str] | None:
    first, separator, body = (raw_text or "").partition("\n")
    if not separator or not first.strip().isdigit() or not body.strip():
        return None
    delay = int(first.strip())
    if not 1 <= delay <= 525600:
        return None
    return delay, body.strip()


def classify_followup_callback(payload: str) -> str | None:
    if payload in {"admin_followups", "admin_fu_list", "admin_fu_add"}:
        return "navigation"
    if re.fullmatch(r"admin_fu_campaign_\d+", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_(topics|conditions|stage|metadata|steps|quiet|jitter|self_test)_\d+", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_metadata_operator_edit_\d+", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_step_\d+_\d+", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_step_text_\d+(?:_ru)?", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_(rename|stage|metadata|stops|quiet|jitter)_\d+", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_step_add_\d+_(?:static|ai)", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_step_edit_\d+_\d+", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_step_text_edit_\d+_ru", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_step_delete_\d+_\d+", payload):
        return "destructive"
    if re.fullmatch(r"admin_fu_delete_ask_\d+", payload):
        return "navigation"
    if re.fullmatch(r"admin_fu_toggle_\d+", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_scope_all_\d+", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_scope_main_\d+", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_scope_topic_\d+_\d+", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_stage_mode_\d+_(?:all|selected|all_except)", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_metadata_op_\d+_(?:equals|not_equals|contains)", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_metadata_clear_\d+", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_stops_clear_\d+", payload):
        return "mutation"
    if re.fullmatch(r"admin_fu_delete_yes_\d+", payload):
        return "destructive"
    if re.fullmatch(r"admin_fu_step_delete_yes_\d+_\d+", payload):
        return "destructive"
    if re.fullmatch(r"admin_fu_self_test_send_\d+", payload):
        return "external"
    return None
