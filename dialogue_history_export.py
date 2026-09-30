from collections.abc import Iterable


DIALOGUE_MARKER_TEMPLATE = "--- **Начало нового диалога №{dialogue_id}** ---"


def serialize_human_dialogue_history(records: Iterable[tuple[int, str]]) -> str:
    lines: list[str] = []
    current_dialogue_id = object()
    for dialogue_id, rendered_line in records:
        if dialogue_id != current_dialogue_id:
            lines.append(DIALOGUE_MARKER_TEMPLATE.format(dialogue_id=dialogue_id))
            current_dialogue_id = dialogue_id
        lines.append(rendered_line.rstrip("\n"))
    return "\n".join(lines)
