from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from response_buttons import ResponseButton, build_action_callback_data


def telegram_response_buttons_markup(
    rows: list[list[ResponseButton]],
) -> InlineKeyboardMarkup | None:
    if not rows:
        return None
    action_counts: dict[str, int] = {}
    for row in rows:
        for button in row:
            if button.kind == "action":
                action_counts[button.value] = action_counts.get(button.value, 0) + 1

    keyboard_rows: list[list[InlineKeyboardButton]] = []
    action_button_index = 0
    for row in rows:
        keyboard_row = []
        for button in row:
            if button.kind == "url":
                keyboard_row.append(InlineKeyboardButton(text=button.text, url=button.value))
            else:
                callback_data = build_action_callback_data(
                    button.value,
                    action_button_index if action_counts[button.value] > 1 else None,
                )
                keyboard_row.append(
                    InlineKeyboardButton(text=button.text, callback_data=callback_data)
                )
                action_button_index += 1
        if keyboard_row:
            keyboard_rows.append(keyboard_row)
    return InlineKeyboardMarkup(inline_keyboard=keyboard_rows) if keyboard_rows else None
