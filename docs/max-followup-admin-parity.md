# MAX follow-up Admin parity

Telegram remains the canonical follow-up authoring contract. MAX uses the same
`FollowupCampaign` and `FollowupStep` rows and mirrors the following screens.

| Telegram screen/action | MAX screen/action | Stored result | Immediate parent |
| --- | --- | --- | --- |
| `💬 Догоняющие сообщения` → chain | `💬 Догоняющие сообщения` → chain | Same campaign id | Campaign list |
| `Статус` | `Статус` | `is_active` | Campaign card |
| `Переименовать` | `Переименовать` | `name` | Campaign card |
| `Темы` | `Темы` | `all_topics`, `include_main_dialogue`, association rows | Campaign card |
| `Шаги` | `Шаги` | Same ordered `FollowupStep` rows | Campaign card |
| `Условия` | `Условия` | Stage, metadata and stop-event fields | Campaign card |
| `Тихие часы` | `Тихие часы` | Timezone and minute fields | Campaign card |
| `Случайная задержка` | `Случайная задержка` | Jitter fields | Campaign card |
| `Проверить на себе` | `Проверить на себе` | Eligibility is evaluated against the same user and campaign | Campaign card |
| `Обычный текст` / `Сгенерировать через AI` | Same labels | `message_type`, text/instruction, delay and sort order | Steps |
| `Редактировать` | `Редактировать` | Same step row | Step card |
| `Текст сообщения` → `Изменить: Сообщение` | Same labels with MAX inline callbacks | Canonical static message text | Step text card → step card |
| `Удалить` campaign | `Удалить` with confirmation | Same cascade/delete semantics | Campaign card → campaign list |
| `Удалить` step | `Удалить` with the same sent/claimed protection | Same ordered step rows | Step card → steps |

Text, labels, button order, validation messages, and parent destinations are
defined from the shared Russian contract constants in `followup_admin_contract.py`.
Only Telegram InlineKeyboardMarkup versus MAX inline-keyboard attachment
encoding differs. Follow-up campaign and step lists are not paginated in the
canonical Telegram UI, so MAX intentionally has no pagination controls for
these lists. The MAX Admin entry point is a direct shortcut because the current
MAX Admin root has no separate automation menu; once inside the follow-up
screens, every action and parent destination is the same.

| Contract | Result |
| --- | --- |
| Shared Russian screen text | PASS |
| Button labels and order | PASS |
| Navigation and immediate parents | PASS |
| Mutation semantics and shared rows | PASS |
| Save / Cancel / Back | PASS |
| Delete confirmation and sent/claimed protection | PASS |
| Pagination | PASS — not applicable; both lists are single-page |

The self-test screen uses the same stage, metadata and stop-event diagnostics,
reason labels, next-step preview, and manual step progression on both platforms.
MAX displays the raw MAX user id in the same `Пользователь … / ID …` position;
the internal database identity remains shared with Telegram scheduling. Follow-up
static text is intentionally a Russian shared business field; it is not part of
the multilingual authoring whitelist on either platform.
