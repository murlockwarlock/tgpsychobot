# Telegram Admin navigation matrix

| Screen | Action | Next screen | Object | Locale | Page | FSM state |
| --- | --- | --- | --- | --- | --- | --- |
| Content list | Open item | Content card | Same Content key | RU by default | Current page | Clear |
| Content list page N | Open item | Content card | Same Content key | RU | N | Clear |
| Content card / locale L | Select locale | Same Content card | Same Content key | L | Same page | Clear |
| Content card / locale L | Изменить: контент | Full Content editor | Same Content key | L | Same page in state | `edit_content` |
| Full Content editor / locale L | Send text/media | Same editor | Same Content key | L | Same page in state | `edit_content` |
| Full Content editor / locale L | Save and exit | Content card | Same Content key | L | Parent page | Clear |
| Full Content editor / locale L | Cancel | Content card | Same Content key | L | Parent page | Clear |
| Content card | Toggle visibility | Same Content card | Same Content key | L | Same page | Clear |
| Content card | Delete | Dependency report or confirmation | Same Content key | L | Same page | Clear |
| Content card | К списку | Content list | Same Content key context | RU list view | Parent page | Clear |
| Content list | Назад | Direct parent Admin section | None | RU | Parent context | Clear |
| New Content | Save title | Content card | New stable key | RU | Page 0 | Clear |
| Content card | Изменить название кнопки | Single-field editor | Same Content key | Current locale | Same page in state | `value` |
| Topic/Plan/Referral/Media card | Select locale | Same object card | Same stable ID | Selected locale | Parent page | Clear |
| Any single-field editor | Cancel | Same object card | Same stable ID | Selected locale | Parent page | Clear |

No callback in the canonical Content path returns to Admin root, another object, or an old Content editor. Legacy callbacks remain only as compatibility shims for already-sent keyboards and are not emitted by current screens.
