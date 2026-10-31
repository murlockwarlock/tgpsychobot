# Настройки генерации моделей

Настройки генерации хранятся в `ai_model_settings` по ключу `provider + model + channel`. Канал `chat` используется сейчас; схема допускает отдельные значения для других каналов. При инициализации создаётся строка активной модели, остальные строки создаются при первом редактировании. Старые поля `ai_config.max_output_tokens`, `ai_config.temperature` и `ai_config.deepseek_thinking_enabled` остаются для безопасного rollback и используются только для первичного заполнения scoped-настройки на legacy-базе. После появления scoped-строк новая модель получает собственные значения Auto, а не наследует старый глобальный override.

## Владение настройками

`Настройки ИИ` содержит активного провайдера, системные prompt-блоки, логи и вход в `Провайдеры и модели`. Max tokens, Temperature и Reasoning отображаются только в карточке активной модели. Карточка и picker строятся по capability registry для конкретной пары `provider + model`; Reasoning не является общим toggle OpenRouter и не является DeepSeek-only UI concept.

| Провайдер | Max tokens | Temperature | Reasoning |
| --- | --- | --- | --- |
| OpenAI | да, Auto без wire override | кроме gpt-5.6 | нет |
| Gemini | да, Auto без wire override | кроме Gemini 3.6/3.7 | нет |
| DeepSeek | да, Auto без wire override | не действует при Auto/Low/High/Max | Auto/Выкл/Low/High/Max |
| Claude | да, обязательный API limit | только модели с sampling | нет |
| KIE | да, обязательный API limit | да | нет |
| OpenRouter | да, по curated model metadata | по model-level contract | только reasoning-capable curated models |
| Perplexity | да, если preset поддерживает | нет | нет |
| Deepgram | не chat-параметр | не chat-параметр | нет |

## Legacy mapping

При первом создании scoped chat-настройки активной модели:

- `max_output_tokens = NULL` сохраняется как `NULL` и означает `Auto`;
- числовой `max_output_tokens` переносится как явный override;
- `deepseek_thinking_enabled = NULL` становится `Auto`;
- `TRUE` становится `High`;
- `FALSE` становится `Выкл`;
- legacy поля не удаляются и не перезаписываются.

При инициализации и при разрешении scoped-настройки числовой persisted override нормализуется к текущему лимиту модели. Например, старый `32768` для OpenRouter DeepSeek V4 Pro становится `16384`; это сохраняется application initialization path и не требует ручного сброса в Admin. Runtime дополнительно применяет тот же fail-safe clamp перед wire serialization.

`Auto` не вычисляет внутренний лимит для отправки. Для провайдеров с обязательным output budget runtime использует их безопасный обязательный limit; для остальных поле не отправляется. Fallback-разговор разрешает настройки по фактической паре fallback provider/model.

Для DeepSeek Chat Completions `Auto` не добавляет thinking-параметры, `Выкл` отправляет `extra_body.thinking.type=disabled`, а `Low`, `High` и `Max` отправляют `reasoning_effort` вместе с `extra_body.thinking.type=enabled`. Это direct DeepSeek contract и он не переносится в OpenRouter.

## OpenRouter Reasoning

OpenRouter capability хранится в `OpenRouterModelSpec.reasoning`. В ней фиксируются документированные modes, наличие и значение `supported_efforts`, `mandatory`, default metadata, возможность отключения, поддержка temperature и `reasoning.exclude`. `supports_max_tokens` reasoning object не заявляется, если live metadata его не содержит; это отдельное поле от outer `max_tokens`. Model picker не проверяет имена с prefix heuristic.

Для подтверждённых curated моделей текущий набор такой:

| Модель | Режимы Admin | OpenRouter mapping |
| --- | --- | --- |
| `qwen/qwen3.7-plus` | `Auto / Выкл / Вкл` | `reasoning` с `enabled`; effort не используется |
| `qwen/qwen3.8-27b` | `Auto / Выкл / Low / Medium / Max` | `Low`, `Medium`, `Max → xhigh`, `Выкл → none` |
| `deepseek/deepseek-v4-flash` | `Auto / Выкл / High / Max` | `High`, `Max → xhigh`, `Выкл → none` |
| `deepseek/deepseek-v4-pro` | `Auto / Выкл / High / Max` | `High`, `Max → xhigh`, `Выкл → none` |
| `deepseek/deepseek-v4.1-flash` | `Auto / Выкл / Low / High / Max` | documented effort values; `Max → max` |

Для Qwen3.7 отсутствие ключа `supported_efforts` означает toggle-only contract. Это не эквивалентно JSON `supported_efforts: null`: verifier сохраняет presence и отмечает такой drift, не добавляя неподтверждённые effort-кнопки.

`Auto` означает отсутствие `reasoning` override. Explicit mode отправляет документированный OpenRouter `reasoning` object и `exclude: true`, чтобы reasoning trace не возвращался в ответ. Для explicit override добавляется `provider.require_parameters=true`; это гарантирует маршрутизацию только на endpoints, поддерживающие фактический request contract. Auto не включает reasoning-specific routing restriction.

`output_limit` curated-модели — portable safe floor для всех request contracts, которые приложение может отправить. Для `qwen/qwen3.8-27b` это `32768`: live `top_provider.max_completion_tokens` может быть `235929`, но endpoint matrix содержит активный совместимый endpoint с лимитом `32768`. Catalog verifier сравнивает отдельно top-provider limit, endpoint-safe floor и static application limit.

`moonshotai/kimi-k2.6` также имеет `configured_output_limit=16384`: текущая live endpoint matrix содержит активный endpoint с минимумом `16384`, поэтому прежний `235929` не был routing-independent safe floor. Это не reasoning UI change, а обязательная защита общего endpoint-safe contract.

Reasoning tokens входят в output budget. Ответ `finish_reason=length` с пустым content классифицируется как `output_budget_exhausted`, сохраняет только finish reason и безопасный usage summary и не запускает платный retry. `reasoning`, `reasoning_details` и аналогичные скрытые поля не попадают в user-facing text, историю, diagnostics или fallback prompt.

## Vision и STT

Capability registry также управляет выбором провайдера для audio, vision, image generation и image edit. Deepgram присутствует только в transcription picker. DeepSeek Flash (`deepseek-flash`) присутствует в vision picker; DeepSeek Pro туда не входит. Telegram и MAX используют один и тот же OpenAI-compatible image payload с inline `data:image/...;base64` только на время запроса. В AILog image data редактируется до сохранения.

API keys остаются в существующих per-bot `AIConfig` полях и маскируются в Admin. `AIModelSettings` не содержит ключей и не меняет secret storage.

В Telegram Admin ключи и карточки provider/model отображаются четырьмя рядами по две кнопки; восьмая карточка — Deepgram Nova-3 для транскрибации. Audio, Vision, генерация и редактирование изображений открывают отдельные capability-picker экраны, построенные из общего каталога. Старые callback-и циклического выбора оставлены только как безопасный redirect в picker.
