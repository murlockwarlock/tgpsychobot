# Настройки генерации моделей

Настройки генерации хранятся в `ai_model_settings` по ключу `provider + model + channel`. Канал `chat` используется сейчас; схема допускает отдельные значения для других каналов. При инициализации создаётся строка активной модели, остальные строки создаются при первом редактировании. Старые поля `ai_config.max_output_tokens`, `ai_config.temperature` и `ai_config.deepseek_thinking_enabled` остаются для безопасного rollback и используются только для первичного заполнения scoped-настройки на legacy-базе. После появления scoped-строк новая модель получает собственные значения Auto, а не наследует старый глобальный override.

## Владение настройками

`Настройки ИИ` содержит активного провайдера, системные prompt-блоки, логи и вход в `Провайдеры и модели`. Max tokens, Temperature и DeepSeek Reasoning отображаются только в карточке активной модели. Карточка строится по capability registry.

| Провайдер | Max tokens | Temperature | Reasoning |
| --- | --- | --- | --- |
| OpenAI | да, Auto без wire override | кроме gpt-5.6 | нет |
| Gemini | да, Auto без wire override | кроме Gemini 3.6/3.7 | нет |
| DeepSeek | да, Auto без wire override | не действует при Auto/Low/High/Max | Auto/Выкл/Low/High/Max |
| Claude | да, обязательный API limit | только модели с sampling | нет |
| KIE | да, обязательный API limit | да | нет |
| OpenRouter | да, по curated model metadata | консервативно по routed model | нет |
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

`Auto` не вычисляет внутренний лимит для отправки. Для провайдеров с обязательным output budget runtime использует их безопасный обязательный limit; для остальных поле не отправляется. Fallback-разговор разрешает настройки по фактической паре fallback provider/model.

Для DeepSeek Chat Completions `Auto` не добавляет thinking-параметры, `Выкл` отправляет `extra_body.thinking.type=disabled`, а `Low`, `High` и `Max` отправляют `reasoning_effort` вместе с `extra_body.thinking.type=enabled`. В режиме reasoning temperature не отправляется.

## Vision и STT

Capability registry также управляет выбором провайдера для audio, vision, image generation и image edit. Deepgram присутствует только в transcription picker. DeepSeek Flash (`deepseek-flash`) присутствует в vision picker; DeepSeek Pro туда не входит. Telegram и MAX используют один и тот же OpenAI-compatible image payload с inline `data:image/...;base64` только на время запроса. В AILog image data редактируется до сохранения.

API keys остаются в существующих per-bot `AIConfig` полях и маскируются в Admin. `AIModelSettings` не содержит ключей и не меняет secret storage.

В Telegram Admin ключи и карточки provider/model отображаются четырьмя рядами по две кнопки; восьмая карточка — Deepgram Nova-3 для транскрибации. Audio, Vision, генерация и редактирование изображений открывают отдельные capability-picker экраны, построенные из общего каталога. Старые callback-и циклического выбора оставлены только как безопасный redirect в picker.
