# PR #57 Dialogue, Mailing, and Provider Hardening Design

## Goal

Complete the remaining MAX follow-up review fixes and add shared dialogue-history markers, canonical mailing response buttons, escaped response-button parsing, configurable dialogue-scoped metadata handling, Telegram reset confirmation copy, and correct KIE application-error reporting without changing runtime history sent to AI or creating another PR.

All work remains on `feat/max-followups` and PR #57. No merge or deployment is in scope.

## Existing reset-state inventory

The inventory below is the contract for the metadata setting. It records the state observed in the existing reset paths before adding the setting.

| State | Existing behavior at new-dialogue/reset entry | Setting decision |
| --- | --- | --- |
| `User.current_dialogue_id` | Incremented by `memory_mode.start_new_dialogue()` for MAX and by Telegram reset/topic-switch state handling. Topic memory may restore a previously mapped topic dialogue instead of incrementing. | Always starts or selects the new dialogue scope according to the existing memory mode. |
| `Message` rows | Telegram keeps old rows but all runtime history queries are scoped to the current dialogue/topic. MAX explicitly deletes rows for the previous dialogue during reset. | Old conversation messages are never copied or sent to AI. Existing platform deletion/query behavior remains unchanged. |
| `AutomationConversationState` | Current algorithm state is keyed by `(user_id, dialogue_id, topic_id)`; a new scope has no current row unless runtime lazily creates one. `current_state_json`, `current_step`, and `metadata_json` are not copied by current reset code. | `current_state_json`, `current_step`, and all temporary FSM-like algorithm state are reset. In preserve mode only the current scoped metadata dictionary is copied into the new scope. |
| `AutomationDialogueState` | Global-mode metadata is keyed by `(user_id, dialogue_id)` and is read only for the current dialogue. A new dialogue has no row unless runtime creates one. | In preserve mode copy only the current dialogue's `metadata_json` into the new dialogue row. Never copy algorithm step/current-state data. In reset mode start empty. |
| `AutomationMetadataRecord` | Append-only metadata history is retained and is filtered/displayed by scope where applicable. | Retain as audit/history in both modes; it is not runtime conversation context and is not the reset target. |
| `AutomationEvent` and `AutomationStepTransition` | Historical event/transition rows are retained; runtime processing is guarded by their existing scope and processed-state rules. | Retain historical rows. Do not resurrect pending work into the new dialogue. |
| `User.metadata_json` | `apply_service_data_blocks()` appends legacy/service DATA records here. It is historical/user metadata history, not the current prompt metadata dictionary. | Retain in both modes. It is outside the dialogue-scoped reset set and must not be deleted. |
| `CardSpreadState` and in-memory card state | Telegram reset/topic paths clear card state; it is temporary interaction state. MAX reset currently relies on the MAX flow and does not carry it into the new scope. | Clear on every new-dialogue/reset path; never preserve. |
| `TestSession` | Telegram reset explicitly deletes the current test session. | Clear on every new-dialogue/reset path; never preserve. |
| FSM/`StateStore` input state | Reset confirmation keys are consumed/cleared by the current flows, but persisted MAX follow-up input states can otherwise survive navigation or admin revocation. | Centralized invalidation clears transient state before the new dialogue is committed. Post-reset onboarding/disclaimer state is written only after the reset and is scoped to the new dialogue. |
| Pending AI/background work | Existing generation/dialogue/topic guards prevent old work from applying to a new scope, but stale persisted/in-memory input state must be invalidated. | Do not transfer or await old work. Scope/generation guards remain authoritative; transient input state is cleared. |
| Profile/account fields | User identity, subscription, referral, language, and other account/profile fields are not reset. | Preserve regardless of setting. |
| Topic navigation mapping | `UserTopicState` is updated/restored by the configured memory mode. | Preserve existing memory-mode semantics; it is navigation state, not metadata reset behavior. |

The new additive `AIConfig.metadata_reset_mode` setting has values `reset` and `preserve`, defaults to `reset`, and is shared by Telegram and MAX. `reset` preserves the current production result: the new scope has empty dialogue-scoped runtime metadata. `preserve` copies only the active scoped metadata dictionary into the new dialogue, while conversation history, current algorithm state, temporary input/FSM state, pending work, and unrelated profile/account data follow the rules above.

## Architecture

### Shared export serialization

`dialogue_history_export.py` provides one human-readable serializer that receives already ordered records containing `dialogue_id` and rendered lines. It emits `--- **Начало нового диалога №N** ---` before the first block and at each dialogue-id change. Telegram/MAX single and mass TXT exports use it. JSON exports keep their existing schema and are not altered solely for this feature.

### Shared response-button parsing and mailing delivery

`response_buttons.py` remains the only parser and `ResponseButton` remains the platform-neutral representation. A narrow line normalizer accepts canonical Markdown, ordinary bullets, and escaped Markdown only when the complete line is a valid standalone action/URL declaration. It never globally unescapes response text. Mailing preview and delivery parse the canonical stored text at the boundary, render clean visible text, and create Telegram/MAX keyboard payloads with existing renderers and `ai_btn:` action semantics. The database stores canonical text/declarations, never platform keyboard objects.

### Dialogue metadata reset

The reset service captures the current scoped metadata before incrementing/selecting the new dialogue. It creates or seeds only the new scoped metadata row when `metadata_reset_mode=preserve`; it leaves the new scope empty when `reset`. The helper also invalidates transient input/test/card state. Telegram and MAX call the same policy helper, while preserving their existing message deletion/history-query behavior and memory-mode topic mapping.

### MAX follow-up hardening

Message-based topic selection awaits the same task boundary as callback selection, so activity finalization runs after the topic/dialogue commit. MAX follow-up message input is routed through one guard that rechecks current admin privilege, validates persisted campaign/step ownership, and invalidates stale state. Metadata operator callbacks must match the campaign stored in the pending state. Static follow-up authoring uses the same translation/readiness coordination and HTML canonicalization as Telegram. Step order allocation locks the campaign row and uses `max(sort_order)+1`. MAX self-test locale resolution uses MAX delivery semantics, and static follow-up output preserves the same formatted/HTML content across both platforms.

### KIE errors and request contract

Both Telegram and MAX KIE validators inspect the decoded application envelope regardless of HTTP status. A non-success KIE code becomes a provider rejection with provider/model/code/message and separate HTTP status; it is never converted into a timeout or network error. Existing `provider_models.py`/`kie_chat.py` routing for `gemini-3-flash` is verified against the repository's encoded contract. A dry-run probe prints a sanitized endpoint/payload and requires an explicit opt-in for any live request; CI never performs a live call.

## UI and verification contract

Every changed Admin/reset/mailing screen is tested through the real Telegram Dispatcher or `MaxBotApplication`, through actual keyboard construction and payload validation, with cancel/back/reopen behavior. Mailing action proof presses the actual recipient-rendered button and follows the normal `ai_btn:` callback path; URL buttons remain URL payloads. Tests cover TG/MAX parity, cross-platform isolation, existing follow-up A→Я→A journeys, concurrent ordering, stale/revoked-admin protection, KIE Admin log filters/detail/export, compileall, diff checks, focused suites, and the full suite.

## Schema policy

Only additive schema changes are allowed. The expected schema addition is `AIConfig.metadata_reset_mode` with a backward-compatible `reset` default. No fake `Message` rows or platform-specific mailing keyboard columns are added.
