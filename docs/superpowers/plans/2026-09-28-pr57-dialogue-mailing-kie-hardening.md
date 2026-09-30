# PR #57 Dialogue, Mailing, and Provider Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining MAX follow-up correctness fixes and deliver shared dialogue markers, canonical mailing buttons, escaped response-button parsing, configurable scoped metadata reset, Telegram reset UX, and KIE application-error visibility in the existing PR #57 branch.

**Architecture:** Keep dialogue serialization, response-button parsing, metadata-reset policy, and follow-up authoring coordination platform-neutral. Telegram and MAX adapters render their own validated payloads at the transport boundary while sharing canonical inputs and stable `ai_btn:` action semantics. Preserve all existing follow-up scheduler, activity, generation, ownership, and cross-platform isolation guarantees.

**Tech Stack:** Python 3.12, SQLAlchemy async, aiogram/Pydantic, MAX API client, pytest/pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-28-pr57-dialogue-mailing-kie-hardening.md`

## Global Constraints

- All work stays on `feat/max-followups` and PR #57; do not create a PR, merge, or deploy.
- The exact human-readable marker is `--- **Начало нового диалога №N** ---`.
- JSON export schemas and provider runtime conversation history remain unchanged unless a required additive field is proven necessary.
- Mailing buttons use the existing `extract_response_buttons()`, `ResponseButton`, renderers, and `ai_btn:` callback semantics.
- Escaped button normalization is line-scoped; never globally unescape LLM responses.
- `AIConfig.metadata_reset_mode` is additive and defaults to current production behavior (`reset`).
- Preserve mode copies only current scoped metadata; it never copies messages, current algorithm state, FSM/input state, pending work, or unrelated profile/account data.
- KIE HTTP 200 application rejection is `provider_rejection`, with provider code and transport status retained separately.
- No live KIE calls in CI; no production writes or deployment.

## Review Focus

- Dialogue boundary markers must be emitted before the first block and only when the ordered dialogue id changes; test non-consecutive ids and all four export routes.
- Canonical mailing buttons must survive preview, persisted retry/history, real recipient delivery, actual callback press, and URL rendering on both platforms.
- Escaped Markdown must be removed only for valid standalone button lines and must not alter prose, code, math, or ordinary bullets.
- Preserve metadata must copy only intended runtime metadata while resetting history/current step/transient state; reset mode must remain the default and keep profile/history audit data.
- Persisted MAX input state must be rejected after admin revocation, campaign/step mismatch, or navigation away; stale state must be cleared centrally.
- Concurrent step creation, translation/readiness cache updates, and formatted HTML output must remain consistent across TG/MAX.
- KIE envelope failures must be visible as provider errors in Admin logs even when HTTP transport status is 200.

## Task 1: Shared export markers and robust response-button parser

**Files:**
- Create: `dialogue_history_export.py`
- Modify: `response_buttons.py`
- Test: `tests/test_dialogue_history_export.py`, `tests/test_response_buttons.py`

**Interfaces:**
- Produce `serialize_human_dialogue_history(records)` for Telegram/MAX TXT exporters.
- Preserve `extract_response_buttons(text) -> tuple[str, list[list[ResponseButton]]]` as the single parser API.

- [ ] Write failing unit tests for one, two, three-plus, and non-consecutive dialogue ids, first-block markers, and chronological preservation.
- [ ] Write failing parser tests for the exact escaped three-button response, escaped URL declarations, ordinary bullet prose, code, math backslashes, and malformed declarations.
- [ ] Run the focused tests and confirm the new expectations fail before implementation.
- [ ] Implement the shared serializer and line-scoped candidate normalization.
- [ ] Run the focused tests and confirm they pass.
- [ ] Add integration assertions for Telegram/MAX single and mass TXT exporter inputs without changing JSON output.

## Task 2: Mailing canonical buttons and real delivery interaction

**Files:**
- Modify: `mailing_utils.py`, `handlers.py`, `max_messenger_bot/services/admin_mailing.py`, `max_messenger_bot/app.py`, `max_messenger_bot/keyboards.py`
- Test: `tests/test_mailing_buttons.py`, existing Telegram/MAX Admin mailing journey tests

**Interfaces:**
- Consume `extract_response_buttons` and `ResponseButton` from Task 1.
- Store canonical mailing text only; render Telegram/MAX payloads at send time.

- [ ] Add failing tests for action, URL, multiple, no-button, and malformed mailing text through preview and persistence.
- [ ] Add failing transport-boundary tests that inspect the actual recipient keyboard and press the returned action callback through Telegram Dispatcher/MAX application routing; assert URL buttons remain links.
- [ ] Run the focused mailing tests and confirm failure.
- [ ] Implement shared clean-preview/delivery parsing, Telegram markup construction, MAX attachment construction, and canonical text persistence/history/retry handling.
- [ ] Run real Admin UI journey tests for Telegram and MAX, including actual callback action semantics and cross-platform payload validation.

## Task 3: Follow-up formatting, coordination, ownership, and concurrency

**Files:**
- Modify: `followups.py`, `automation_admin.py`, `max_messenger_bot/services/admin_followups.py`, `max_messenger_bot/app.py`, `translation_pack_manager.py` or a focused shared follow-up admin helper
- Test: `tests/test_followups.py`, `tests/test_followup_admin.py`, `tests/test_max_followups.py`

**Interfaces:**
- Produce shared step-order allocation and coordinated static-step mutation helpers used by TG/MAX.
- Keep canonical static HTML source compatible with both platform renderers.

- [ ] Add failing tests for concurrent step adds, static step add/edit/save readiness, formatted HTML/button parity, metadata operator campaign mismatch, and revoked-admin persisted inputs.
- [ ] Add failing message-path activity tests for `/start topic_<id>` and visible topic-name selection with delayed topic commits.
- [ ] Add failing stale-input tests for navigation-away and missing campaign/step ownership.
- [ ] Run focused tests and confirm failure.
- [ ] Implement row-locked `max(sort_order)+1`, shared translation/readiness commit path, canonical formatted static delivery, campaign ownership validation, central stale-state invalidation, and privilege re-check.
- [ ] Await message-based topic selection tasks so follow-up activity finalizes after committed scope changes.
- [ ] Run the existing follow-up regression suite plus new tests.

## Task 4: Metadata reset setting and shared runtime policy

**Files:**
- Modify: `database.py`, `memory_mode.py`, Telegram reset/runtime handlers, `max_messenger_bot/services/common.py`, canonical common AI/general Admin settings and shared labels/keyboards
- Test: `tests/test_metadata_reset_setting.py`, `tests/test_memory_mode.py`, `tests/test_max_followups.py`, Telegram Admin/runtime journey tests

**Interfaces:**
- Produce `metadata_reset_mode` values `reset`/`preserve` with additive migration and common TG/MAX Admin controls.
- Produce one reset policy helper that captures/copies only current scoped metadata and invalidates transient state.

- [ ] Add failing schema/default and Admin save/reopen tests for both platforms.
- [ ] Add failing runtime tests proving preserve creates a new dialogue with empty history/context but available intended metadata, and reset keeps current behavior with empty scoped metadata.
- [ ] Add failing tests proving old messages, `current_state_json`, `current_step`, `TestSession`, card state, stale FSM/StateStore state, and pending input are not carried, while profile fields and historical `user.metadata_json` remain.
- [ ] Run focused tests and confirm failure.
- [ ] Implement migration, shared setting UI, reset policy, and platform integrations.
- [ ] Run TG/MAX real Admin and runtime journeys, including reopen/back/cancel.

## Task 5: Telegram reset confirmation UX

**Files:**
- Modify: Telegram translations/keyboards/reset handlers
- Test: `tests/test_telegram_reset_confirmation.py`, existing Dispatcher journey tests

**Interfaces:**
- Preserve token/dialogue/topic stale checks and main/topic switch semantics.

- [ ] Add failing validation tests for exact main/topic text and labels, including safe bold topic rendering.
- [ ] Add failing real Dispatcher journeys for main cancel/yes and topic cancel/main-navigation/yes, checking DB mutation and resulting scope.
- [ ] Run focused tests and confirm failure.
- [ ] Implement only the requested Telegram confirmation copy/buttons and preserve current metadata policy behavior.
- [ ] Validate actual aiogram/Pydantic outgoing methods and run the journey suite.

## Task 6: KIE application errors, contract verification, and Admin visibility

**Files:**
- Modify: `ai_integration.py`, `max_messenger_bot/ai.py`, `error_reporting.py` or log-record construction as needed
- Create: `scripts/probe_kie_chat.py`
- Test: `tests/test_kie_chat_models.py`, `tests/test_kie_application_errors.py`, Admin AI log tests

**Interfaces:**
- Keep `provider_models.py`/`kie_chat.py` request routing unchanged unless contract inspection identifies a concrete bug.
- Classify non-success KIE envelopes as `provider_rejection` with `provider`, `model`, `provider_code`, `provider_message`, and separate `http_status`.

- [ ] Add failing exact-envelope tests for Telegram and MAX with HTTP 200 and KIE code 422.
- [ ] Add failing Admin filter/detail/export visibility tests.
- [ ] Add request-contract tests for `gemini-3-flash` and a sanitized dry-run probe test; do not perform live calls.
- [ ] Run focused tests and confirm failure.
- [ ] Implement shared/consistent envelope classification and Admin record propagation.
- [ ] Run focused KIE and Admin log tests.

## Task 7: Full regression and delivery gate

**Files:**
- Modify only files required by failing regression evidence.
- Test: all focused suites and full repository suite.

- [ ] Run the MAX follow-up blocker regression matrix: message topic-switch finalization, metadata ownership, stale invalidation, revoked admin, coordination/concurrent ordering, A→Я→A, callback classification, one scheduler, Telegram regression, and cross-platform isolation.
- [ ] Run all new TG/MAX journey proofs from Tasks 1–6.
- [ ] Run `python -m compileall` over the project.
- [ ] Run `git diff --check`, inspect staged diff/stat, and ensure unrelated pre-existing untracked files remain unstaged.
- [ ] Run the full suite and record exact pass/fail results.
- [ ] Commit the implementation with the repository's established message style, push only `feat/max-followups`, verify PR #57 remains open/unmerged, and report the exact new HEAD. Do not deploy.
