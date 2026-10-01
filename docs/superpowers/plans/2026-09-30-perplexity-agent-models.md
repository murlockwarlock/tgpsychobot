# Implementation Plan: Perplexity Agent API Expansion (Presets & Direct Models) [REVISED]

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Branch:** `feat/perplexity-agent-models`  
**Repository:** `murlockwarlock/tgpsychobot`  
**Baseline:** PR #57 merge commit (`d9d06afc2bf02517c976783d0bea9365c80700fc`)  
**Status:** Revised plan incorporating all mandatory corrections. Code NOT modified. Waiting for approval.

---

## Global Constraints

- **STRICTLY NO CODE CHANGES YET**: Only implementation planning in this phase.
- **BRANCH**: Work strictly in `feat/perplexity-agent-models`. Never touch `main` or merged PR #57.
- **NO DEPLOY / NO PRODUCTION**: No production deployment, no database migration scripts on production.
- **NO PAID CALLS IN PLANNING**: Live probe matrix is defined for later manual execution; no live API calls are run now.
- **DO NOT TOUCH UNRELATED MODULES**: Do not modify OpenRouter, Deepgram, KIE, response button parser, followups, or mailing logic.
- **SINGLE PROVIDER**: Do not create a new provider ID or parallel settings table; expand existing `PROVIDER_PERPLEXITY = "perplexity"`.

---

## A. Corrections Made to Previous Plan

1. **Preset + Model Contract**: Corrected false claim of mutual exclusivity. Agent API supports `preset="low"` + `model="anthropic/claude-sonnet-4-6"` where `model` overrides the underlying preset model while retaining other preset defaults. UI continues to separate Presets vs Direct Models in V1, but the shared adapter `build_perplexity_payload()` supports both `preset` and `model` arguments without architectural restriction.
2. **Preset Tools Semantics**: Corrected false claim that custom tools wipe the entire preset tool definition. In Agent API, tools merge *per tool* (e.g. passing `web_search` options overrides preset `web_search` options, leaving other preset tools intact). For our canonical preset execution, we deliberately omit `tools` entirely to let Perplexity run its dynamic preset-managed tools without local interference.
3. **Direct Model Tool / Search Semantics**: Corrected false claim that direct models have web search disabled by default. The official Agent API Models doc shows direct model requests (e.g. `openai/gpt-5.6-sol`) without explicit `tools` yielding web search citations and tool results. In V1, we do NOT force our own tools and do NOT attempt to disable Perplexity defaults; we send the documented minimal payload and follow Agent API default semantics.
4. **GET /v1/models Auth**: Official documentation demonstrates `GET https://api.perplexity.ai/v1/models` without `Authorization` header. Catalog fetch is public and does NOT require `perplexity_api_key`. If upstream later enforces auth, it will be treated as a catalog fetch failure and fall back safely without crashing Admin.
5. **Compact Telegram Callbacks**: Removed raw model IDs in callbacks (e.g. `ai_set_model:perplexity:<model_id>`). All Telegram model callbacks reuse the project's compact SHA-256 digest architecture (`build_telegram_model_callback_data` and `resolve_telegram_model_callback`), strictly guaranteeing `<= 64 bytes`.
6. **No Shared Disk File in V1**: Removed `.cache/perplexity_models_cache.json`. In production (18 PM2 processes), a shared disk file introduces multi-process locking and deploy friction. V1 uses a bundled static snapshot + process-local in-memory cache (1h TTL) + live unauthenticated `GET /v1/models`. Zero disk writes to the repo tree.
7. **Model Availability & Negative Authority**: A fresh successful `GET /v1/models` is authoritative (positive and negative). If a stored direct model is absent from a fresh live catalog, it is marked unavailable and blocked from runtime (raising `configuration` error to trigger app fallback without a paid call). However, if the live fetch fails, absence from the stale cache or static snapshot does NOT mark a model retired.
8. **Anthropic Validation Error Classification**: If an Anthropic request is formed without `max_output_tokens` and rejected by Perplexity (HTTP 400), it is classified as `configuration` (adapter contract defect), NOT `provider_rejection`. Auto mode will materialize a safe project default (`8192`).
9. **HTTP 403 Inspection**: Removed hardcoded `403 -> provider_rejection`. Perplexity classification helper inspects response body for permission/auth markers to distinguish `auth`, `insufficient_balance_quota`, or `provider_rejection`.
10. **HTTP 200 Application Non-Success**: Strictly enforced that HTTP 200 with `status != "completed"` (`failed`, `incomplete`, `cancelled`, `queued`, `in_progress`) or non-null `error` is a failure. No polling on synchronous chat path.

---

## B. Final Catalog & Cache Architecture

```
                    GET /v1/models Catalog Service
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
Fresh Live Request                               In-Memory Process Cache
(GET https://api.perplexity.ai/v1/models)        - TTL: 3600s (1 hour)
- No Authorization header required               - Bounded generations (gen A & B)
- 5s HTTP timeout                                - Process-local (no disk writes)
         │                                                 │
         ├────────────────── Success ──────────────────────┤
         │  Updates in-memory cache                        │
         │  Authoritative: positive & negative             │
         │                                                 │
         └── Network / HTTP Failure ───────────────────────┤
                                                           ▼
                                                Fallback Hierarchy:
                                                1. Last known in-memory catalog
                                                2. Bundled static snapshot:
                                                   PERPLEXITY_STATIC_DIRECT_MODELS
                                                (Fallback is NOT negative authority)
```

### Static Snapshot (`PERPLEXITY_STATIC_DIRECT_MODELS`):
- **Anthropic**: `anthropic/claude-sonnet-4-6`, `anthropic/claude-sonnet-4-5`, `anthropic/claude-haiku-4-5`, `anthropic/claude-opus-4-6`, `anthropic/claude-fable-5`
- **OpenAI**: `openai/gpt-5.6-sol`, `openai/gpt-5.6-terra`, `openai/gpt-5.6-luna`, `openai/gpt-6.1-sol`, `openai/gpt-6-sol`
- **Google**: `google/gemini-3.7-flash`, `google/gemini-3.5-flash`, `google/gemini-3.1-pro-preview`
- **xAI**: `xai/grok-4.20-reasoning`, `xai/grok-4.7`, `xai/grok-4.5`
- **Z.AI / Moonshot / NVIDIA**: `perplexity/glm-5.3`, `perplexity/kimi-k3`, `perplexity/nemotron-3-ultra-550b-a55b`
- **Perplexity**: `perplexity/sonar`

---

## C. Exact Preset & Direct Payload Rules

All requests target `POST https://api.perplexity.ai/v1/agent`.  
Headers: `Authorization: Bearer <PERPLEXITY_API_KEY>`, `Content-Type: application/json`.

### 1. Canonical Preset Request
```json
{
  "preset": "medium",
  "input": "User inquiry text"
}
```
- `tools` is OMITTED. Allows Perplexity to execute dynamic preset search and tool configuration without local overrides.

### 2. Direct Model Request (OpenAI, Google, xAI, Perplexity-hosted)
```json
{
  "model": "openai/gpt-5.6-sol",
  "input": "User inquiry text",
  "temperature": 0.7
}
```
- `tools` is OMITTED. Uses Perplexity Agent API default semantics.
- `temperature` included only if set and supported.

### 3. Direct Model Request (Anthropic — Mandatory `max_output_tokens`)
```json
{
  "model": "anthropic/claude-sonnet-4-6",
  "input": "User inquiry text",
  "max_output_tokens": 8192,
  "temperature": 0.7
}
```
- **Rule**: If `model.startswith("anthropic/")`:
  - If `max_tokens` is None/Auto, materialize safe default `8192`.
  - Field `"max_output_tokens"` is **NEVER** omitted.
  - If omitted locally, contract test asserts failure before HTTP dispatch.

### 4. Overridden Preset (Architecture Capability)
```json
{
  "preset": "low",
  "model": "anthropic/claude-sonnet-4-6",
  "input": "User inquiry text",
  "max_output_tokens": 8192
}
```
- Supported by `build_perplexity_payload(preset=..., model=...)`. The UI does not expose this combination in V1, but the adapter supports it without restrictions.

---

## D. Final Tool & Search Semantics

- **Presets**: Execute with dynamic preset tool configurations. We do not pass `tools` in our payload.
- **Direct Models**: The official Agent API Models documentation shows that direct models invoke web search and tool execution by default (returning citations and search results) even without explicit `tools` declared in the payload.
- **V1 Product Stance**:
  - We do NOT pass explicit `tools` in the payload.
  - We do NOT attempt to invent undocumented mechanisms to suppress Perplexity's default agentic behavior.
  - We accept Agent API default response semantics and parse citations / tool outputs cleanly without polluting user message text.

---

## E. Final Error Classification Matrix

| Scenario / Status Code | Agent API Status / Body Markers | Taxonomy Classification | Retryable? | App Fallback? | Logged Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Network error (DNS, TLS, connect reset) | N/A | `network_connection` | Yes (2x exponential backoff) | Yes (if exhausted) | `error_type`, `str(exc)` |
| Read / Connect Timeout | N/A | `timeout` | Yes (1x if connect timeout) | Yes | `timeout_seconds` |
| HTTP 401 | Body contains invalid API key / auth error | `auth` | No | Yes | `http_status: 401` |
| HTTP 403 | Body contains permission / account tier / access | `auth` (or permission classification) | No | Yes | `http_status: 403`, body |
| HTTP 403 | Generic rejection | `provider_rejection` | No | Yes | `http_status: 403`, body |
| HTTP 400 | `max_output_tokens is required` (local omission) | `configuration` | No | Yes | `http_status: 400`, `error.message` |
| HTTP 400 / 404 | Model not found / unavailable | `configuration` | No | Yes | `http_status`, `model`, `error.message` |
| HTTP 400 | Content policy / prompt rejection | `provider_rejection` | No | Yes | `http_status: 400`, `error.message` |
| HTTP 429 | Rate limit exceeded / quota | `rate_limit` | Yes (respect `Retry-After`) | Yes (if exhausted) | `retry_after`, `rate_limit_info` |
| HTTP 429 / 402 | Out of credits / balance exhausted | `insufficient_balance_quota` | No | Yes | `http_status`, `error.message` |
| HTTP 5xx | Provider server error | `provider_5xx` | Yes (2x with jitter) | Yes (if exhausted) | `http_status`, body |
| HTTP 200 + `status: "failed"` | `error.type` / `code` in body | Mapped via error code / type | According to code | Yes | `status`, `error.type`, `error.code`, `error.message` |
| HTTP 200 + `status: "incomplete"` | Output truncated / incomplete | `invalid_response` | No | Yes | `status: "incomplete"` |
| HTTP 200 + `status: "cancelled"` | Request cancelled | `provider_rejection` | No | Yes | `status: "cancelled"` |
| HTTP 200 + `status: "queued" / "in_progress"` | Unexpected non-terminal in sync path | `invalid_response` | No | Yes | `status`, unexpected async state |
| HTTP 200 + missing assistant message | No `type == "message"` with text | `empty_response` | No | Yes | `status: "completed"`, empty text |
| Malformed non-JSON body | N/A | `invalid_response` | No | Yes | `raw_body_snippet` |

---

## F. Compact Callback Design & Stale Resolution

### 1. Telegram 64-Byte Limit Protection
- Telegram callbacks NEVER embed raw model strings (e.g. `ai_set_model:perplexity:anthropic/claude-sonnet-4-6` is forbidden).
- Telegram uses the existing digest architecture:
  `build_telegram_model_callback_data(provider="perplexity", channel="chat", model=model_key)`
- Generates: `f"ai_m_c_{sha256('chat\0perplexity\0' + model)[:32]}"` (39 bytes total, comfortably under 64 bytes).

### 2. MAX Messenger Payload
- MAX uses structured payload: `f"admin_ai_set_model_perplexity_{model_key}"`.
- MAX has no 64-byte callback limit, but shares identical model identity keys with Telegram.

### 3. Safe Stale Callback Resolution (Preventing Race Conditions)
- When the live catalog refreshes from Generation A to Generation B:
  - `provider_models.py` maintains:
    1. Active Generation models;
    2. Bounded previous Generation models (kept in memory for 2 hours);
    3. Bundled static models (`PERPLEXITY_STATIC_DIRECT_MODELS`);
    4. Canonical presets (`PERPLEXITY_MODES`).
  - `resolve_telegram_model_callback()` checks active + previous generation + static fallback + presets.
  - If an admin clicks an inline button rendered just before a cache refresh, the callback resolves correctly.
  - Arbitrary forged strings cannot resolve because digests are validated against known models only.

---

## G. Unavailable Model Behavior

- **Authoritative Invalidation**:
  - Only a **successful, fresh** `GET /v1/models` response has negative authority.
  - If a stored model is absent from a fresh live response:
    - **Admin UI**: Displayed as `⚠️ [Model Label] (недоступна)` with option to re-select an active model.
    - **Runtime**: Raises `ModelUnavailableError` / `configuration` error before dispatching HTTP. Application fallback engages immediately, saving cost and time.
- **Non-Authoritative Fallback**:
  - If `GET /v1/models` fails (network error, timeout, 5xx):
    - Absence from the stale cache or static snapshot does **NOT** declare the model retired.
    - Runtime attempts the request normally against `POST /v1/agent`.
    - If Perplexity returns 400/404 indicating model retired upstream, the error is classified as `configuration` and triggers application fallback.

---

## H. Exact Telegram & MAX Admin Journeys

### 1. Screen Hierarchy
```
Admin Main -> Settings -> AI Settings -> Provider: Perplexity
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
              [⚡ Пресеты поиска]                             [🤖 Прямые модели]
                        │                                             │
               Presets Menu:                                 Direct Models (Paginated):
               - ⚡ fast                                      - anthropic/claude-sonnet-4-6
               - ⚡ low                                       - openai/gpt-5.6-sol
               - ⚡ medium (current)                          - google/gemini-3.7-flash
               - ⚡ high                                      - xai/grok-4.20-reasoning
               - ⚡ xhigh                                     - perplexity/sonar
               [« Назад] (to Perplexity Menu)                [⬅️ Пред] [1/5] [След ➡️]
                                                              [« Назад] (to Perplexity Menu)
```

### 2. Telegram Admin Journey
1. Admin opens Perplexity provider menu (`callback_data="admin_ai_provider_perplexity"`).
2. Screen displays current selection (`Текущая: medium (Пресет)` or `Текущая: Claude Sonnet 4.6 (Модель)`).
3. Admin clicks `[⚡ Пресеты поиска]` -> callback `ai_ppx_presets`.
   - Admin selects `⚡ xhigh` -> callback `build_telegram_model_callback_data("perplexity", "chat", "xhigh")`.
   - Dispatcher routes to handler -> updates DB -> re-renders screen with checkmark.
   - Admin clicks `[« Назад]` -> returns to Perplexity Provider menu.
4. Admin clicks `[🤖 Прямые модели]` -> callback `ai_ppx_models:0`.
   - Handler retrieves cached models list -> displays Page 1 (6 models).
   - Admin clicks `[След ➡️]` -> callback `ai_ppx_models:1`.
   - Admin selects `Claude Sonnet 4.6` -> callback `build_telegram_model_callback_data("perplexity", "chat", "anthropic/claude-sonnet-4-6")`.
   - Updates DB -> re-renders page with checkmark.
   - Admin clicks `[« Назад]` -> returns to Perplexity Provider menu.

### 3. MAX Messenger Journey (Exact Parity)
1. Admin opens Perplexity provider screen.
2. Clicks `[⚡ Пресеты поиска]` -> payload `ai_ppx_presets`.
   - Selects preset -> payload `admin_ai_set_model_perplexity_<preset>`.
   - Updates DB -> re-renders view.
   - Clicks `[⬅️ Назад]` -> returns to Perplexity Provider screen.
3. Clicks `[🤖 Прямые модели]` -> payload `ai_ppx_models_0`.
   - Views paginated models -> clicks `[След ➡️]` -> payload `ai_ppx_models_1`.
   - Selects model -> payload `admin_ai_set_model_perplexity_<model_key>`.
   - Updates DB -> re-renders view.
   - Clicks `[⬅️ Назад]` -> returns to Perplexity Provider screen.

---

## I. Comprehensive Test Matrix

### 1. Unit & Contract Tests (`tests/test_perplexity_agent_models.py`)
- `test_is_perplexity_preset`: Presets recognized, direct models recognized, zero collisions.
- `test_build_perplexity_payload_preset`: Verify preset payload; `tools` omitted.
- `test_build_perplexity_payload_direct_openai`: Verify `model: "openai/..."`; `tools` omitted; temperature included.
- `test_build_perplexity_payload_anthropic_mandatory_tokens`:
  - With Auto/None tokens: asserts `max_output_tokens == 8192`.
  - With explicit tokens: asserts exact value passed.
  - Contract check: asserts `max_output_tokens` is never absent for `anthropic/*`.
- `test_build_perplexity_payload_preset_model_override`: Verify adapter correctly supports both `preset` and `model` in a single payload.
- `test_extract_perplexity_text_clean_message`: Assistant message extracted; search results, fetch_url, sandbox tool outputs excluded from visible prose.
- `test_extract_perplexity_text_citations`: Citations collected and deduplicated.
- `test_extract_perplexity_text_failed_status`: HTTP 200 with `status: "failed"` raises `ProviderAdapterError`.
- `test_extract_perplexity_text_incomplete_or_cancelled`: HTTP 200 with non-terminal/incomplete status raises error.
- `test_table_driven_error_classification`: Table-driven tests for 400 (validation vs content), 401, 403 (body auth vs rejection), 404, 429, 500, network, timeout.

### 2. Catalog Service Tests
- `test_catalog_public_get_no_auth`: Verifies `GET /v1/models` sent without Authorization header.
- `test_catalog_cache_in_memory_hit`: Cache returns within 1-hour TTL without network call.
- `test_catalog_stale_fallback_to_previous_and_static`: Live network failure gracefully falls back to previous generation or static snapshot without crash.
- `test_authoritative_negative_catalog`: Fresh catalog absent model marks unavailable; stale catalog absent model does not.
- `test_stale_callback_resolution`: Callback generated under Generation A resolves under Generation B.

### 3. Telegram & MAX Admin Journey Tests
- `test_telegram_admin_perplexity_presets_flow`: Full journey using real `Dispatcher.feed_update()`.
- `test_telegram_admin_perplexity_models_pagination`: Boundary tests (0 items, 1 item, page_size, page_size+1, first, middle, last, back).
- `test_max_admin_perplexity_presets_flow`: Full journey using `MaxBotApplication.handle_update()`.
- `test_max_admin_perplexity_models_pagination`: MAX pagination parity.

### 4. Regression Suites
- `tests/test_new_provider_adapters.py`
- `tests/test_ai_model_settings_architecture.py`
- `tests/test_generation_controls.py`
- `tests/test_response_buttons.py`

---

## J. Capped Live-Probe Matrix (For Future Manual Verification)

> [!WARNING]
> **NO LIVE PROBES WILL BE RUN IN THIS SESSION.**  
> Probes are strictly documented here for future authorization after code implementation.

| Probe ID | Target | Purpose / Validation | Payload Characteristics | Capped Tokens |
| :--- | :--- | :--- | :--- | :--- |
| `PROBE-PPX-PRESET` | `preset: "fast"` | Verify canonical preset execution | `{"preset": "fast", "input": "Ответь одним словом: тест"}` (no tools) | Max 20 tokens |
| `PROBE-PPX-DIRECT-OPENAI` | `model: "openai/gpt-5.6-sol"` | **CRITICAL**: Verify actual default tool/search behavior of direct model without explicit `tools` parameter | `{"model": "openai/gpt-5.6-sol", "input": "Какая погода в Токио сейчас?"}` (no tools) | Max 50 tokens |
| `PROBE-PPX-DIRECT-ANTHROPIC` | `model: "anthropic/claude-sonnet-4-6"` | Verify mandatory `max_output_tokens` accepted (no 400 validation error) | `{"model": "anthropic/claude-sonnet-4-6", "max_output_tokens": 50, "input": "Ответь одним словом: тест"}` | Max 20 tokens |
| `PROBE-PPX-DIRECT-GOOGLE` | `model: "google/gemini-3.7-flash"` | Verify representative Google model | `{"model": "google/gemini-3.7-flash", "input": "Ответь одним словом: тест"}` | Max 20 tokens |
| `PROBE-PPX-ERROR-VALIDATION` | `model: "invalid/non-existent-model"` | Verify error classification on non-existent model | `{"model": "invalid/non-existent-model", "input": "тест"}` -> Expect HTTP 400/404 mapped to `configuration` | 0 tokens |

---

## K. Exact Expected Changed Files

| File | Nature of Changes |
| :--- | :--- |
| [provider_models.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/provider_models.py) | Expand `PERPLEXITY_MODES` (5 presets); add `PERPLEXITY_STATIC_DIRECT_MODELS`; in-memory catalog cache service (unauthenticated `GET /v1/models`, 1h TTL, bounded generations for stale callback safety); updated `get_selectable_models()` and `ensure_model_available()`. |
| [provider_adapters.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/provider_adapters.py) | Update `build_perplexity_payload()`: omit tools for presets and direct models; enforce mandatory `max_output_tokens` for Anthropic; support optional `preset` + `model` override; update `_extract_perplexity_text()` to reject HTTP 200 non-success (`status != "completed"` or non-null `error`); extract message text strictly excluding intermediate tool outputs; update error classification helper. |
| [ai_model_settings.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/ai_model_settings.py) | Update `get_generation_capabilities()` for Perplexity: model-aware (temperature on direct models, `max_output_tokens_required=True` on Anthropic). |
| [keyboards.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/keyboards.py) | Telegram Perplexity keyboard builders: Presets menu (`ai_ppx_presets`), paginated Direct Models (`ai_ppx_models:<page>`) using compact `build_telegram_model_callback_data()`. |
| [handlers.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/handlers.py) | Telegram handlers for `ai_ppx_presets`, `ai_ppx_models:<page>`, model selection, and Back navigation. |
| [max_messenger_bot/keyboards.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/max_messenger_bot/keyboards.py) | MAX Perplexity keyboards: Presets menu and paginated Direct Models. |
| [max_messenger_bot/services/admin_ai.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/max_messenger_bot/services/admin_ai.py) | MAX handlers for Perplexity Presets and paginated Direct Models with Back navigation. |
| [tests/test_perplexity_agent_models.py](file:///Users/ivankorakin/PyCharmMiscProject/.venv/imported_projects/psychonewbot2303/tests/test_perplexity_agent_models.py) | **New test suite**: Discriminator, payload builders, Anthropic token contract, HTTP 200 error states, table-driven error classification, unauthenticated catalog cache, stale callback resolution, Telegram journeys, MAX journeys. |

---

## L. Ready / Not Ready for Implementation

**VERDICT: READY FOR IMPLEMENTATION.**

All 10 mandatory corrections and clarifications are fully integrated:
1. Preset + Model contract corrected (protocol permits override, UI remains cleanly separated).
2. Preset tools semantics corrected (tools merge per tool; omitted intentionally).
3. Direct model tool semantics corrected (no false claims of plain LLM; Agent API default semantics followed).
4. `GET /v1/models` unauthenticated public request established.
5. Telegram callback length strictly guarded via SHA-256 digest architecture.
6. Process-local in-memory cache adopted (no multi-process disk writes).
7. Negative authority restricted strictly to fresh live responses.
8. Anthropic token omission classified as `configuration` error; project default auto-materialized.
9. HTTP 403 inspected via body markers instead of hardcoded rejection.
10. HTTP 200 non-success fully handled.

**Awaiting user authorization before modifying any code.**
