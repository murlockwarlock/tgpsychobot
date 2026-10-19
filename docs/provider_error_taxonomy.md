# Provider failure contract

Audit date: 2026-10-19. Base: `072e52c4d28a2b5d1971e041432faaddc7bf3d1f`.
This document is explicitly requested for this PR. Sources below are provider-owned;
examples and mappings are paraphrases, not copied documentation.

## Endpoint inventory and sources

| Provider | Actual application operations / endpoints | Authoritative sources |
|---|---|---|
| OpenAI | chat/vision `/v1/chat/completions`, image generation `/v1/images/generations`, transcription `/v1/audio/transcriptions` | [Errors](https://developers.openai.com/api/docs/guides/error-codes), [Images](https://developers.openai.com/api/reference/resources/images), [Audio](https://developers.openai.com/api/reference/resources/audio) |
| Claude | chat/vision `/v1/messages` | [Errors](https://platform.claude.com/docs/en/api/errors), [Spend/rate limits](https://platform.claude.com/docs/en/api/rate-limits) |
| Gemini | chat, vision, image generation/editing, inline-audio transcription: `/v1beta/models/{model}:generateContent` | [Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting), [FinishReason](https://ai.google.dev/api/generate-content), [Limits](https://ai.google.dev/gemini-api/docs/rate-limits) |
| DeepSeek | chat `/chat/completions` through OpenAI SDK | [Chat errors](https://api-docs.deepseek.com/quick_start/error_codes/); Search/Fetch codes are not imported |
| OpenRouter | chat/vision `/api/v1/chat/completions`; catalog GET `/api/v1/models` | [Errors and typed vocabulary](https://openrouter.ai/docs/api_reference/errors-and-debugging), [Live catalog](https://openrouter.ai/api/v1/models) |
| KIE | chat/vision `/{model}/v1/chat/completions`, `/claude/v1/messages`, `/codex/v1/responses`, `/grok/v1/responses`, `/gemini/v1/models/{model}:streamGenerateContent`; image and transcription jobs `/api/v1/jobs/createTask`, `/api/v1/jobs/recordInfo`; `/api/file-stream-upload`; `/api/v1/common/download-url`; `/api/v1/chat/credit` | [Transport/auth](https://docs.kie.ai/), [Task status](https://docs.kie.ai/market/common/get-task-detail), [Upload](https://docs.kie.ai/file-upload-api/quickstart), [Common API](https://docs.kie.ai/common-api/quickstart), [Codex](https://docs.kie.ai/market/codex/gpt-codex) |
| Perplexity | chat `/v1/agent`, catalog GET `/v1/models` | [Agent schema](https://docs.perplexity.ai/api-reference/agent-post), [Models](https://docs.perplexity.ai/api-reference/models-get), [SDK error handling](https://docs.perplexity.ai/docs/sdk/error-handling), [Background status](https://docs.perplexity.ai/docs/agent-api/background-mode) |
| Deepgram | prerecorded transcription `/v1/listen` | [Errors](https://developers.deepgram.com/docs/errors) |

All sources accessed on the audit date. KIE protocol families are distinct: job
codes are not upstream model error types. Gemini's [snake_case error table](https://ai.google.dev/gemini-api/docs/api-errors)
expressly describes **Interactions**, not generateContent. Snake_case codes on
generateContent are defensive compatibility inputs, not documented wire fixtures.
Perplexity SPACE is excluded. Agent ErrorInfo documents string `code`, `type`,
`message`, but does not enumerate a closed code vocabulary; familiar codes are
defensive mappings, not claimed Agent guarantees.

Each matrix row uses the authoritative provider URL(s) in the endpoint inventory
for its provider; the audit date above applies to every row.

## Interpretation matrix

`T` means transient, eligible for a later user retry; `P` means no immediate
same-request retry. Internal retry additionally requires an idempotent/safe stage,
remaining attempt budget and enough deadline to honor the delay. `RA` means honor
Retry-After when present. `change` requires changed input/configuration; `operator`
requires account/access/configuration intervention. HTTP is transport status;
KIE body codes and OpenRouter embedded codes do not overwrite it.

| Provider / operation | HTTP / structured signal | Relevant fields | Meaning / canonical class and detail | T/P | Internal / user retry | RA | Required change / operator |
|---|---|---|---|---|---|---|---|
| OpenAI / all above | 401 invalid key | error.code/type/message | auth / invalid_api_key | P | no/no | — | operator |
| OpenAI / all | 401 IP not authorized | error.message; code if supplied | forbidden_geo / ip_not_authorized | P | no/no | — | operator |
| OpenAI / all | 403 unsupported country | error.code/message | forbidden_geo / unsupported_country_region_territory | P | no/no | — | operator |
| OpenAI / all | 429 rate_limit_error, slow_down | error.type/code | rate_limit / original code | T | bounded/yes | yes | wait |
| OpenAI / all | 429 credit_balance_exhausted | error.code | insufficient_balance_quota / credit_balance_exhausted | P | no/no | — | operator |
| OpenAI / all | 429 organization_spend_limit_exceeded, project_spend_limit_exceeded, organization_usage_limit_exceeded, insufficient_quota | error.code/type | insufficient_balance_quota / original code | P | no/no | — | operator/reset |
| OpenAI / all | 500; 503 server_is_overloaded | error.code/type | provider_5xx; provider_overloaded | T | bounded/yes | yes | wait |
| Claude / messages | 400 invalid_request_error | error.type/message | invalid_request | P | no/no | — | change |
| Claude / messages | 400 spend limit; 429 spend cap | error.type/message; absent retry-after on tier cap | insufficient_balance_quota / spend_limit | P | no/no | — | operator/reset |
| Claude / messages | 401 authentication_error; 402 billing_error; 403 permission_error | error.type | auth; insufficient_balance_quota; permission | P | no/no | — | operator |
| Claude / messages | 404 not_found_error; 409 conflict_error; 413 request_too_large | error.type | invalid_model_or_resource; invalid_request; context_limit | P | no/no | — | change |
| Claude / messages | 429 rate_limit_error | error.type/message, retry-after | rate_limit (spend evidence takes precedence) | T | bounded/yes | yes | wait |
| Claude / messages | 500 api_error; 504 timeout_error; 529 overloaded_error | error.type, request_id | provider_5xx; timeout; provider_overloaded | T | bounded/yes | yes | wait |
| Gemini / generateContent | 400 INVALID_ARGUMENT; FAILED_PRECONDITION | error.status, error.details | invalid_request; configuration | P | no/no | — | change/operator |
| Gemini / generateContent | UNAUTHENTICATED; PERMISSION_DENIED; NOT_FOUND | error.status/details | auth; permission; invalid_model_or_resource | P | no/no | — | operator/change |
| Gemini / generateContent | 429 RESOURCE_EXHAUSTED | ErrorInfo reason, QuotaFailure violations, RetryInfo retryDelay | rate_limit for explicit short-window limit; insufficient_balance_quota for daily/unspecified quota | conditional | conditional | yes | wait/reset/operator |
| Gemini / generateContent | INTERNAL; UNAVAILABLE; DEADLINE_EXCEEDED | error.status | provider_5xx; provider_unavailable; timeout | T | bounded/yes | yes | wait |
| Gemini / generateContent | CANCELLED; UNIMPLEMENTED; ALREADY_EXISTS; ABORTED | error.status | cancelled; invalid_request; invalid_request; provider_rejection | P | no/no | — | change/higher-level resolution |
| Gemini / generation | SAFETY, RECITATION, LANGUAGE, PROHIBITED_CONTENT, SPII, BLOCKLIST, IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT, IMAGE_RECITATION | promptFeedback.blockReason, candidates[].finishReason | content_policy / original reason | P | no/no | — | change |
| Gemini / generation | MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, TOO_MANY_TOOL_CALLS, MISSING_THOUGHT_SIGNATURE, NO_IMAGE | candidates[].finishReason | provider_rejection / original reason | P | no/no | — | change |
| Gemini / defensive only | invalid_request, failed_precondition, authentication, payment_required, permission_denied, not_found, model_not_found, already_exists, aborted, rate_limit_exceeded, too_many_requests, quota_exceeded, cancelled, api_error, unimplemented, service_unavailable, deadline_exceeded | error.code | same semantic classes; rate codes transient, quota permanent | conditional | conditional | yes | see native equivalent |
| DeepSeek / chat | 400, 422 | error.message/type/code when supplied | invalid_request | P | no/no | — | change |
| DeepSeek / chat | 401; 402 | error fields | auth; insufficient_balance_quota | P | no/no | — | operator |
| DeepSeek / chat | 429; 500; 503 | error fields | rate_limit; provider_5xx; provider_overloaded | T | bounded/yes | yes | wait |
| OpenRouter / chat+vision | any, including 200; context_length_exceeded, string_too_long | error.metadata.error_type | context_limit | P | no/no | — | change |
| OpenRouter / chat+vision | max_tokens_exceeded; token_limit_exceeded | error.metadata.error_type | output_budget_exhausted; context_limit | P | no/no | — | change |
| OpenRouter / chat+vision | authentication; permission_denied; payment_required | error.metadata.error_type | auth; permission; insufficient_balance_quota | P | no/no | — | operator |
| OpenRouter / chat+vision | rate_limit_exceeded; provider_overloaded; provider_unavailable; server; timeout | error.metadata.error_type | rate_limit; provider_overloaded; provider_unavailable; provider_5xx; timeout | T | bounded/yes | yes | wait |
| OpenRouter / chat+vision | invalid_request, invalid_prompt, precondition_failed, payload_too_large, unprocessable | error.metadata.error_type | invalid_request | P | no/no | — | change |
| OpenRouter / chat+vision | not_found | error.metadata.error_type | invalid_model_or_resource | P | no/no | — | change/operator |
| OpenRouter / chat+vision | content_policy_violation, refusal | error.metadata.error_type | content_policy | P | no/no | — | change |
| OpenRouter / vision | invalid_image, image_too_large, image_too_small, unsupported_image_format, image_not_found, image_download_failed | error.metadata.error_type | invalid_request / original type | P | no/no | — | change |
| OpenRouter / chat+vision | unmapped or future type | error.metadata.error_type, provider_code | unknown; preserve type | P | no/no | — | investigate |
| OpenRouter / chat+vision | 402 in-flight budget | metadata.limit_source=openrouter_in_flight_budget AND Retry-After | insufficient_balance_quota / openrouter_in_flight_budget | T | bounded/yes | required | wait |
| OpenRouter / chat+vision | ordinary 402 | metadata, Retry-After | insufficient_balance_quota | P | no/no | — | operator |
| KIE / jobs, upload, common | HTTP 200 body code 401 | code, msg | auth; permission when documented access-permissions/IP text proves it | P | no/no | — | operator |
| KIE / jobs, upload, common | HTTP 200 body code 402; 400/422; 429; 500 | code, msg | insufficient_balance_quota; invalid_request; rate_limit; provider_5xx | conditional | conditional by stage | yes | operator/change/wait |
| KIE / model protocols | error.type/message/code envelope | error, request_id | parse supplied type; unknown types stay unknown | conditional | conditional by stage | yes | type-dependent |
| KIE / task polling | data.state=fail | failCode, failMsg, taskId | structured task failure, never success/empty | conditional | do not recreate task automatically | yes | code-dependent |
| Deepgram / listen | 401 INVALID_AUTH | err_code, err_msg, request_id | auth | P | no/no | — | operator |
| Deepgram / listen | 401/403 INSUFFICIENT_PERMISSIONS | same | permission | P | no/no | — | operator |
| Deepgram / listen | 402 ASR_PAYMENT_REQUIRED; 422 ASR_UNPROCESSABLE_ENTITY; corrupt audio 400 | same | insufficient_balance_quota; invalid_request | P | no/no | — | operator/change |
| Deepgram / listen | 429 TOO_MANY_REQUESTS; 408; 500/502/503/504 | same | rate_limit; timeout; provider_5xx/provider_unavailable/timeout | T | bounded/no voice button | yes | wait |
| Perplexity / agent | HTTP errors; ErrorInfo | error.type/code/message, x-request-id | defensive typed mapping before HTTP fallback; unknown type is unknown | conditional | bounded only safe stages | yes | code-dependent |
| Perplexity / agent | completed | status/output | success only if valid final text | — | no/no | — | — |
| Perplexity / agent | failed | status/error | normalized ErrorInfo | conditional | no automatic paid replay | yes | code-dependent |
| Perplexity / agent | incomplete | status, incomplete_details if supplied (defensive) | output_budget_exhausted for explicit max_output_tokens, otherwise invalid_response | P | no/no | — | change/investigate |
| Perplexity / agent | cancelled; queued; in_progress | status | cancelled; invalid_response in synchronous pipeline | P | no/no | — | no duplicate run |
| All / transport | timeout, connection, TLS, proxy exceptions | typed exception chain | timeout, network_connection, network_ssl, proxy_error | T | stage-dependent/yes for chat | — | recover network |
| All / application | TypeError, invariant failure | typed exception | application_internal | P | no/no | — | fix code |
| Catalog GETs | structured error then status fallback | same provider envelope, request ID | read-only failure; never synthesize model availability | conditional | bounded/no chat button | yes | type-dependent |

## Policy invariants

Precedence: local validation; typed transport; provider structured envelope;
SDK exception with embedded response; status fallback; provider-specific documented
message fallback; generic message fallback; unknown. Provider message fallback is
marked `message_fallback`. No whole request bodies belong in normalized diagnostics.
Messages are sanitized before truncation. Request/task IDs are retained.

AILog rows mean logical routes, not physical HTTP calls. Diagnostics carry explicit
retryability and network-attempt state. Historical rows use the legacy canonical
classification mapping. A successful route anywhere suppresses Retry. Otherwise
any actually executed retryable route permits chat Retry, regardless of the final
route's deterministic failure. Telegram/MAX display and callback must share this rule.

Unknown new provider types and unstructured external failures are preserved and
fail closed for retry. TypeError/AttributeError/invariant failures are
`application_internal` and never become provider retry loops. Voice replay is not added.
Retry-After must not be shortened to fit a cap: if it exceeds the internal wait
budget, skip internal retry and preserve the delay for diagnostics.

## SDK audit and implementation tracking

Installed locally: OpenAI 2.32.0, Anthropic 0.97.0, httpx 0.28.1.
Both SDK defaults are two retries, so every OpenAI/Anthropic client construction
used by the Telegram/MAX AI paths now sets `max_retries=0`. Gemini, KIE,
OpenRouter, Perplexity and Deepgram use HTTPX paths (no SDK retry layer).
HTTP adapters own bounded retries: OpenRouter/Deepgram at most two physical
calls, Perplexity at most two, and KIE transcription stages at most three.

## Additional owner requirements

The owner explicitly expanded this PR to correct misleading admin result wording
(successful fallback must not say `Ошибка: Ответ получен`) and add existing economical
OpenRouter models. Public catalog verified on the audit date: `google/gemini-2.5-flash`,
`google/gemini-3-flash-preview`, `openai/gpt-5.6-luna`, `deepseek/deepseek-v4-flash`,
`deepseek/deepseek-v4-pro`. Gemini 3 Flash's exact catalog ID includes `-preview`.
Duplicate primary/fallback configuration remains a separate follow-up risk.

Implementation is in `provider_failures.py` and the shared adapter/runtime paths;
behavioral evidence is in the provider contract, fallback-group, Telegram/MAX
journey and full-suite tests. No live provider calls are made by CI.
