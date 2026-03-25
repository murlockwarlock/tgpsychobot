# KIE.AI Image Generation API

## Common

- **Base URL:** `https://api.kie.ai`
- **Endpoint:** `POST /api/v1/jobs/createTask`
- **Auth:** `Authorization: Bearer YOUR_API_KEY` (keys at https://kie.ai/api-key)
- **Response:** `{ "code": int, "msg": "string", "data": { "taskId": "string" } }`
- **Result retrieval:** Poll task status or use `callBackUrl` webhook
- **Error codes:** 200 OK, 401 Unauthorized, 402 Insufficient Credits, 404 Not Found, 422 Validation Error, 429 Rate Limited, 455 Service Unavailable, 500 Server Error, 501 Generation Failed, 505 Feature Disabled

## Request Format (common)

```json
{
  "model": "<model_name>",
  "input": {
    "prompt": "...",
    ...model-specific params...
  },
  "callBackUrl": "https://optional-webhook.com/callback"
}
```

## Task Status Retrieval

```
GET /api/v1/jobs/getTaskDetails?taskId=<taskId>
Authorization: Bearer YOUR_API_KEY
```

Response includes task status and output URLs when complete.

---

## Models

### 1. nano-banana-2 (Nano Banana 2)

Text-to-image + optional reference images.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input.prompt` | string | Yes | Max 20,000 chars |
| `input.image_input` | array of URIs | No | Up to 14 images, max 30MB each, JPEG/PNG/WebP |
| `input.aspect_ratio` | string | No | `1:1`, `1:4`, `1:8`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `8:1`, `9:16`, `16:9`, `21:9`, `auto` (default: `auto`) |
| `input.resolution` | string | No | `1K`, `2K`, `4K` (default: `1K`) |
| `input.output_format` | string | No | `png`, `jpg` (default: `jpg`) |

### 2. google/imagen4-fast (Imagen 4 Fast)

Fast text-to-image, supports multiple images per request.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input.prompt` | string | Yes | Max 5,000 chars |
| `input.negative_prompt` | string | No | Max 5,000 chars |
| `input.aspect_ratio` | string | No | `1:1`, `16:9`, `9:16`, `3:4`, `4:3` (default: `16:9`) |
| `input.num_images` | string | No | `"1"`, `"2"`, `"3"`, `"4"` (default: `"1"`) |
| `input.seed` | integer | No | Reproducibility seed |

### 3. google/imagen4-ultra (Imagen 4 Ultra)

Highest quality text-to-image, 1 image per request.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input.prompt` | string | Yes | Max 5,000 chars |
| `input.negative_prompt` | string | No | Max 5,000 chars |
| `input.aspect_ratio` | string | No | `1:1`, `16:9`, `9:16`, `3:4`, `4:3` (default: `1:1`) |
| `input.seed` | string | No | Max 500 chars |

### 4. google/imagen4 (Imagen 4 Standard)

Same as Ultra parameters.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input.prompt` | string | Yes | Max 5,000 chars |
| `input.negative_prompt` | string | No | Max 5,000 chars |
| `input.aspect_ratio` | string | No | `1:1`, `16:9`, `9:16`, `3:4`, `4:3` (default: `1:1`) |
| `input.seed` | string | No | Max 500 chars |

### 5. google/nano-banana-edit (Nano Banana Edit)

Image editing — input images required.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input.prompt` | string | Yes | Max 5,000 chars |
| `input.image_urls` | array of strings | **Yes** | Up to 10 images, JPEG/PNG/WebP, max 10MB each |
| `input.output_format` | string | No | `png`, `jpeg` (default: `png`) |
| `input.image_size` | string | No | `1:1`, `9:16`, `16:9`, `3:4`, `4:3`, `3:2`, `2:3`, `5:4`, `4:5`, `21:9`, `auto` (default: `1:1`) |

### 6. nano-banana-pro (Pro Image-to-Image)

Text-to-image + optional reference images (pro quality).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input.prompt` | string | Yes | Max 10,000 chars |
| `input.image_input` | array of URIs | No | Up to 8 images, max 30MB each, JPEG/PNG/WebP |
| `input.aspect_ratio` | string | No | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, `auto` (default: `1:1`) |
| `input.resolution` | string | No | `1K`, `2K`, `4K` (default: `1K`) |
| `input.output_format` | string | No | `png`, `jpg` (default: `png`) |

---

## Example: Generate image with Imagen 4 Fast

```bash
curl -X POST https://api.kie.ai/api/v1/jobs/createTask \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/imagen4-fast",
    "input": {
      "prompt": "A red cat sitting on a windowsill",
      "aspect_ratio": "1:1",
      "num_images": "1"
    }
  }'
```

Response:
```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "taskId": "abc123..."
  }
}
```

Then poll:
```bash
curl "https://api.kie.ai/api/v1/jobs/getTaskDetails?taskId=abc123..." \
  -H "Authorization: Bearer YOUR_API_KEY"
```
