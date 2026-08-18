# AraContract Analyzer — API Documentation

**Base URL:** `http://localhost:8000`  
**API prefix:** `/api/contract`  
**Interactive docs:** [`/docs`](http://localhost:8000/docs) (Swagger UI) · [`/redoc`](http://localhost:8000/redoc) (ReDoc)

---

## Table of Contents

1. [General](#1-general)
2. [Authentication](#2-authentication)
3. [Common Types](#3-common-types)
4. [Upload](#4-upload)
5. [Segmentation](#5-segmentation)
6. [Classification](#6-classification)
7. [Summarization](#7-summarization)
8. [Full Analysis](#8-full-analysis)
9. [Contract Comparison](#9-contract-comparison)
10. [RAG — Contract Q&A](#10-rag--contract-qa)
11. [Error Handling](#11-error-handling)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. General

### Content Types

| Scenario | Content-Type |
|----------|-------------|
| JSON body | `application/json` |
| File upload | `multipart/form-data` |

### File Constraints

| Setting | Value |
|---------|-------|
| Max file size | 20 MB |
| Allowed formats | `.pdf` `.png` `.jpg` `.jpeg` `.tiff` `.bmp` |

### Health & Info

#### `GET /`

Returns API metadata. **No authentication required.**

**Response `200`**
```json
{
  "message": "AraContract Analyzer API",
  "version": "1.0.0",
  "docs": "/docs",
  "status": "running"
}
```

---

#### `GET /health`

Lightweight health check — use for load-balancer probes. **No authentication required.**

**Response `200`**
```json
{ "status": "healthy" }
```

---

## 2. Authentication

All contract endpoints (upload, segment, classify, summarize, analyze, compare) are protected with **Firebase Authentication** using Bearer tokens.

### How it works

The API validates Firebase ID tokens via the `Authorization` header on every protected request. Tokens are issued by Firebase Auth after the user signs in on the client side.

### Getting a token

1. Sign in via Firebase Auth on the client (email/password, Google, etc.)
2. Retrieve the ID token:
   ```javascript
   const token = await firebase.auth().currentUser.getIdToken();
   ```
3. Include it in every API request header.

### Request header

```
Authorization: Bearer <firebase_id_token>
```

### Token lifecycle

| Property | Detail |
|----------|--------|
| Expiry | 1 hour |
| Refresh | Call `getIdToken(/* forceRefresh */ true)` on the client to get a fresh token |
| Scope | Per-user — the token encodes the Firebase UID |

### Authentication errors

| Code | Detail | Cause |
|------|--------|-------|
| `401` | `"Authorization token missing"` | `Authorization` header absent |
| `401` | `"Invalid or expired token"` | Token failed Firebase verification (expired, revoked, malformed) |

### Example with curl

```bash
# Get your token first (via Firebase client SDK or REST API), then:
curl -X POST http://localhost:8000/api/contract/analyze \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -F "file=@contract.pdf"
```

### Protected vs. public endpoints

| Endpoint | Protected |
|----------|-----------|
| `GET /` | ❌ Public |
| `GET /health` | ❌ Public |
| `POST /api/contract/upload` | ✅ Firebase token required |
| `POST /api/contract/segment` | ✅ Firebase token required |
| `POST /api/contract/segment/file` | ✅ Firebase token required |
| `POST /api/contract/classify` | ✅ Firebase token required |
| `POST /api/contract/classify/batch` | ✅ Firebase token required |
| `POST /api/contract/summarize` | ✅ Firebase token required |
| `POST /api/contract/analyze` | ✅ Firebase token required |
| `POST /api/contract/compare` | ✅ Firebase token required |
| `POST /api/contract/rag/ingest` | ❌ Public |
| `POST /api/contract/rag/ask` | ❌ Public |
| `POST /api/contract/rag/summarize` | ❌ Public |
| `DELETE /api/contract/rag/session/{id}` | ❌ Public |
| `GET /api/contract/rag/health` | ❌ Public |
| `GET /api/contract/rag/status` | ❌ Public |

---

## 3. Common Types

These types appear in multiple endpoints.

### `AnalyzedClause`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Raw clause text |
| `predicted_type_clause` | `string` | One of the 8 canonical type keys |
| `type_display_name` | `string` | Arabic label, e.g. `"مالي / دفع"` |
| `predicted_risk_level` | `string` | `"low"` · `"medium"` · `"high"` |
| `risk_display_name` | `string` | Arabic label, e.g. `"مرتفع"` |
| `warning` | `string` | Explanatory text for risky clauses (empty string if none) |

### Clause Type Keys

| Key | Arabic |
|-----|--------|
| `general_provisions` | أحكام عامة |
| `payment_financial` | مالي / دفع |
| `party_obligations_a` | التزامات الطرف الأول |
| `party_obligations_b` | التزامات الطرف الثاني |
| `duration_expiration` | مدة / انتهاء |
| `termination` | فسخ / إنهاء |
| `penalties_damages` | غرامات / تعويضات |
| `dispute_resolution` | تسوية نزاعات |

### Risk Level Keys

| Key | Arabic | Meaning |
|-----|--------|---------|
| `low` | منخفض | Standard clause |
| `medium` | متوسط | Requires attention |
| `high` | مرتفع | Potentially unfair/risky |

### `ErrorResponse`

Returned by all endpoints on failure.

| Field | Type | Description |
|-------|------|-------------|
| `message` | `string` | Human-readable error description |
| `details` | `any` \| `null` | Extra info (validation errors, etc.) |

---

## 4. Upload

### `POST /api/contract/upload`

🔒 **Requires Firebase token**

Upload a contract file and extract its text. Does **not** classify or segment — returns raw text only. Useful when you want to inspect the extracted text before running the full pipeline.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | PDF or image file |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `string` | Original filename |
| `file_size` | `integer` | Bytes |
| `is_scanned` | `boolean` | `true` if OCR was used |
| `extracted_text` | `string` | Full extracted text |
| `message` | `string` | `"File uploaded and text extracted successfully"` |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `400` | Unsupported file extension |
| `413` | File exceeds 20 MB |
| `500` | Extraction failure |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/upload \
  -H "Authorization: Bearer <firebase_token>" \
  -F "file=@contract.pdf"
```

```json
{
  "filename": "contract.pdf",
  "file_size": 145678,
  "is_scanned": false,
  "extracted_text": "المادة الأولى: يلتزم الطرف الأول ...",
  "message": "File uploaded and text extracted successfully"
}
```

---

## 5. Segmentation

### `POST /api/contract/segment`

🔒 **Requires Firebase token**

Segment raw Arabic contract text into individual clauses using the article-based extraction pipeline. Supports `المادة N`, `البند الأول/الثاني/...`, ordinal markers (`أولاً/ثانياً/...`), and paragraph fallback.

**Request** — `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | Full contract text |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `clauses` | `string[]` | Ordered list of extracted clauses |
| `count` | `integer` | Number of clauses found |
| `message` | `string` | e.g. `"Text segmented into 14 clauses"` |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `500` | Segmentation failure |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/segment \
  -H "Authorization: Bearer <firebase_token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "المادة الأولى: يلتزم الطرف الأول بتقديم الخدمة...\nالمادة الثانية: يحق للطرف الثاني..."}'
```

```json
{
  "clauses": [
    "المادة الأولى: يلتزم الطرف الأول بتقديم الخدمة...",
    "المادة الثانية: يحق للطرف الثاني..."
  ],
  "count": 2,
  "message": "Text segmented into 2 clauses"
}
```

---

### `POST /api/contract/segment/file`

🔒 **Requires Firebase token**

Upload a file, extract its text, then segment into clauses in a single request.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | PDF or image file |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `string` | Original filename |
| `is_scanned` | `boolean` | `true` if OCR was used |
| `extracted_text_preview` | `string` | First 500 characters of extracted text |
| `clauses` | `string[]` | Ordered list of extracted clauses |
| `count` | `integer` | Number of clauses |
| `message` | `string` | e.g. `"File processed and segmented into 14 clauses"` |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `400` | Unsupported file extension or no text extracted |
| `413` | File exceeds 20 MB |
| `500` | Processing failure |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/segment/file \
  -H "Authorization: Bearer <firebase_token>" \
  -F "file=@contract.pdf"
```

```json
{
  "filename": "contract.pdf",
  "is_scanned": false,
  "extracted_text_preview": "المادة الأولى: يلتزم الطرف الأول ...",
  "clauses": [
    "المادة الأولى: يلتزم الطرف الأول بتقديم الخدمة في الموعد المحدد.",
    "المادة الثانية: يحق للطرف الثاني إنهاء العقد عند الإخلال."
  ],
  "count": 2,
  "message": "File processed and segmented into 2 clauses"
}
```

---

## 6. Classification

### `POST /api/contract/classify`

🔒 **Requires Firebase token**

Classify a single clause into a type and risk level. Returns full probability distributions over all classes.

**Request** — `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | Clause text to classify |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `predicted_type_clause` | `string` | Best-match clause type key |
| `predicted_risk_level` | `string` | `"low"` · `"medium"` · `"high"` |
| `type_clause_probabilities` | `object` | Map of type key → probability (0–1) |
| `risk_level_probabilities` | `object` | Map of risk key → probability (0–1) |
| `warning` | `string` \| `null` | Explanatory warning for risky clauses |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `500` | Classification failure |
| `503` | Model checkpoint not found |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/classify \
  -H "Authorization: Bearer <firebase_token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "يحق للطرف الأول إنهاء العقد في أي وقت دون إشعار مسبق."}'
```

```json
{
  "predicted_type_clause": "termination",
  "predicted_risk_level": "high",
  "type_clause_probabilities": {
    "termination": 0.7821,
    "general_provisions": 0.0934,
    "party_obligations_b": 0.0612,
    "payment_financial": 0.0212,
    "duration_expiration": 0.0143,
    "penalties_damages": 0.0134,
    "party_obligations_a": 0.0089,
    "dispute_resolution": 0.0055
  },
  "risk_level_probabilities": {
    "low": 0.0812,
    "medium": 0.1105,
    "high": 0.8083
  },
  "warning": "هذا البند يمنح الطرف الأول صلاحية الإنهاء الانفرادي دون إشعار، مما قد يكون مجحفاً للطرف الثاني."
}
```

---

### `POST /api/contract/classify/batch`

🔒 **Requires Firebase token**

Classify a list of clauses in one request. Uses the same model inference as the single endpoint but batched for efficiency.

**Request** — `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | `string[]` | ✅ | List of clause texts |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `results` | `ClassificationResponse[]` | One result per input text (same shape as single classify) |
| `count` | `integer` | Number of texts processed |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `500` | Batch classification failure |
| `503` | Model checkpoint not found |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/classify/batch \
  -H "Authorization: Bearer <firebase_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "يلتزم الطرف الثاني بدفع المبلغ خلال 30 يوماً.",
      "تبلغ مدة هذا العقد سنة واحدة قابلة للتجديد تلقائياً."
    ]
  }'
```

```json
{
  "results": [
    {
      "predicted_type_clause": "payment_financial",
      "predicted_risk_level": "low",
      "type_clause_probabilities": { "payment_financial": 0.9122, "...": "..." },
      "risk_level_probabilities": { "low": 0.9266, "medium": 0.0512, "high": 0.0222 },
      "warning": null
    },
    {
      "predicted_type_clause": "duration_expiration",
      "predicted_risk_level": "low",
      "type_clause_probabilities": { "duration_expiration": 0.8741, "...": "..." },
      "risk_level_probabilities": { "low": 0.9759, "medium": 0.0163, "high": 0.0078 },
      "warning": null
    }
  ],
  "count": 2
}
```

---

## 7. Summarization

### `POST /api/contract/summarize`

🔒 **Requires Firebase token**

Generate a 3–5 sentence Arabic executive summary from the full contract text and its classified clauses. Powered by the configured LLM (Groq / Llama by default).

**Request** — `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | Full contract text |
| `classified_clauses` | `object[]` | ✅ | Array of clause objects from the classify endpoint — each must have at minimum `text`, `predicted_type_clause`, `predicted_risk_level` |

**`classified_clauses` item shape**

```json
{
  "text": "...",
  "predicted_type_clause": "termination",
  "predicted_risk_level": "high",
  "warning": "..."
}
```

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `string` | Arabic executive summary |
| `message` | `string` | `"Executive summary generated successfully"` |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `500` | LLM failure |
| `503` | LLM service unavailable |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/summarize \
  -H "Authorization: Bearer <firebase_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "عقد خدمات بين الطرفين...",
    "classified_clauses": [
      {
        "text": "يحق للطرف الأول إنهاء العقد دون إشعار.",
        "predicted_type_clause": "termination",
        "predicted_risk_level": "high",
        "warning": "إنهاء انفرادي قد يكون مجحفاً"
      }
    ]
  }'
```

```json
{
  "summary": "يتضمن هذا العقد بنوداً تتعلق بإنهاء العقد بصورة انفرادية قد تشكّل خطراً على الطرف الثاني. يُنصح بمراجعة بند الإنهاء قبل التوقيع.",
  "message": "Executive summary generated successfully"
}
```

---

## 8. Full Analysis

### `POST /api/contract/analyze`

🔒 **Requires Firebase token**

**The recommended entry point.** Runs the complete pipeline in a single call:

1. Extract text (PDF digital or OCR)
2. Segment into clauses
3. Classify each clause (type + risk)
4. Generate risk warnings
5. Produce an Arabic executive summary
6. Return unified statistics

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | Contract file (PDF or image) |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `string` | Name of the uploaded file |
| `is_scanned` | `boolean` | Whether OCR was applied |
| `clauses` | `AnalyzedClause[]` | All classified clauses (see [Common Types](#3-common-types)) |
| `summary` | `string` | Arabic executive summary |
| `stats` | `object` | Aggregate statistics (see below) |
| `message` | `string` | `"تم تحليل العقد بنجاح واستخراج البنود والمخاطر."` |

**`stats` object**

| Field | Type | Description |
|-------|------|-------------|
| `total_clauses` | `integer` | Total number of clauses extracted |
| `high_risk_clauses` | `integer` | Count of high-risk clauses |
| `medium_risk_clauses` | `integer` | Count of medium-risk clauses |
| `low_risk_clauses` | `integer` | Count of low-risk clauses |
| `type_distribution` | `object` | Map of clause type key → count |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `400` | Unsupported file type or no text extracted |
| `413` | File exceeds 20 MB |
| `500` | Pipeline failure |
| `503` | Classification model not found |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/analyze \
  -H "Authorization: Bearer <firebase_token>" \
  -F "file=@contract.pdf"
```

```json
{
  "filename": "contract.pdf",
  "is_scanned": false,
  "clauses": [
    {
      "text": "يلتزم الطرف الثاني بدفع المبلغ المتفق عليه خلال 30 يوماً من تاريخ الاستلام.",
      "predicted_type_clause": "payment_financial",
      "type_display_name": "مالي / دفع",
      "predicted_risk_level": "low",
      "risk_display_name": "منخفض",
      "warning": ""
    },
    {
      "text": "يحق للطرف الأول إنهاء العقد في أي وقت دون إشعار مسبق.",
      "predicted_type_clause": "termination",
      "type_display_name": "فسخ / إنهاء",
      "predicted_risk_level": "high",
      "risk_display_name": "مرتفع",
      "warning": "هذا البند يمنح الطرف الأول صلاحية الإنهاء الانفرادي دون إشعار، مما قد يكون مجحفاً."
    }
  ],
  "summary": "يتضمن العقد بنوداً مالية معتدلة وبنداً للإنهاء يُشكّل خطراً عالياً على الطرف الثاني.",
  "stats": {
    "total_clauses": 2,
    "high_risk_clauses": 1,
    "medium_risk_clauses": 0,
    "low_risk_clauses": 1,
    "type_distribution": {
      "payment_financial": 1,
      "termination": 1
    }
  },
  "message": "تم تحليل العقد بنجاح واستخراج البنود والمخاطر."
}
```

---

## 9. Contract Comparison

### `POST /api/contract/compare`

🔒 **Requires Firebase token**

Upload two contract files and receive a side-by-side comparison of their clauses, types, risk distributions, and structural differences.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file1` | file | ✅ | First contract file |
| `file2` | file | ✅ | Second contract file |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `contract1_summary` | `object` | Statistics snapshot for contract 1 |
| `contract2_summary` | `object` | Statistics snapshot for contract 2 |
| `differences` | `object[]` | List of detected differences between the two contracts |
| `message` | `string` | `"Contract comparison completed successfully"` |

**`contract_summary` object**

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `string` | Original filename |
| `total_clauses` | `integer` | Total clause count |
| `high_risk_clauses` | `integer` | High-risk clause count |
| `medium_risk_clauses` | `integer` | Medium-risk clause count |
| `low_risk_clauses` | `integer` | Low-risk clause count |
| `type_distribution` | `object` | Type key → count |

**`differences` item**

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Difference category (e.g. `"risk_level"`, `"clause_type"`, `"missing_clause"`) |
| `description` | `string` | Human-readable description of the difference |
| `contract1_value` | `any` | Value in contract 1 |
| `contract2_value` | `any` | Value in contract 2 |

**Error responses**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid Firebase token |
| `400` | Unsupported file extension for either file |
| `413` | Either file exceeds 20 MB |
| `500` | Comparison pipeline failure |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/compare \
  -H "Authorization: Bearer <firebase_token>" \
  -F "file1=@contract_v1.pdf" \
  -F "file2=@contract_v2.pdf"
```

```json
{
  "contract1_summary": {
    "filename": "contract_v1.pdf",
    "total_clauses": 10,
    "high_risk_clauses": 3,
    "medium_risk_clauses": 2,
    "low_risk_clauses": 5,
    "type_distribution": { "termination": 2, "payment_financial": 3, "...": "..." }
  },
  "contract2_summary": {
    "filename": "contract_v2.pdf",
    "total_clauses": 12,
    "high_risk_clauses": 1,
    "medium_risk_clauses": 3,
    "low_risk_clauses": 8,
    "type_distribution": { "termination": 1, "payment_financial": 4, "...": "..." }
  },
  "differences": [
    {
      "type": "risk_level",
      "description": "Contract 1 has more high-risk clauses",
      "contract1_value": 3,
      "contract2_value": 1
    }
  ],
  "message": "Contract comparison completed successfully"
}
```

---

## 10. RAG — Contract Q&A

The RAG (Retrieval-Augmented Generation) module enables conversational question-answering over a previously ingested contract. All endpoints are under `/api/contract/rag`.

> **Note:** RAG endpoints are currently **public** (no authentication required).

**Typical workflow:**
```
POST /rag/ingest  →  store session_id
POST /rag/ask     →  ask questions using session_id
POST /rag/summarize (optional, stateless)
DELETE /rag/session/{session_id}  →  cleanup
```

---

### `POST /api/contract/rag/ingest`

Chunk the contract clauses and build a vector store in Qdrant for the session.

**Request** — `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `clauses` | `string[]` | ✅ | List of clause strings — use the output of `/segment` or `/analyze` |
| `session_id` | `string` | ❌ | Custom session identifier. Auto-generated (UUID) if omitted |

**Response `201`**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Session identifier to use in subsequent requests |
| `clauses_count` | `integer` | Number of clauses indexed |
| `message` | `string` | `"تم استيعاب العقد بنجاح وبناء الـ Vector Store"` |

**Error responses**

| Code | Condition |
|------|-----------|
| `500` | Vector store build failure |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "clauses": [
      "المادة الأولى: يلتزم الطرف الأول بتقديم الخدمة.",
      "المادة الثانية: يحق للطرف الأول إنهاء العقد دون إشعار."
    ]
  }'
```

```json
{
  "session_id": "3f7a2b1c-9e4d-4a8f-b123-456789abcdef",
  "clauses_count": 2,
  "message": "تم استيعاب العقد بنجاح وبناء الـ Vector Store"
}
```

---

### `POST /api/contract/rag/ask`

Ask an Arabic question about the ingested contract. Retrieves the most relevant clauses, re-ranks them with a cross-encoder, then generates an answer via the configured LLM.

**Request** — `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `session_id` | `string` | ✅ | — | From `IngestResponse` |
| `question` | `string` | ✅ | — | Arabic question (min 3 chars) |
| `top_k` | `integer` | ❌ | `3` | Number of clauses to retrieve (1–10) |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | LLM-generated answer based on contract content only |
| `retrieved_clauses` | `RetrievedClause[]` | Source clauses used to generate the answer |
| `session_id` | `string` | Echo of the session id |

**`RetrievedClause` object**

| Field | Type | Description |
|-------|------|-------------|
| `clause_index` | `integer` | Index of the clause in the original clauses list |
| `parent_text` | `string` | Full text of the retrieved clause |
| `score` | `float` | Cross-encoder relevance score |

**Error responses**

| Code | Condition |
|------|-----------|
| `404` | Session not found or question is empty |
| `500` | LLM or retrieval failure |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "3f7a2b1c-9e4d-4a8f-b123-456789abcdef",
    "question": "هل يمكن إنهاء العقد دون إشعار مسبق؟",
    "top_k": 3
  }'
```

```json
{
  "answer": "نعم، وفقاً للمادة الثانية يحق للطرف الأول إنهاء العقد دون إشعار مسبق، وهو بند يُعدّ مجحفاً للطرف الثاني.",
  "retrieved_clauses": [
    {
      "clause_index": 1,
      "parent_text": "المادة الثانية: يحق للطرف الأول إنهاء العقد دون إشعار.",
      "score": 0.9231
    }
  ],
  "session_id": "3f7a2b1c-9e4d-4a8f-b123-456789abcdef"
}
```

---

### `POST /api/contract/rag/summarize`

Generate an Arabic executive summary (3–5 sentences) directly from raw clauses. Optionally accepts the classifier output to enrich the summary with risk context. This endpoint is **stateless** — it does not require a prior `/ingest` call.

**Request** — `application/json`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `clauses` | `string[]` | ✅ | All contract clauses (min 1) |
| `analyzed_clauses` | `object[]` | ❌ | Classifier output from `/analyze` or `/classify/batch` — adds risk context to the summary |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `string` | Arabic executive summary |
| `clauses_count` | `integer` | Number of clauses processed |

**Error responses**

| Code | Condition |
|------|-----------|
| `400` | Empty clauses list |
| `500` | LLM failure |

**Example**

```bash
curl -X POST http://localhost:8000/api/contract/rag/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "clauses": [
      "المادة الأولى: يلتزم الطرف الأول بتقديم الخدمة.",
      "المادة الثانية: يحق للطرف الأول إنهاء العقد دون إشعار."
    ],
    "analyzed_clauses": [
      {
        "text": "المادة الثانية: يحق للطرف الأول إنهاء العقد دون إشعار.",
        "predicted_type_clause": "termination",
        "predicted_risk_level": "high",
        "warning": "إنهاء انفرادي"
      }
    ]
  }'
```

```json
{
  "summary": "يتضمن العقد التزامات تقديم الخدمة مع بند إنهاء انفرادي خطير يمنح الطرف الأول صلاحيات غير متوازنة.",
  "clauses_count": 2
}
```

---

### `DELETE /api/contract/rag/session/{session_id}`

Delete the vector store for a session. Call this after the user is done to free Qdrant memory.

**Path parameter**

| Param | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Session identifier from `/rag/ingest` |

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | The deleted session id |
| `deleted` | `boolean` | `true` if found and deleted, `false` if session did not exist |
| `message` | `string` | `"تم حذف الجلسة بنجاح"` or `"الجلسة غير موجودة"` |

**Error responses**

| Code | Condition |
|------|-----------|
| `500` | Deletion failure |

**Example**

```bash
curl -X DELETE http://localhost:8000/api/contract/rag/session/3f7a2b1c-9e4d-4a8f-b123-456789abcdef
```

```json
{
  "session_id": "3f7a2b1c-9e4d-4a8f-b123-456789abcdef",
  "deleted": true,
  "message": "تم حذف الجلسة بنجاح"
}
```

---

### `GET /api/contract/rag/health`

Check that the RAG pipeline service is reachable.

**Response `200`**

```json
{ "status": "ok", "service": "RAG Pipeline" }
```

---

### `GET /api/contract/rag/status`

Check the current LLM provider status (Groq / Qwen local / Ollama).

**Response `200`**

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"ok"` |
| `service` | `string` | `"RAG Pipeline"` |
| `provider` | `string` | Active LLM provider name |
| `ready` | `boolean` | Whether the LLM is ready to serve requests |
| `model` | `string` \| `null` | Model name/identifier |
| `base_url` | `string` \| `null` | Base URL for remote providers |
| `details` | `any` \| `null` | Additional provider-specific info |

**Example**

```bash
curl http://localhost:8000/api/contract/rag/status
```

```json
{
  "status": "ok",
  "service": "RAG Pipeline",
  "provider": "groq",
  "ready": true,
  "model": "llama-3.1-8b-instant",
  "base_url": null,
  "details": null
}
```

---

## 11. Error Handling

All endpoints return `ErrorResponse` on failure.

```json
{
  "message": "Human-readable description",
  "details": null
}
```

Validation errors (422) include structured `details`:

```json
{
  "message": "Invalid input",
  "details": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### HTTP Status Code Summary

| Code | Meaning |
|------|---------|
| `200` | Success |
| `201` | Resource created (RAG ingest) |
| `400` | Bad request — invalid file type, empty text, etc. |
| `401` | Unauthorized — missing or invalid Firebase token |
| `404` | Not found — RAG session does not exist |
| `413` | File too large (> 20 MB) |
| `422` | Validation error — missing or malformed JSON fields |
| `500` | Internal server error — pipeline or model failure |
| `503` | Service unavailable — model checkpoint missing |

---

## 12. Configuration Reference

Settings are loaded from `backend/.env`. All variables have defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server port |
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]` | CORS allowed origins |
| `MAX_FILE_SIZE` | `20971520` (20 MB) | Maximum upload size in bytes |
| `UPLOAD_DIR` | `./uploads` | Temporary file directory |
| `ALLOWED_EXTENSIONS` | `.pdf .png .jpg .jpeg .tiff .bmp` | Accepted file extensions |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | — | Path to Firebase service account JSON key file |
| `CLASSIFIER_MODEL_PATH` | `../models/checkpoints/aracontract_classifier.pt` | Path to the `.pt` checkpoint |
| `MAX_SEQUENCE_LENGTH` | `512` | BERT tokenizer max length |
| `EMBEDDING_MODEL_PATH` | `models_local/paraphrase-multilingual-MiniLM-L12-v2` | Sentence embedding model |
| `EMBEDDING_VECTOR_SIZE` | `384` | Embedding dimension |
| `CHUNK_MAX_CHARS` | `450` | RAG chunk size in characters |
| `CHUNK_OVERLAP_CHARS` | `50` | RAG chunk overlap |
| `CROSS_ENCODER_MODEL_PATH` | `models_local/cross-encoder-ms-marco-MiniLM-L-6-v2` | Re-ranking model |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.45` | Minimum score to include a chunk |
| `RETRIEVAL_INITIAL_TOP_K` | `15` | Initial retrieval candidates |
| `RETRIEVAL_FINAL_TOP_K` | `3` | Final results after re-ranking |
| `LLM_PROVIDER` | `groq` | LLM backend: `groq` or `qwen` |
| `GROQ_API_KEY` | `""` | Groq API key (required if `LLM_PROVIDER=groq`) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `LLM_MODEL_PATH` | `models_local/Qwen2.5-7B-Instruct` | Local Qwen model path |
| `QDRANT_MODE` | `memory` | Qdrant storage: `memory` or `url` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (when `QDRANT_MODE=url`) |

---

## Endpoint Quick Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | ❌ | API info |
| `GET` | `/health` | ❌ | Health check |
| `POST` | `/api/contract/upload` | 🔒 | Upload file → extract text |
| `POST` | `/api/contract/segment` | 🔒 | Segment text → clauses |
| `POST` | `/api/contract/segment/file` | 🔒 | Upload file → extract → segment |
| `POST` | `/api/contract/classify` | 🔒 | Classify single clause |
| `POST` | `/api/contract/classify/batch` | 🔒 | Classify multiple clauses |
| `POST` | `/api/contract/summarize` | 🔒 | Generate executive summary |
| `POST` | `/api/contract/analyze` | 🔒 | Full pipeline (recommended) |
| `POST` | `/api/contract/compare` | 🔒 | Compare two contracts |
| `POST` | `/api/contract/rag/ingest` | ❌ | Index clauses into vector store |
| `POST` | `/api/contract/rag/ask` | ❌ | Ask a question about the contract |
| `POST` | `/api/contract/rag/summarize` | ❌ | RAG-powered summary (stateless) |
| `DELETE` | `/api/contract/rag/session/{id}` | ❌ | Delete RAG session |
| `GET` | `/api/contract/rag/health` | ❌ | RAG service health |
| `GET` | `/api/contract/rag/status` | ❌ | LLM provider status |
