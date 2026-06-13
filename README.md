# AraContract Analyzer

AI-powered Arabic contract analysis API. Automatically extracts, segments, classifies, and summarizes Arabic legal contracts using machine learning and NLP.

## Features

- **Text Extraction**: Extract text from PDFs (digital) and scanned documents (OCR)
- **Clause Segmentation**: Split contracts into individual clauses using Arabic legal markers
- **Type Classification**: Classify each clause into 8 canonical legal categories
- **Risk Assessment**: Detect low/medium/high risk levels with explanatory warnings
- **Executive Summary**: Generate Arabic summaries of contract key points
- **Contract Comparison**: Compare two contracts for differences

## Quick Start

### Prerequisites

```bash
# Python 3.10+
# Node.js 18+ (for frontend)
```

### Backend Setup

```bash
cd AraContract_Analyzer/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access API docs at `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

All endpoints are under `/api/contract/`

### 1. Full Analysis (Recommended)

Analyze an entire contract in one call:

```bash
curl -X POST http://localhost:8000/api/contract/analyze \
  -F "file=@contract.pdf"
```

**Response:**

```json
{
  "filename": "contract.pdf",
  "is_scanned": false,
  "clauses": [
    {
      "text": "...",
      "predicted_type_clause": "payment_financial",
      "type_display_name": "الشؤون المالية والدفع",
      "predicted_risk_level": "high",
      "risk_display_name": "عالي",
      "warning": "هذا البند قد يكون مجحفاً..."
    }
  ],
  "summary": "ملخص تنفيذي للعقد...",
  "stats": {
    "total_clauses": 15,
    "high_risk_clauses": 3,
    "medium_risk_clauses": 5,
    "low_risk_clauses": 7,
    "type_distribution": {...}
  }
}
```

### 2. Upload & Extract Text

```bash
curl -X POST http://localhost:8000/api/contract/upload \
  -F "file=@contract.pdf"
```

### 3. Segment

#### A. Segment text

```bash
curl -X POST http://localhost:8000/api/contract/segment \
  -H "Content-Type: application/json" \
  -d '{"text": "المادة الأولى: يلتزم الطرف الأول..."}'
```

#### B. Segment File

```bash
curl -X POST http://localhost:8000/api/contract/segment/file \
  -H "Content-Type: application/json" \
  -F "file=@/path/to/contract.pdf"
```

### 4. Classify Clause(s)

Single clause:

```bash
curl -X POST http://localhost:8000/api/contract/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "يلتزم الطرف الأول بدفع المبلغ خلال 30 يوماً"}'
```

Batch classification:

```bash
curl -X POST http://localhost:8000/api/contract/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["clause 1", "clause 2"]}'
```

### 5. Generate Summary

```bash
curl -X POST http://localhost:8000/api/contract/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "full contract text...",
    "classified_clauses": [...]
  }'
```

### 6. Compare Contracts

```bash
curl -X POST http://localhost:8000/api/contract/compare \
  -F "file1=@contract1.pdf" \
  -F "file2=@contract2.pdf"
```

## Clause Types

| Type                     | Arabic                        | Description                        |
|--------------------------|-------------------------------|------------------------------------|
| `general_provisions`     | أحكام عامة                    | General provisions                 |
| `payment_financial`      | الشؤون المالية والدفع         | Payment & financial terms          |
| `party_obligations_a`    | التزامات الطرف الأول          | Obligations of party A             |
| `party_obligations_b`    | التزامات الطرف الثاني         | Obligations of party B             |
| `duration_expiration`    | المدة والانتهاء                | Duration & expiration              |
| `termination`            | الإنهاء والفسخ                | Termination                        |
| `penalties_damages`      | العقوبات والتعويضات           | Penalties & damages                |
| `dispute_resolution`     | حل النزاعات                   | Dispute resolution                 |

## Risk Levels

| Level    | Arabic | Description              |
|----------|--------|--------------------------|
| `low`    | منخفض  | Standard clauses         |
| `medium` | متوسط  | Requires attention       |
| `high`   | عالي   | Potentially unfair/risky |

## Model Information

- **Base Model**: CAMeL-Lab/bert-base-arabic-camelbert-msa
- **Task**: Multi-task classification (clause type + risk level)
- **Current Version**: v2.3 (best Macro Risk F1)
- **Training**: See `AraContract_Training_Colab.ipynb` for the training notebook
- **Checkpoint**: `models/checkpoints/aracontract_v2.3_best.pt`

---

## Version History

All versions share the same base architecture (CAMeL-Lab/bert-base-arabic-camelbert-msa, 5 epochs, batch size 16, LR 2e-5). The differences are in imbalance handling, threshold strategy, and the selection metric used during training.

### At a Glance

| Version | Type F1 | Macro Risk F1 | Medium F1 | Real Contract HIGH ✓ | Key Change |
|---------|---------|---------------|-----------|----------------------|------------|
| **v1** | 0.8882 | 0.7404 | 0.4516 | ✅ | Baseline, 8 classes (party_obligations split) |
| **v1.5** | 0.8624 | 0.7317 | 0.4818 | — | Class weights + oversampling (over-compensated) |
| **v2** | 0.8635 | 0.7547 | **0.5333** | ❌ | Oversampling only + threshold tuning |
| **v2.1** | **0.8982** | 0.7170 | 0.4176 | — | New `selection_score` metric, medium recall regressed |
| **v2.2** | 0.8910 | 0.7401 | 0.4565 | — | Added `medium_f1` to selection score |
| **v2.3** ★ | 0.8920 | **0.7564** | 0.5047 | ✅ | Hybrid inference (`_apply_legal_overrides` + diacritic removal) |

> **★ v2.3 is the recommended checkpoint.** It achieves the best Macro Risk F1 (0.7564), recovers most of v2's Medium F1 gain, and correctly identifies HIGH risk on a real Arabic contract clause (المادة 11 — automatic termination) — something v2 missed.

---

### Version Comparison Charts

The charts below are generated by `scripts/plot_training_results.py`. Run once to produce the static images:

```bash
cd AraContract_Analyzer
python scripts/plot_training_results.py
```

**Type F1 & Macro Risk F1 across all versions**

```
Type F1  ████████████████████████████████████████  v2.1  0.8982  ← peak
         ████████████████████████████████████████  v2.3  0.8920
         ████████████████████████████████████████  v2.2  0.8910
         ████████████████████████████████████████  v2    0.8635
         ████████████████████████████████████████  v1    0.8882
         ██████████████████████████████████████    v1.5  0.8624

Risk Macro F1
         █████████████████████████████████████████  v2.3  0.7564  ← peak ★
         █████████████████████████████████████████  v2    0.7547
         ████████████████████████████████████████   v2.2  0.7401
         ████████████████████████████████████████   v1    0.7404
         ██████████████████████████████████████     v1.5  0.7317
         ███████████████████████████████████████    v2.1  0.7170
```

**Medium F1 evolution**

```
0.55 ┤
0.53 ┤  ─────── v2 ★ (0.5333)
0.51 ┤
0.50 ┤  ─────── v2.3 (0.5047)
0.48 ┤  ─────── v1.5 (0.4818)
0.46 ┤  ─────── v2.2 (0.4565)
0.45 ┤  ─────── v1   (0.4516)
0.42 ┤  ─────── v2.1 (0.4176)
     └──────────────────────────────────
      v1  v1.5  v2   v2.1  v2.2  v2.3
```

For the interactive HTML report with Chart.js bar charts see [`full_version_comparison_v23.html`](full_version_comparison_v23.html).

---

### v1 (Baseline) — CAMeL-Lab/bert-base-arabic-camelbert-msa

**Test Results**

| Task | Metric | Score |
|------|--------|-------|
| Type Classification | Weighted F1 | 0.8882 |
| Type Classification | Accuracy | 0.8872 |
| Risk Level | Weighted F1 | 0.8702 |
| Risk Level | **Macro F1** | **0.7404** |
| Risk Level | Medium F1 | 0.4516 |

**Key facts**

- 7-class type model → split `party_obligations` into `_a` / `_b` → 8-class model used from this version onward
- No imbalance handling, fixed argmax threshold (0.5)
- Medium recall only **35%** — missed two-thirds of medium-risk clauses
- Real contract HIGH detection: ✅ correct

**Training curve (5 epochs)**

| Epoch | Train Loss | Val Type F1 | Val Risk F1 |
|-------|-----------|-------------|-------------|
| 1 | 1.9824 | 0.7179 | 0.8394 |
| 2 | 1.0379 | 0.8075 | 0.8765 |
| 3 | 0.6993 | 0.8315 | 0.8921 |
| 4 | 0.5135 | 0.8593 | 0.9022 |
| 5 | 0.4123 | 0.8604 | 0.9046 |

---

### v1.5 — Class Weights + Oversampling (Experimental)

**Test Results**

| Task | Metric | Score |
|------|--------|-------|
| Type Classification | Weighted F1 | 0.8624 |
| Type Classification | Accuracy | 0.8619 |
| Risk Level | **Macro F1** | **0.7317** |
| Risk Level | Medium F1 | 0.4818 |

**Key changes**

```python
risk_class_weights = [1.0, 2.0, 1.2]   # penalty boost for medium
oversampling = True                       # WeightedRandomSampler
threshold = 0.39                          # tuned
```

**Learnings**

- Double compensation: class weights × oversampling over-penalised the medium head
- Medium precision dropped to 0.43 — too many false positives
- Task 1 accuracy regressed 2.5% vs v1
- Led directly to the "oversampling only" hypothesis in v2

---

### v2 — Oversampling Only + Threshold Tuning

**Test Results**

| Task | Metric | Score |
|------|--------|-------|
| Type Classification | Weighted F1 | 0.8635 |
| Type Classification | Accuracy | 0.8631 |
| Risk Level | **Macro F1** | **0.7547** |
| Risk Level | **Medium F1** | **0.5333** |

**Key changes from v1.5**

```python
risk_class_weights = [1.0, 1.0, 1.0]   # no class weights
oversampling = True                       # WeightedRandomSampler kept
threshold = 0.43                          # higher than v1.5's 0.39
```

**Per-class Risk (test set)**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| low | 0.9440 | 0.8736 | 0.9074 |
| **medium** | **0.5333** | **0.5333** | **0.5333** |
| high | 0.7437 | 0.9219 | 0.8233 |
| **macro** | 0.7403 | 0.7763 | **0.7547** |

**Training curve**

| Epoch | Train Loss | Val Type F1 | Val Risk F1 | Best Avg F1 |
|-------|-----------|-------------|-------------|-------------|
| 1 | 2.1983 | 0.7219 | 0.7531 | 0.7375 |
| 2 | 0.9885 | 0.7860 | 0.8624 | 0.8242 |
| 3 | 0.5624 | 0.8089 | 0.8889 | 0.8489 |
| 4 | 0.3986 | 0.8263 | 0.8752 | 0.8507 |
| 5 | 0.3261 | 0.8241 | 0.8996 | **0.8618** ★ |

**Real contract test**: ❌ Missed HIGH on المادة 11 (auto-termination clause) — classified as LOW

---

### v2.1 — New `selection_score` Metric

**Test Results**

| Task | Metric | Score |
|------|--------|-------|
| Type Classification | Weighted F1 | **0.8982** ← peak |
| Type Classification | Accuracy | 0.8964 |
| Risk Level | Macro F1 | 0.7170 |
| Risk Level | Medium F1 | 0.4176 |

**Key change**

Replaced simple `(type_f1 + risk_f1) / 2` checkpoint criterion with a domain-aware selection score:

```python
selection_score = 0.5 * type_macro_f1
                + 0.3 * risk_macro_f1_thresholded
                + 0.2 * high_recall_thresholded
```

The new score drove the model toward better Type F1 and high-risk recall, but at the cost of medium F1 regressing to 0.4176 — the `medium` component had no weight in the score.

**Training curve (best epoch: 4)**

| Epoch | Train Loss | Val Type F1 | Val Risk F1 | Selection Score |
|-------|-----------|-------------|-------------|-----------------|
| 1 | 2.3141 | 0.7389 | 0.8315 | 0.6479 |
| 2 | 1.1878 | 0.8153 | 0.8637 | 0.7615 |
| 3 | 0.7941 | 0.8371 | 0.8787 | 0.7944 |
| 4 | 0.5917 | 0.8595 | 0.8888 | **0.8168** ★ |
| 5 | 0.4732 | 0.8576 | 0.8944 | 0.8167 |

---

### v2.2 — Medium F1 Added to `selection_score`

**Test Results**

| Task | Metric | Score |
|------|--------|-------|
| Type Classification | Weighted F1 | 0.8910 |
| Type Classification | Accuracy | 0.8907 |
| Risk Level | Macro F1 | 0.7401 |
| Risk Level | Medium F1 | 0.4565 |

**Key change**

Added `medium_f1_thresholded` as an explicit term in the selection score:

```python
selection_score = 0.4 * type_macro_f1
                + 0.3 * risk_macro_f1_thresholded
                + 0.15 * medium_f1_thresholded
                + 0.15 * high_recall_thresholded
```

Medium F1 recovered partially (0.4176 → 0.4565) but didn't reach v2's 0.5333. Type F1 remained strong. Real contract test not run for this checkpoint.

**Training curve (best epoch: 5)**

| Epoch | Train Loss | Val Type F1 | Val Risk F1 | Selection Score |
|-------|-----------|-------------|-------------|-----------------|
| 1 | 2.3086 | 0.7444 | 0.8249 | 0.5816 |
| 2 | 1.1878 | 0.8084 | 0.8698 | 0.7209 |
| 3 | 0.7825 | 0.8387 | 0.8769 | 0.7588 |
| 4 | 0.5842 | 0.8596 | 0.8926 | 0.7783 |
| 5 | 0.4643 | 0.8536 | 0.9008 | **0.7806** ★ |

---

### v2.3 ★ — Hybrid Inference (Current Recommended)

**Test Results**

| Task | Metric | Score |
|------|--------|-------|
| Type Classification | Weighted F1 | 0.8920 |
| Type Classification | Accuracy | 0.8918 |
| Risk Level | **Macro F1** | **0.7564** ← peak |
| Risk Level | Medium F1 | 0.5047 |

**Key changes**

Same training setup as v2.2 but adds post-model rule-based corrections at inference time:

```python
# 1. Arabic normalisation — strip diacritics before tokenisation
_normalize_arabic(text)   # removes tashkeel, normalises alef/hamza/taa

# 2. Legal pattern overrides — catch patterns the DL head misses
_apply_legal_overrides(text, dl_prediction)
# e.g. "فسخ تلقائي" / "إنهاء فوري" → force termination + HIGH
#      "غرامة" / "تعويض" patterns → force penalties + HIGH
```

**Per-class Risk (test set)**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| low | 0.9349 | 0.9076 | 0.9211 |
| **medium** | **0.5745** | **0.4500** | **0.5047** |
| high | 0.7848 | 0.9115 | 0.8434 |
| **macro** | 0.7647 | 0.7564 | **0.7564** |

**Training curve (best epoch: 5)**

| Epoch | Train Loss | Val Type F1 | Val Risk F1 | Selection Score |
|-------|-----------|-------------|-------------|-----------------|
| 1 | 2.2178 | 0.7424 | 0.8302 | 0.5868 |
| 2 | 1.1547 | 0.8079 | 0.8668 | 0.7272 |
| 3 | 0.7640 | 0.8345 | 0.8824 | 0.7630 |
| 4 | 0.5725 | 0.8494 | 0.8999 | 0.7978 |
| 5 | 0.4558 | 0.8509 | 0.9065 | **0.8081** ★ |

**Why v2.3 wins on the real contract test**

The hybrid approach combines deep learning confidence with explicit Arabic legal pattern matching. The clause *"يلتزم الطرف الأول بإنهاء هذا العقد فوراً عند..."* triggers `_apply_legal_overrides` → `termination` + `HIGH`, which the pure DL model (v2) classified as `LOW` because it was unfamiliar with that specific phrasing after diacritic normalisation.

---

### Why Each Version Existed

```
v1 ──► Good baseline, but medium blindspot
  │
  └─► v1.5: tried class weights — over-compensated medium
        │
        └─► v2: dropped weights, kept oversampling → best medium F1
              │
              └─► v2.1: custom selection_score → best Type F1, medium regressed
                    │
                    └─► v2.2: added medium_f1 to score → partial recovery
                          │
                          └─► v2.3 ★: hybrid inference → best Macro Risk F1
                                        + real-world correctness
```

For the full Arabic analysis, training configs, and lessons learned, see [`VERSION_COMPARISON.md`](VERSION_COMPARISON.md).

---

## Training History & Plots

Run the plotting script to regenerate all charts:

```bash
python scripts/plot_training_results.py
```

This produces:

- **`plots/version_overview.png`** — Type F1 & Macro Risk F1 bar chart across all 6 versions
- **`plots/medium_f1_evolution.png`** — Medium F1 line chart from v1 to v2.3
- **`plots/v2_training_loss.png`** — v2 training loss curves (total, type, risk)
- **`plots/v2_training_dashboard.png`** — Single-page v2 training summary
- **`plots/risk_breakdown_v23.png`** — Per-class risk metrics for v2.3

Interactive HTML report (Chart.js, works offline): [`full_version_comparison_v23.html`](full_version_comparison_v23.html)

### v1 Plots (generated from `Notes.md` logs)

| Plot | Description |
|------|-------------|
| `outputs/epoch_f1.png` | Type & Risk F1 across 5 epochs |
| `outputs/epoch_before_after.png` | Val Type F1 before/after `party_obligations` split |
| `outputs/class_f1.png` | Per-class F1 — 8-class model (test set) |
| `outputs/class_before_after.png` | Per-class F1 comparison: 7-class vs 8-class |
| `outputs/class_delta.png` | F1 change per class after split |
| `outputs/class_table.png` | Full precision / recall / F1 / support table |
| `outputs/test_summary.png` | Test set metrics summary |

## Python Usage (Direct Inference)

```python
from app.models.inference import AraContractInference

# Load model
model = AraContractInference("path/to/checkpoint.pt")

# Single prediction
result = model.predict_single("يلتزم الطرف الأول بدفع...")
print(result["predicted_type_clause"])
print(result["predicted_risk_level"])

# Batch prediction
results = model.predict_batch(["clause 1", "clause 2"])
```

## File Upload Limits

| Setting         | Value                          |
|-----------------|--------------------------------|
| Max file size   | 20 MB                          |
| Allowed formats | PDF, PNG, JPG, JPEG, TIFF, BMP |

## Configuration

Environment variables (via `.env`):

```bash
# Server
HOST=0.0.0.0
PORT=8000

# CORS
ALLOWED_ORIGINS=["http://localhost:3000"]

# Model paths
CLASSIFIER_MODEL_PATH=../models/checkpoints/aracontract_v1_best.pt

# Processing
MAX_SEQ_LENGTH=512
CHUNK_SIZE=500
```

## Project Structure

```text
AraContract_Analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── routers/         # API endpoints
│   │   ├── models/          # Inference & schemas
│   │   ├── services/        # Business logic
│   │   └── core/            # Config & exceptions
│   └── tests/
├── models/
│   ├── checkpoints/         # Trained models
│   ├── config.py            # Training config
│   ├── train.py             # Training script
│   └── inference.py         # Inference wrapper
├── frontend/                # React UI
├── data/                    # Training data
└── scripts/                 # Utility scripts
```

## License

MIT
