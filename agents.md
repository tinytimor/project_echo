# EchoGuard-Peds: AI-Assisted Pediatric Cardiac Function Assessment

> **Version:** 3.0 | **Date:** 2026-02-23
> **Challenge:** MedGemma Impact Challenge 2026

---

## The 30-Second Version

We taught Google's medical AI (MedGemma) to read children's heart ultrasounds
— something it was *never* trained to do. From 7,810 pediatric echo videos,
our system estimates how strongly the heart pumps (ejection fraction) using an
8-specialist Model Garden, validates predictions with MedGemma's VLM as a
"Senior Attending" critic, and provides geometric EF via LV segmentation —
all in a privacy-first architecture that runs entirely on a laptop.

**For clinicians:** EchoGuard-Peds is an agentic "second opinion" — a $2k
handheld probe + laptop replaces the $150k echo machine for pediatric EF triage,
with age-adjusted Z-scores and natural language reasoning.

**For developers:** Frozen VideoMAE encoders extract spatiotemporal video
embeddings. A Model Garden of 4 lightweight regression heads per view (TCN,
Temporal Transformer, Multi-Task, MLP) produces an ensemble EF prediction.
DeepLabV3 segmentation provides geometric EF. MedGemma 4B VLM validates visually.

---

## 1. Project History: Three Strategic Pivots

### 1.1 Pancreatic Cancer (Sessions 1–2) — Dead End
**Goal:** Multi-agent Local Tumor Board for CPTAC-PDA patients.
**Why it failed:** Circular self-distillation (training MedGemma on its own outputs),
only 23 patients, no ground truth text, mocked pathology.

### 1.2 CXR Reports (Session 2–3) — Abandoned
**Goal:** MIMIC-CXR report generation with external ground truth.
**Why abandoned:** Data contamination — MedGemma was trained on MIMIC-CXR.

### 1.3 Pediatric Echocardiography (Sessions 3–10) — Current
**The insight:** MedGemma was trained on CT, MRI, CXR, dermatology, pathology,
ophthalmology — but **NOT echocardiography**. Perfect transfer learning test.
Clean, uncontaminated dataset of 7,810 pediatric echo videos.

---

## 2. Architecture Evolution: VLM Failure → Regression

### 2.1 Phase 1: VLM Text Generation (Failed)

Fine-tuned MedGemma 4B with LoRA to *generate* EF as text. Every approach
produced **negative R²** — literally worse than predicting the population mean.

| Approach | MAE | R² | Verdict |
|---|---|---|---|
| Zero-shot 4B | 9.33% | -0.22 | Worse than mean |
| LoRA SFT v1 | 7.33% | -0.17 | Clustered at mean |
| GRPO v1–v3 | 6.95% | neg | Mode collapse: 3 unique predictions |
| GRPO v4 | 17.32% | -1.64 | Catastrophic |
| SFT v2 | 10.46% | -0.40 | Collapsed to 5 values |

**Root cause:** Cross-entropy loss treats "60%" and "62%" as equally distant as
"60%" and "banana". The model cannot learn regression through token generation.

### 2.2 Phase 2: Frozen Encoder + Regression (Current — Works)

**Key insight:** MedGemma-4b contains vision encoders pre-trained on medical images.
Freeze them. Use as feature extractors. Train only lightweight regression heads.

Huber loss knows 60% vs 62% is better than 60% vs 38%. Cross-entropy does not.

---

## 3. Current Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 EchoGuard-Peds: Agentic Pipeline                       │
│                                                                         │
│  LAYER 1 — MEASURE                                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  AVI Video → VideoMAE Encoder (MCG-NJU/videomae-base, frozen)    │  │
│  │  → 16-frame spatiotemporal embeddings (16, 768)                  │  │
│  │                                                                   │  │
│  │  Model Garden (4 specialists per view):                           │  │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌─────────┐             │  │
│  │  │   TCN   │ │ Temporal │ │ MultiTask │ │   MLP   │             │  │
│  │  │ Pattern │ │  Motion  │ │ Guardrail │ │Baseline │             │  │
│  │  │ Matcher │ │ Analyst  │ │Classifier │ │         │             │  │
│  │  └────┬────┘ └────┬─────┘ └─────┬─────┘ └────┬────┘             │  │
│  │       └──────┬─────┴──────┬──────┘            │                  │  │
│  │              ▼            ▼                   ▼                  │  │
│  │         Weighted Ensemble EF (per view)                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  LAYER 1.5 — GEOMETRIC VERIFICATION                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  AVI → DeepLabV3 segmentation → LV area per frame                │  │
│  │  → ED/ES detection → Area-based EF → Calibrated → Graduated Blend│  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  LAYER 2 — VALIDATE                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  MedGemma 4B VLM ("Senior Attending")                             │  │
│  │  Reviews ED/mid/ES frames + regression EF + demographics          │  │
│  │  → AGREE / UNCERTAIN / DISAGREE + LV description + reasoning      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  LAYER 3 — SYNTHESIZE                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Dual-View Fusion (A4C + PSAX) → Conservative consensus          │  │
│  │  Age-adjusted Z-scores + BSA indexing + Clinical narrative        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  OUTPUT: Structured Pediatric Echo Report                               │
│  • EF estimate ± confidence interval (8 specialist votes)               │
│  • Age-adjusted Z-score and clinical category                           │
│  • Geometric EF cross-check                                             │
│  • VLM visual validation with reasoning                                 │
│  • Dual-view fusion with disagreement flags                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Embedding Extraction

```
AVI Video (112×112, T frames)
    │
    ▼
extract_all_frames() → List[np.ndarray]
    │
    ▼
get_ed_es_frames_from_tracings() → (ED_index, ES_index)
    │
    ▼
sample_frames_for_videomae(n=16) → 16 frames spanning cardiac cycle
    │
    ▼
VideoMAE (MCG-NJU/videomae-base, 86M params, frozen)
    │
    ▼
Mean-pool spatiotemporal tokens → (16, 768)
    │
    ▼
torch.save() → data/embeddings_videomae/{view}/{video_id}.pt
```

### 3.3 Model Garden Architectures

All models consume frozen VideoMAE embeddings (embed_dim=768, 16 frames).

#### TCN (Temporal Convolutional Network) — Best A4C model
```
Input: (B, 16, 768) → permute → (B, 768, 16)
4 × DilatedResidualConv1d blocks (dilation=1,2,4,8)
→ AdaptiveAvgPool1d(1) → LayerNorm(hidden)
→ FC(hidden, 64) → GELU → Dropout → FC(64, 1)
Params: ~1.36M | Loss: Huber(δ=5)
```

#### Temporal Transformer — Best PSAX model
```
Input: (B, 16, 768) — preserves frame sequence
PosEnc(16, 768) → TransformerEncoder(2 layers, 8 heads) → MeanPool
→ LayerNorm(768) → FC(768, 512) → GELU → FC(512, 128) → GELU → FC(128, 1)
Params: ~21.9M | Loss: Huber(δ=5)
```

#### Multi-Task (Joint Regression + Classification)
```
Shared: LN → FC(16×768, 512) → GELU → Dropout
Reg head: FC(512, 128) → GELU → FC(128, 1)       → EF%
Cls head: FC(512, 128) → GELU → FC(128, 4)        → P(reduced/borderline/normal/hyper)
Loss: L_Huber + 0.3 × L_CE
```

#### Base MLP
```
Input: (B, 16×768=12288) — mean-pooled
LayerNorm(12288) → FC(12288, 512) → GELU → Dropout(0.3)
                → FC(512, 128) → GELU → Dropout(0.2)
                → FC(128, 1)
Params: ~4.7M | Loss: Huber(δ=5)
```

### 3.4 Ensemble Weighting

Ensemble weights from validation performance:
```
w_i = (1/MAE_i) × (1 + R²_i) × (1 + ClinAcc_i)
```

### 3.5 Geometric EF (DeepLabV3 Segmentation)

DeepLabV3-MobileNetV3 models trained on expert LV contours from VolumeTracings.csv:
- A4C: IoU=0.809 (`checkpoints/lv_seg_deeplabv3.pt`)
- PSAX: IoU=0.828 (`checkpoints/lv_seg_psax_deeplabv3.pt`)

Graduated ensemble blending:
- Geometric EF < 40%: trust geometric 80% (structural abnormality)
- 40–55%: 50/50 blend
- > 55%: trust regression 70% (better precision in normal range)

### 3.6 VLM Critic (MedGemma 4B)

The VLM does NOT predict a new EF — it validates visual consistency:
1. Receives 3 key frames (ED, mid-systole, ES) + regression EF + demographics
2. Outputs: AGREE / UNCERTAIN / DISAGREE + LV description + reasoning
3. Adjusts confidence (AGREE: ×1.1, UNCERTAIN: ×0.85, DISAGREE: ×0.6)

### 3.7 Dual-View Fusion

When both A4C and PSAX are available:
- Primary = view with lower EF (more pathological)
- |ΔEF| > 10% → flag disagreement, reduce confidence
- Mirrors clinical practice: act on the more concerning reading

---

## 4. Dataset: EchoNet-Pediatric

### 4.1 Overview

| Property | Value |
|---|---|
| **Source** | Stanford AIMI / Lucile Packard Children's Hospital |
| **URL** | https://stanfordaimi.azurewebsites.net/ |
| **Size** | 7,810 labeled echocardiogram videos (2.34 GB) |
| **A4C Videos** | 3,284 (EF 7.0%–73.0%, mean 60.9% ± 10.5%) |
| **PSAX Videos** | 4,526 (EF 4.1%–73.0%, mean 61.3% ± 10.1%) |
| **Resolution** | 112×112 pixels (downsampled, de-identified) |
| **Format** | AVI video files |
| **Metadata** | EF, Sex, Age, Weight (kg), Height (cm) |
| **Annotations** | Expert LV contour tracings (ED and ES frames) |
| **Splits** | 10-fold cross-validation (0–9) |
| **License** | Stanford Research Use Agreement (non-commercial) |
| **Contamination** | **NOT in MedGemma training data** ✅ |

### 4.2 Ground Truth Files

**FileList.csv** — one row per video:
| Column | Type | Description |
|---|---|---|
| `FileName` | str | AVI filename (anonymized hash) |
| `EF` | float | Ejection fraction % (expert-verified) |
| `Sex` | str | `M` or `F` |
| `Age` | float | Age in years (0.0 for neonates) |
| `Weight` | float | Weight in kg |
| `Height` | float | Height in cm |
| `Split` | int | 10-fold CV assignment (0–9) |

**VolumeTracings.csv** — LV endocardial contour points:
| Column | Type | Description |
|---|---|---|
| `FileName` | str | AVI filename |
| `X` | float | Contour point X coordinate |
| `Y` | float | Contour point Y coordinate |
| `Frame` | int | Video frame index |

### 4.3 Split Mapping

| Folds | Split | A4C Count | PSAX Count |
|---|---|---|---|
| 0–7 | TRAIN | 2,580 | ~3,560 |
| 8 | VAL | 336 | ~470 |
| 9 | TEST | 368 | ~496 |

### 4.4 Data Download Instructions

1. Visit https://stanfordaimi.azurewebsites.net/
2. Search for **"EchoNet-Pediatric"**
3. Sign in with a Microsoft account
4. Accept the **Stanford Research Use Agreement** (DUA)
5. Click **"Export Dataset"** → copy the Azure SAS URL
6. Download:

```bash
# Option 1: azcopy (recommended — handles Azure blob storage natively)
azcopy copy "<SAS_URL>" ./data/echonet_pediatric --recursive

# Option 2: wget (if SAS URL points to a zip file)
wget -O echonet_pediatric.zip "<SAS_URL>"
unzip echonet_pediatric.zip -d ./data/echonet_pediatric
```

Expected structure after download:
```
data/echonet_pediatric/
├── A4C/
│   ├── Videos/            (3,284 .avi files)
│   ├── FileList.csv
│   └── VolumeTracings.csv
└── PSAX/
    ├── Videos/            (4,526 .avi files)
    ├── FileList.csv
    └── VolumeTracings.csv
```

---

## 5. Best Results

### 5.1 Specialist Performance (Validation Set)

**A4C View:**

| Role | Architecture | Val MAE | R² | ClinAcc |
|---|---|---|---|---|
| Pattern Matcher | TCN | **5.49%** | **0.437** | **76.2%** |
| Motion Analyst | Temporal | 5.78% | 0.362 | 74.1% |
| Guardrail | MultiTask | 6.14% | 0.315 | 67.9% |
| Baseline | MLP | 6.55% | 0.270 | 67.0% |

**PSAX View:**

| Role | Architecture | Val MAE | R² | ClinAcc |
|---|---|---|---|---|
| Motion Analyst | Temporal | **5.08%** | **0.536** | **74.8%** |
| Pattern Matcher | TCN | 5.14% | 0.499 | 74.6% |
| Guardrail | MultiTask | 5.43% | 0.480 | 69.6% |
| Baseline | MLP | 5.64% | 0.439 | 67.2% |

### 5.2 Comparison: Regression vs VLM (All VLM Failed)

| Approach | MAE | R² | Unique Preds | Verdict |
|---|---|---|---|---|
| VLM Zero-shot | 9.33% | -0.22 | many | Worse than mean |
| VLM LoRA SFT | 7.33% | -0.17 | many | Negative R² |
| VLM GRPO v1–v4 | 6.95–17.3% | all neg | 3 | Mode collapse |
| **VideoMAE TCN (A4C)** | **5.49%** | **+0.44** | diverse | ✅ |
| **VideoMAE Temporal (PSAX)** | **5.08%** | **+0.54** | diverse | ✅ |

---

## 6. Clinical Significance

### 6.1 Why This Matters
- **1.35M+ children** live with congenital heart disease in the US
- Pediatric echo interpretation requires 3+ year fellowship training
- Rural/underserved areas face critical pediatric cardiologist shortages
- $150k+ echo machines have Auto-EF but fail on fast-beating pediatric hearts
- A $2k handheld POCUS probe + laptop running EchoGuard fills the gap

### 6.2 Pediatric EF Norms (Age-Adjusted)

| Age Group | Normal EF | Borderline | Reduced |
|---|---|---|---|
| Neonate (0–28d) | ≥ 55% | 45–55% | < 45% |
| Infant (1m–1y) | ≥ 56% | 45–56% | < 45% |
| Toddler (1–3y) | ≥ 57% | 47–57% | < 47% |
| Child (3–12y) | ≥ 55% | 45–55% | < 45% |
| Adolescent (12–18y) | ≥ 52% | 42–52% | < 42% |

---

## 7. File Inventory

All source code lives under `src/`. Data, checkpoints, and models remain at the
project root. A `PROJECT_ROOT` constant in `config.py` ensures all paths resolve
correctly regardless of working directory.

### Production (Demo UI/Backend — `src/`)

| File | Purpose |
|---|---|
| `src/demo_api.py` | FastAPI backend (all API endpoints) |
| `src/demo_frontend/index.html` | Frontend SPA |
| `src/echoguard/config.py` | Configuration, EF norms, PROJECT_ROOT, split mapping |
| `src/echoguard/confidence.py` | Confidence scoring (MC-Dropout, Z-score clarity) |
| `src/echoguard/zscore.py` | Pediatric Z-score calculation |
| `src/echoguard/inference.py` | Unified inference — loads 8 specialists |
| `src/echoguard/dual_view.py` | Dual-view conservative fusion |
| `src/echoguard/vlm_critic.py` | MedGemma 4B VLM visual validation |
| `src/echoguard/video_utils.py` | AVI frame extraction, tracing parser |
| `src/echoguard/regression/model.py` | Base MLP model definitions |
| `src/echoguard/regression/model_garden.py` | TCN, Temporal, MultiTask architectures |
| `src/echoguard/regression/infer.py` | Inference wrapper |
| `src/echoguard/regression/geometric_ef.py` | DeepLabV3 segmentation + geometric EF |

### Training

| File | Purpose |
|---|---|
| `train.sh` | Unified training pipeline |
| `src/echoguard/regression/extract_videomae.py` | VideoMAE embedding extraction |
| `src/echoguard/regression/train.py` | Base MLP training loop |
| `src/echoguard/regression/train_garden.py` | Model Garden training |
| `src/echoguard/regression/evaluate.py` | Test set evaluation |
| `src/echoguard/regression/evaluate_garden.py` | Model Garden evaluation |

### Checkpoints (10 `.pt` files + 2 calibrations)

| Checkpoint | Architecture | View | MAE | R² | ClinAcc |
|---|---|---|---|---|---|
| `regression_videomae_tcn_a4c/` | TCN | A4C | 5.49% | 0.437 | 76.2% |
| `regression_videomae_a4c/` | Temporal Transformer | A4C | 5.78% | 0.362 | 74.1% |
| `regression_videomae_multitask_a4c/` | Multi-Task | A4C | 6.14% | 0.315 | 67.9% |
| `regression_videomae_mlp_a4c/` | MLP | A4C | 6.55% | 0.270 | 67.0% |
| `regression_videomae_psax/` | Temporal Transformer | PSAX | 5.08% | 0.536 | 74.8% |
| `regression_videomae_tcn_psax/` | TCN | PSAX | 5.14% | 0.499 | 74.6% |
| `regression_videomae_multitask_psax/` | Multi-Task | PSAX | 5.43% | 0.480 | 69.6% |
| `regression_videomae_mlp_psax/` | MLP | PSAX | 5.64% | 0.439 | 67.2% |
| `lv_seg_deeplabv3.pt` | DeepLabV3 (LV seg) | A4C | IoU 0.809 | — | — |
| `lv_seg_psax_deeplabv3.pt` | DeepLabV3 (LV seg) | PSAX | IoU 0.828 | — | — |
| `a4c_geo_calibration.json` | Linear calibration | A4C | — | — | — |
| `psax_geo_calibration.json` | Linear calibration | PSAX | — | — | — |

---

## 8. Hardware Requirements

| Resource | Requirement |
|---|---|
| **GPU** | NVIDIA RTX 5090 (32 GB VRAM) or equivalent |
| **RAM** | 32 GB minimum |
| **Storage** | ~12 GB (models + dataset + checkpoints) |
| **CUDA** | 12.x with BF16 support |

---

## 9. Lessons Learned

### Why VLM Text Generation Failed for Regression
- Cross-entropy loss cannot learn ordinal relationships between numbers
- Token-level prediction requires every digit correct independently
- 86% normal class imbalance → mode collapse to population mean
- GRPO: identical generations → std(r)=0 → zero gradient → no learning

### Why Frozen Encoder + Regression Works
- Huber loss provides proper ordinal regression signal
- Continuous scalar output, not discrete tokens
- VideoMAE captures temporal/motion patterns natively
- Tiny trainable footprint (~1–22M params vs 4B for VLM)
- Training takes minutes, not hours

### Key Technical Decisions
- **VideoMAE over SigLIP:** Video-aware encoder (temporal masking pre-training)
  vs per-frame encoding with zero motion awareness
- **16 frames:** More cardiac cycle coverage than 4 or 8
- **TCN wins A4C, Temporal wins PSAX:** Different views favor different
  temporal architectures
- **Graduated geometric ensemble:** Geometric EF trustworthy at low EF,
  regression more precise in normal range
- **No curriculum learning:** Adult EchoNet-Dynamic data curriculum learning was
  implemented in the training code (`--use-adult-curriculum` flag) but never
  activated. All 8 production specialists were trained exclusively on pediatric
  EchoNet-Pediatric data (`adult_epochs: 0` in all checkpoint configs). The
  pediatric-only training produced better results than expected, making the
  adult pre-training step unnecessary.
- **`src/` layout with PROJECT_ROOT:** All source code lives under `src/`.
  A `PROJECT_ROOT` constant in `config.py` (computed from `__file__`) ensures
  all checkpoint/data/model paths resolve correctly regardless of CWD.
