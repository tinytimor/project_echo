# project_echo: AI-Assisted Pediatric Cardiac Function Assessment

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

**For clinicians:** project_echo is an agentic "second opinion" — a $2k
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
│                 project_echo: Agentic Pipeline                         │
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
- A $2k handheld POCUS probe + laptop running project_echo fills the gap

### 6.2 Clinical Workflow: How Clinicians Use Echocardiography

#### 6.2.1 The Patient Journey

A pediatric echocardiogram typically begins with a referral from a pediatrician
or neonatologist who detects a murmur, abnormal pulse oximetry, or symptoms
suggestive of cardiac dysfunction (poor feeding, cyanosis, exercise intolerance).
The clinical workflow proceeds as follows:

1. **Referral & Scheduling** — The referring physician orders a transthoracic
   echocardiogram (TTE). Wait times range from days (outpatient) to immediate
   (inpatient/emergent). In underserved areas, families may travel hours to the
   nearest pediatric cardiology center.

2. **Image Acquisition** (30–60 minutes) — A trained cardiac sonographer
   acquires standardized echo views using a phased-array transducer placed
   on the chest wall. Pediatric studies take longer than adult studies because
   children are often uncooperative, heart rates are faster (100–160 BPM in
   infants [6]), and smaller cardiac structures demand higher spatial
   resolution [9]. The sonographer captures 2D B-mode video loops, M-mode
   tracings, and Doppler flow measurements from multiple acoustic windows.

3. **Interpretation** (15–30 minutes) — A pediatric cardiologist (or fellow
   under attending supervision) reviews the acquired images on a PACS
   workstation. They assess chamber sizes, wall motion, valve function, and
   calculate ejection fraction. This step requires the 3+ year pediatric
   cardiology fellowship [7].

4. **Reporting & Communication** (24–48 hours typical turnaround) — The
   cardiologist dictates a structured report with measurements, diagnoses,
   and recommendations. Urgent findings are communicated immediately by phone.

5. **Follow-up** — Depending on findings, patients may return for serial
   echos (weeks to months) to monitor disease progression or treatment response.

#### 6.2.2 The A4C View (Apical Four-Chamber)

The **Apical Four-Chamber (A4C)** view is obtained by placing the transducer
at the cardiac apex (left lateral chest wall, near the nipple line) with the
beam directed superiorly toward the right shoulder [10]. This window
simultaneously visualizes all four cardiac chambers and both atrioventricular
valves in a single plane, making it the most information-dense standard view
in echocardiography.

**What clinicians assess in the A4C view:**

- **Left ventricular (LV) size and function** — The A4C provides the long-axis
  view of the LV, essential for biplane Simpson's method of discs, the ASE/EACVI
  recommended technique for EF measurement in adults [11] and the basis of
  pediatric EF quantification guidelines [10]. The endocardial border is traced
  at end-diastole (ED, maximum filling) and end-systole (ES, maximum contraction)
  to calculate EDV, ESV, and EF = (EDV − ESV) / EDV.

- **Regional wall motion** — The A4C displays the basal, mid, and apical
  segments of the LV septum and lateral wall. Reduced or absent wall thickening
  in specific segments indicates ischemia or prior infarction.

- **Mitral and tricuspid valve function** — Regurgitation, stenosis, and
  leaflet morphology are assessed with 2D and color Doppler.

- **Right ventricular size and function** — RV dilation or dysfunction may
  indicate pulmonary hypertension or congenital anomalies.

- **Atrial sizes** — Left atrial enlargement suggests chronic volume overload
  (e.g., significant mitral regurgitation or left-to-right shunts).

- **Septal defects** — Atrial septal defects (ASD) and ventricular septal
  defects (VSD) are directly visualized with 2D and color Doppler.

**Why A4C is critical for project_echo:** The A4C provides the primary
long-axis LV geometry needed for volumetric EF estimation. Our TCN model
achieves MAE 5.49% on A4C because the view captures global LV contractile
function — the temporal pattern of endocardial inward motion from ED to ES
is the fundamental signal for EF.

#### 6.2.3 The PSAX View (Parasternal Short-Axis)

The **Parasternal Short-Axis (PSAX)** view is obtained from the left parasternal
window (2nd–4th intercostal space) with the transducer rotated 90° clockwise
from the parasternal long-axis position [10]. This produces a cross-sectional
"donut" view of the LV, cutting perpendicular to its long axis.

**What clinicians assess in the PSAX view:**

- **Regional wall motion analysis** — The PSAX at the papillary muscle level
  displays all 6 mid-ventricular wall segments simultaneously (anterior,
  anterolateral, inferolateral, inferior, inferoseptal, anteroseptal). This
  is the single best view for detecting regional wall motion abnormalities
  because all coronary artery territories are represented in one image.

- **LV cross-sectional area** — The short-axis view provides the
  cross-sectional area needed for the area-length (5/6 × A × L) method of
  EF calculation [1], which is the method used in the EchoNet-Pediatric
  dataset. At ED and ES, the endocardial border is traced to measure
  the LV cavity area, and combined with the long-axis length from the A4C,
  LV volumes are estimated.

- **Ventricular geometry** — A normal LV appears circular in PSAX. Deviation
  toward D-shaped geometry indicates RV pressure or volume overload (septal
  flattening). This is particularly important in pediatric patients with
  congenital heart disease.

- **Papillary muscle morphology** — Papillary muscle abnormalities can
  indicate hypertrophic cardiomyopathy (HCM) or mitral valve dysfunction.

- **Valve-level assessment** — At the aortic valve level, PSAX shows the
  aortic valve cusps, interatrial septum, tricuspid and pulmonary valves,
  enabling assessment of bicuspid aortic valve and proximal coronary arteries.

**Why PSAX is critical for project_echo:** The PSAX provides cross-sectional
LV geometry complementary to the A4C long-axis view. Our Temporal Transformer
achieves the best overall performance (MAE 5.08%) on PSAX because the
concentric contraction pattern — the "donut" squeezing symmetrically — is
a strong temporal signal for EF. The geometric EF pipeline (DeepLabV3
segmentation) also achieves its best IoU (0.828) on PSAX because the
near-circular LV cross-section is easier to segment than the complex
A4C geometry.

#### 6.2.4 Why Both Views Are Needed

The EchoNet-Pediatric dataset uses the **biplane area-length (5/6 × A × L)**
method [1] to compute ground-truth EF, which inherently requires both views:
the PSAX provides the cross-sectional area (A) and the A4C provides the
long-axis length (L). This dual-view requirement is standard in pediatric
echocardiographic guidelines [10].

Clinically, A4C and PSAX are complementary:
- **A4C** excels at global LV function, valve assessment, and septal defects
- **PSAX** excels at regional wall motion and ventricular geometry
- **Together** they provide the geometric data for volumetric EF estimation
- **Discordance** between views (our dual-view fusion flags |ΔEF| > 10%)
  often indicates a technical limitation (foreshortening, off-axis imaging)
  or true pathology (regional dysfunction visible in one view but not the other)

This is why project_echo's architecture processes both views independently
with separate specialist ensembles, then fuses conservatively — mirroring
how a cardiologist integrates information from multiple acoustic windows.

#### 6.2.5 Inter-Observer Variability: The Case for AI Assistance

A well-documented challenge in echocardiography is **inter-observer
variability** — two expert readers may measure EF values that differ by
5–10 percentage points on the same study [11]. Sources of variability
include endocardial border tracing differences, frame selection for ED/ES,
and geometric assumptions. In pediatric patients, this variability is
often worse due to faster heart rates, smaller structures, and less
cooperative patients.

project_echo addresses this by providing a **reproducible, deterministic
EF estimate** that serves as a calibrated second opinion. The system's
MAE of 5.08–5.49% is within the range of inter-observer variability
reported in expert-vs-expert comparisons, making it a clinically meaningful
benchmark for automated assessment.

### 6.3 Pediatric EF Norms (Age-Adjusted) [10]

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

### Competition Submission

| File | Purpose |
|---|---|
| `SUBMISSION.md` | MedGemma Impact Challenge 2026 writeup (Kaggle template) |
| `checkpoints/README.md` | Model guide — architecture, roles, metrics for each checkpoint |
| `data/README.md` | EchoNet-Pediatric download instructions |
| `local_models/README.md` | Model download guide (MedGemma, VideoMAE, DeepLabV3) |

### Training

| File | Purpose |
|---|---|
| `train.sh` | Unified training pipeline |
| `download_models.sh` | Downloads MedGemma 4B + pre-caches VideoMAE (reads HF_TOKEN from .env) |
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

| Resource | CUDA (Training + Inference) | Apple Silicon (Inference Only) |
|---|---|---|
| **GPU** | NVIDIA RTX 5090 (32 GB VRAM) or equivalent | M1/M2/M3/M4 (16+ GB unified memory) |
| **RAM** | 32 GB minimum | 16 GB minimum (32 GB recommended) |
| **Storage** | ~12 GB (models + dataset + checkpoints) | ~12 GB |
| **Framework** | CUDA 12.x with BF16 support | PyTorch MPS backend |

---

## 8.5 Running the Demo (Backend + Frontend)

### Prerequisites

Ensure all data and models are downloaded and in the expected locations:

```
project_echo/
├── .venv/                               # Python 3.10+ virtual environment
├── local_models/medgemma-4b/            # MedGemma 4B weights (optional, for VLM layer)
├── data/
│   ├── echonet_pediatric/
│   │   ├── A4C/Videos/                  # 3,284 .avi files
│   │   └── PSAX/Videos/                 # 4,526 .avi files
│   └── embeddings_videomae/
│       ├── pediatric_a4c/               # .pt embeddings + manifest.json
│       └── pediatric_psax/              # .pt embeddings + manifest.json
└── checkpoints/
    ├── regression_videomae_tcn_a4c/     # 8 specialist checkpoints (each has best_model.pt)
    ├── regression_videomae_a4c/
    ├── regression_videomae_multitask_a4c/
    ├── regression_videomae_mlp_a4c/
    ├── regression_videomae_psax/
    ├── regression_videomae_tcn_psax/
    ├── regression_videomae_multitask_psax/
    ├── regression_videomae_mlp_psax/
    ├── lv_seg_deeplabv3.pt              # A4C segmentation (IoU 0.809)
    ├── lv_seg_psax_deeplabv3.pt         # PSAX segmentation (IoU 0.828)
    ├── a4c_geo_calibration.json         # Geometric EF calibration
    └── psax_geo_calibration.json
```

### Setup

```bash
cd project_echo

# Create virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch — choose one:
# CUDA (Linux/Windows):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Apple Silicon (macOS):
pip install torch torchvision

# Install project in editable mode
pip install -e .
```

### Start the Server

```bash
source .venv/bin/activate
cd src
uvicorn demo_api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

### What Happens at Startup

1. **Patient manifests** are loaded from `data/embeddings_videomae/*/manifest.json`
   (embedding paths are resolved to absolute paths automatically).
2. **8 specialist models** (TCN, Temporal, MultiTask, MLP × A4C/PSAX) are loaded
   onto the detected device (CUDA → MPS → CPU).
3. **2 DeepLabV3 segmentation models** and **2 geometric calibrations** are loaded.
4. **MedGemma 4B VLM** availability is checked (loaded on-demand during narration).

Expected startup log:
```
INFO:     Loaded 4,467 patients (A4C: 3284, PSAX: 4526) ...
INFO:     8 specialists on [cuda/mps/cpu]
INFO:     2 segmentation models loaded
INFO:     VLM: [READY/UNAVAILABLE]
```

### Apple Silicon Memory Management

On MPS devices with limited unified memory (e.g., 16–24 GB), the backend
automatically manages GPU memory for the VLM:

- Before VLM inference: specialist models are offloaded to CPU
- After VLM inference: specialists are reloaded back to MPS
- VLM uses float16 with greedy decoding (avoids softmax overflow on MPS)

### Demo UI Features

- **Patient browser:** Search/filter 7,810 patients by ID, view, EF range
- **Regression EF:** 8-specialist ensemble prediction with confidence intervals
- **Geometric EF:** DeepLabV3 segmentation overlay + area-based EF cross-check
- **LV Area Timeline:** Animated chart of LV area across all video frames
- **VLM Validation:** MedGemma "Senior Attending" reviews ED/mid/ES frames
  (click "Generate Narrative" — takes ~4 minutes on MPS, ~30s on A100)
- **Clinical Report:** Age-adjusted Z-scores, dual-view fusion, full narrative

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

---

## 10. References

### Dataset

1. **EchoNet-Pediatric:** Reddy C, Lopez L, Ouyang D, Zou JY, He B.
   "Video-Based Deep Learning for Automated Assessment of Left Ventricular
   Ejection Fraction in Pediatric Patients." *Journal of the American Society
   of Echocardiography*. 2023.
   - Dataset page: https://echonet.github.io/pediatric/
   - Download: https://stanfordaimi.azurewebsites.net/
   - License: Stanford Research Use Agreement (non-commercial)

### Models

2. **MedGemma 4B IT** (VLM critic — Layer 2): Sellergren A, Kazemzadeh S,
   Jaroensri T, et al. "MedGemma Technical Report." arXiv:2507.05201, 2025.
   - Model card: https://huggingface.co/google/medgemma-4b-it
   - Paper: https://arxiv.org/abs/2507.05201
   - Training data: CXR (MIMIC-CXR), histopathology (TCGA, CAMELYON),
     dermatology (PAD-UFES-20, SCIN), ophthalmology (EyePACS), CT, knee X-rays.
     **No echocardiography, no ultrasound, no Stanford AIMI pediatric data.**

3. **VideoMAE** (video encoder — Layer 1): Tong Z, Song Y, Wang J, Wang L.
   "VideoMAE: Masked Autoencoders are Data-Efficient Learners for
   Self-Supervised Video Pre-Training." *NeurIPS* 2022.
   - Paper: https://arxiv.org/abs/2203.12602
   - Model: https://huggingface.co/MCG-NJU/videomae-base
   - Pre-trained on Kinetics-400 (human action recognition videos — no
     medical data). 86M parameters, frozen during our training.

4. **DeepLabV3** (LV segmentation — Layer 1.5): Chen LC, Papandreou G,
   Schroff F, Adam H. "Rethinking Atrous Convolution for Semantic Image
   Segmentation." arXiv:1706.05587, 2017.
   - Paper: https://arxiv.org/abs/1706.05587
   - We use DeepLabV3-MobileNetV3 backbone trained on expert LV contours
     from EchoNet-Pediatric VolumeTracings.csv.

### Clinical References

5. **CHD Prevalence (1.35M children):** Gilboa SM, Devine OJ, Kucik JE,
   et al. "Congenital Heart Defects in the United States: Estimating the
   Magnitude of the Affected Population in 2010." *Circulation*.
   2016;134(2):101–109.
   https://doi.org/10.1161/CIRCULATIONAHA.115.019307

6. **Pediatric Heart Rate Norms:** Fleming S, Thompson M, Stevens R, et al.
   "Normal ranges of heart rate and respiratory rate in children from birth
   to 18 years of age." *Lancet*. 2011;377(9770):1011–1018.
   https://doi.org/10.1016/S0140-6736(10)62226-X

7. **Pediatric Cardiology Fellowship:** ACGME Program Requirements for
   Graduate Medical Education in Pediatric Cardiology.
   https://www.acgme.org/specialties/pediatric-cardiology/

8. **HRSA Health Centers:** HRSA Bureau of Primary Health Care, "About the
   Health Center Program," 2024. ~1,400 funded centers, 16,200+ sites.
   https://bphc.hrsa.gov/about-health-centers

9. **POCUS Probe Pricing (~$2,000):** Butterfly Network, Inc. Butterfly iQ3
   handheld ultrasound device. List price approximately $2,000–$3,000.
   https://www.butterflynetwork.com/

10. **Pediatric Echocardiography Quantification Guidelines:** Lopez L,
    Colan SD, Frommelt PC, et al. "Recommendations for Quantification
    Methods During the Performance of a Pediatric Echocardiogram: A Report
    from the Pediatric Measurements Writing Group of the American Society
    of Echocardiography Pediatric and Congenital Heart Disease Council."
    *J Am Soc Echocardiogr*. 2010;23(5):465–495.
    https://doi.org/10.1016/j.echo.2010.03.019

11. **Adult Cardiac Chamber Quantification Guidelines:** Lang RM, Badano LP,
    Mor-Avi V, et al. "Recommendations for Cardiac Chamber Quantification
    by Echocardiography in Adults: An Update from the American Society of
    Echocardiography and the European Association of Cardiovascular Imaging."
    *J Am Soc Echocardiogr*. 2015;28(1):1–39.
    https://doi.org/10.1016/j.echo.2014.10.003
