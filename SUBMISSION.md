# project_echo — MedGemma Impact Challenge 2026

> **Tracks:** Main Track · Agentic Workflow Prize · The Novel Task Prize · The Edge AI Prize

---

### Project Name

**project_echo:** AI-Assisted Pediatric Cardiac Function Assessment

---

### Your Team

| Member | Specialty | Role |
|---|---|---|
| Stefan Lehman | ML Engineering / Healthcare AI | Sole developer — architecture design, model training, pipeline engineering, demo application, evaluation |

---

### Problem Statement

**1.35 million children** [1] in the United States live with congenital heart disease.
Assessing cardiac function requires measuring **ejection fraction (EF)** — how
strongly the heart pumps — via echocardiography. This interpretation demands a
**3+ year pediatric cardiology fellowship** [2] and access to **$150,000+ echo machines**
with automated EF software that fails on fast-beating pediatric hearts (100–160 BPM) [10].

The result: **critical workforce shortages** in rural and underserved communities.
A child in rural Appalachia or sub-Saharan Africa may wait weeks for a pediatric
cardiologist to read their echo — time that matters when heart failure progresses.

**Impact potential:** project_echo turns a **$2,000 handheld POCUS probe** [9] **+ laptop**
into a pediatric EF triage tool. A nurse in a rural clinic captures an echo video,
and our system provides an EF estimate with age-adjusted Z-scores, confidence
intervals, and **MedGemma-powered natural language clinical reasoning** — in seconds,
with no internet required. If deployed across the ~1,400 HRSA-funded health centers
(16,200+ service sites) [3], project_echo could provide initial cardiac screening to
hundreds of thousands of children who currently lack access to pediatric echo
interpretation.

---

### Related Work & Inspiration

project_echo's agentic architecture draws direct inspiration from two
pioneering healthcare AI frameworks, and builds upon the foundational
pediatric echocardiography research that produced our training dataset.

#### Multi-Agent Healthcare Orchestration

**Microsoft Research Healthcare Agent Orchestrator** [13,14]. In May 2025,
Microsoft Research introduced the Healthcare Agent Orchestrator — a multi-agent
framework for clinical decision support that mirrors the structure of real
multidisciplinary team meetings. The system coordinates specialized AI agents
(radiology, pathology, genomics, clinical trials, patient history) through an
orchestrator that manages shared context, assigns tasks, resolves conflicting
outputs, and maintains traceability across agents. Developed in collaboration
with Stanford Health Care, Johns Hopkins, Providence Genomics, and Mass General
Brigham, the framework demonstrated that healthcare AI systems must be
*collaborative* rather than monolithic — reflecting how clinical decisions
actually emerge from structured dialogue between specialists [13]. The
architecture explicitly addressed the limitations of single-agent systems:
error propagation across agents, agent selection optimization ("more agents are
not always better"), and the need for transparent hand-offs with visible
intermediate reasoning [14]. project_echo adapts this multi-specialist
orchestration pattern to pediatric echocardiography: our 4-architecture
Model Garden (TCN, Temporal Transformer, Multi-Task, MLP) functions as a
specialist roundtable — each model contributes independent expertise on EF
estimation, their predictions are surfaced with agreement metrics (inter-specialist σ),
and MedGemma serves as the orchestrating "Senior Attending" who reviews the
collective output and issues a supervisory verdict.

**Google Research Personal Health Agent (PHA)** [15]. In September 2025,
Google Research proposed a Personal Health Agent framework that decomposes
personal health support into three specialist sub-agents — a data scientist,
a domain expert, and a health coach — coordinated by an intelligent
orchestrator. The key finding was that multi-agent collaboration
"significantly outperformed the sum of its parts" compared to both single-agent
systems and parallel multi-agent baselines without dynamic orchestration [15].
This validated a core design principle that project_echo shares: specialist
agents with distinct inductive biases (our TCN for temporal patterns vs.
Transformer for motion attention vs. Multi-Task for classification guardrails),
coordinated through structured ensemble weighting and VLM-based supervisory review,
produce more reliable outputs than any single model alone.

**Google Cloud Agentic AI for Life Sciences** [16]. Google Cloud's November
2025 framework for drug discovery R&D positions MedGemma as "the strategic
intelligence agent" — a specialized knowledge agent that executes deep search
and synthesis across biomedical data when directed by a cognitive orchestrator
(Gemini 2.5 Pro) [16]. This hierarchical pattern — where MedGemma provides
domain-specific medical reasoning under orchestrator direction — directly
inspired project_echo's Layer 2 design. In our system, the regression ensemble
(Layer 1) and geometric segmentation (Layer 1.5) provide quantitative
measurements, while MedGemma acts as the medical intelligence layer that
visually validates whether those measurements are consistent with what it
observes in the echo frames — a critic, not a predictor.

#### Prior AI Research on Pediatric Echocardiography

**EchoNet-Pediatric (Reddy et al., 2023)** [6]. The foundational work that
produced our training dataset introduced a video-based deep learning approach
using an (2+1)D ResNet architecture for automated pediatric EF assessment.
Trained on 7,643 labeled echocardiogram videos from Lucile Packard Children's
Hospital at Stanford, the model achieved expert-level performance in LV
segmentation and EF estimation. The dataset includes both A4C and PSAX views
with expert LV tracings at end-diastole and end-systole, using the "5/6 Area
Length" (bullet) method for ground-truth EF computation [6]. project_echo
extends this work in three significant directions: (1) we replace the (2+1)D
ResNet with frozen VideoMAE [7] spatiotemporal embeddings fed to lightweight
regression heads, achieving comparable accuracy with dramatically fewer
trainable parameters (~1–22M vs. the full end-to-end CNN); (2) we introduce a
multi-model ensemble (4 architectures per view) that provides uncertainty
quantification via inter-specialist agreement; and (3) we add an agentic
validation layer (MedGemma VLM) and geometric cross-check (DeepLabV3
segmentation) that the original single-model approach lacks.

**EchoNet-Dynamic (Ouyang et al., 2020)** [17]. The adult predecessor to
EchoNet-Pediatric demonstrated that deep learning could match expert
cardiologists in EF assessment from echocardiography videos, achieving MAE of
4.1% on 10,030 adult studies. The key insight — that temporal video information
captures cardiac contractile function better than single-frame analysis —
directly informed our use of VideoMAE (a video-native encoder with temporal
masking pre-training) rather than per-frame image encoders. However, models
trained on adult data do not generalize well to pediatric populations due to
faster heart rates, smaller cardiac structures, and greater anatomical
variability [6], motivating our pediatric-only training strategy.

**Transfer Learning for Pediatric Cardiac Assessment** [18]. Adhikari et al.
(2025) applied transfer learning to predict CMR-derived LVEF from
echocardiographic videos in children with Tetralogy of Fallot, demonstrating
that pre-trained adult echocardiography models can be adapted for specific
pediatric populations. This work reinforces project_echo's approach of using
pre-trained video encoders (VideoMAE, trained on non-medical action recognition
videos) as feature extractors, then training lightweight task-specific heads on
pediatric data — a strategy that preserves general spatiotemporal understanding
while adapting to the pediatric domain without requiring massive labeled
datasets.

Collectively, these works establish that (1) agentic multi-specialist
architectures outperform monolithic models in healthcare AI, (2) MedGemma is
effective as a supervisory intelligence agent rather than a direct predictor,
and (3) pediatric echocardiography presents unique challenges that demand
dedicated models and validation strategies. project_echo synthesizes all three
insights into a unified system.

---

### Overall Solution: MedGemma as the Central Agentic Intelligence

project_echo uses **MedGemma 4B** [4,5] as the **centerpiece** of an agentic clinical
pipeline — applying it to **echocardiography**, a modality MedGemma was **never trained on**.
According to the [MedGemma model card](https://huggingface.co/google/medgemma-4b-it) [4],
its SigLIP image encoder was pre-trained on chest X-rays, histopathology, dermatology,
ophthalmology, CT, and knee X-rays — **no ultrasound, no echocardiography, no Stanford
AIMI data**. This is the **first-ever application of MedGemma to cardiac ultrasound**.

#### The Discovery: Why MedGemma Can't Regress — But Can Reason

We first attempted to make MedGemma directly predict EF as text. Every approach failed:

| Approach | MAE | R² | Verdict |
|---|---|---|---|
| Zero-shot 4B | 9.33% | -0.22 | Worse than predicting the mean |
| LoRA SFT | 7.33% | -0.17 | Collapsed to population mean |
| GRPO v1–v4 | 6.95–17.3% | all negative | Mode collapse: only 3 unique values |

**Root cause:** Cross-entropy loss treats "60%" and "62%" as equally wrong as "60%" and
"banana". The VLM cannot learn regression through token generation.

**The key insight:** MedGemma can't *count*, but it can *see*. Its medical vision
encoders — though never trained on ultrasound — correctly identify LV wall motion
patterns, chamber geometry, and contractile dysfunction in echocardiographic frames.
This led to our agentic architecture: use specialized regression models to measure EF
numerically, then **funnel those predictions to MedGemma for visual validation and
clinical reasoning**.

#### The Agentic Architecture: Specialist Roundtable → MedGemma VLM

project_echo operates as a **4-stage agentic pipeline** — modeled after how a real
cardiology department works:

```
Stage 1: MEASURE — Specialist Roundtable (4 regression models per view)
    ↓ predictions + agreement σ + outlier flags
Stage 1.5: VERIFY — DeepLabV3 LV Segmentation → Geometric EF
    ↓ cross-check signal
Stage 2: VALIDATE — MedGemma 4B VLM ("Senior Attending")
    ↓ AGREE / UNCERTAIN / DISAGREE + LV description + reasoning
Stage 3: SYNTHESIZE — MedGemma Clinical Narrator
    → Final report with confidence-adjusted EF + natural language narrative
```

**Stage 1 — The Specialist Roundtable:** Four independent regression specialists
analyze the echo video per view (8 total across A4C and PSAX). Each consumes frozen
VideoMAE [7] embeddings (768-d × 16 frames) but uses a different architecture:

| Specialist Role | Architecture | What It Captures |
|---|---|---|
| **Pattern Matcher** | Temporal Convolutional Network (TCN) — dilated causal convolutions | Multi-scale temporal patterns at different time scales |
| **Motion Analyst** | Temporal Transformer — multi-head self-attention over frame sequence | How wall motion *evolves* across the full cardiac cycle |
| **Guardrail Classifier** | Multi-Task — joint EF regression + clinical category classification | Constrains predictions to clinically plausible values |
| **Sonographer Baseline** | MLP — mean-pooled 2-layer network | Ensemble diversity; sharp disagreement signals uncertainty |

The ensemble combines predictions using composite weighting:
`weight = (1/MAE) × (1+R²) × (1+ClinAcc)`. Their **agreement signals confidence**
(low σ) and **disagreement flags uncertainty** (high σ). Outlier specialists (>2σ from
median) are automatically down-weighted.

**Stage 1.5 — Geometric Verification:** DeepLabV3-MobileNetV3 [8] segmentation models
(trained on expert LV contours) segment the left ventricle in every frame, computing
area-based EF from the physical contraction. A **graduated ensemble** blends geometric
and regression EF: trusting geometric 80% when EF < 40% (structural abnormality),
50/50 at 40–55%, and regression 70% above 55%.

**Stage 2 — MedGemma as "Senior Attending":** This is where MedGemma takes center
stage. The VLM receives:

1. **Three key echo frames** (end-diastole, mid-systole, end-systole) at 448×448
   native resolution — the same frames a cardiologist would examine
2. **The complete specialist roundtable** — all 4 specialist predictions with their
   role labels and EF values
3. **Inter-specialist agreement σ** — with guidance text: when σ > 8%, MedGemma is told
   "high disagreement; use visual evidence to arbitrate"; when σ < 4%, "strong
   specialist agreement — look for subtle findings they may have missed"
4. **Outlier flags** — specialists whose predictions deviate >10% from consensus
5. **Clinical context** — patient age, sex, BSA, Z-score, clinical category

MedGemma is prompted as a *"senior paediatric cardiologist acting as final reviewer."*
It performs two critical analyses:

- **LV Description:** MedGemma describes what it *sees* in the echo frames — LV size,
  wall thickness, wall motion quality, contractile function. This natural language
  description provides the clinical explainability that pure regression models lack.
  Example output: *"The LV appears moderately dilated with globally reduced wall
  motion. Fractional shortening appears visually reduced, consistent with the
  specialist consensus of EF 38%."*

- **Clinical Verdict:** Based on visual evidence, MedGemma renders one of three verdicts:
  - **AGREE** → confidence ×1.10 — visual evidence confirms the regression EF
  - **UNCERTAIN** → confidence ×0.85 — ambiguous findings, recommend cautious interpretation
  - **DISAGREE** → confidence ×0.60 — visual evidence contradicts the regression EF;
    flag for manual review

This mirrors how a senior attending reviews a fellow's measurements — not generating
new numbers, but **validating the reasoning with clinical judgment**.

**Stage 3 — MedGemma Clinical Narrator:** MedGemma synthesizes all pipeline signals
(regression EF, geometric cross-check, VLM verdict, demographic context) into a
coherent clinical narrative. The output includes: per-view regression ensemble EF with
inter-specialist agreement, geometric EF cross-check, VLM visual validation verdict
with LV description, confidence adjustment chain, and age-adjusted Z-score classification.

#### How the Models Are Performing

**A4C View (Apical Four-Chamber):**

| Specialist | Val MAE ↓ | R² ↑ | Clinical Acc |
|---|---|---|---|
| VideoMAE → TCN | **5.49%** | **0.437** | **76.2%** |
| VideoMAE → Transformer | 5.78% | 0.362 | 74.1% |
| VideoMAE → Multi-Task | 6.14% | 0.315 | 67.9% |
| VideoMAE → MLP | 6.55% | 0.270 | 67.0% |

**PSAX View (Parasternal Short-Axis):**

| Specialist | Val MAE ↓ | R² ↑ | Clinical Acc |
|---|---|---|---|
| VideoMAE → Transformer | **5.08%** | **0.536** | **74.8%** |
| VideoMAE → TCN | 5.14% | 0.499 | 74.6% |
| VideoMAE → Multi-Task | 5.43% | 0.480 | 69.6% |
| VideoMAE → MLP | 5.64% | 0.439 | 67.2% |

**Geometric EF (DeepLabV3 Segmentation):** IoU = 0.809 (A4C), 0.828 (PSAX)

All 8 specialists trained on EchoNet-Pediatric [6] (7,810 pediatric echo videos,
Stanford AIMI). **Zero data contamination** — MedGemma was never trained on this dataset [4].
The best ensemble MAE of 5.08% is within the **5–10% inter-observer variability** [12]
reported between expert echocardiographers, making it a clinically meaningful benchmark.

---

### Technical Details

#### Demo Application

The interactive demo application exposes the full agentic pipeline through a
FastAPI backend + single-page frontend (Tailwind CSS + Alpine.js):

- **Specialist Roundtable tab:** Each specialist appears as a color-coded row with live
  EF predictions, horizontal bar visualization, and validated accuracy metrics
  (MAE, R², Clinical Accuracy). Hovering reveals tooltips explaining each architecture.
  Outlier specialists are flagged with red badges. Inter-specialist agreement σ and
  age-adjusted Z-scores are computed and displayed in real-time.

- **Geometric EF & Segmentation tab:** An animated LV segmentation player shows
  DeepLabV3 segmenting the left ventricle frame-by-frame with severity-colored overlays.
  Side-by-side ED/ES key frames show maximum filling vs. maximum contraction.
  An LV Area Timeline chart tracks cavity size across the cardiac cycle, with smooth and
  raw area curves.

- **MedGemma Agentic Pipeline tab:** A step-by-step pipeline trace shows each stage
  (Measure → Geometric → Validate → Narrate) with timing, engine labels, and data payloads.
  MedGemma's LV description and verdict are displayed with confidence adjustment arrows.
  The full VLM prompt is inspectable via an expandable "Show prompt" section.

**Architecture highlights:**
- 100% local inference — no cloud, no internet, no PHI leaves the device
- MedGemma 4B loads lazily on first VLM request (~12 GB VRAM in BF16)
- 8 specialist models pre-loaded at startup (< 100 MB total)
- Full pipeline (Measure + Geometric + Validate + Narrate) completes in ~10 seconds on GPU

#### Deployment Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| MedGemma 4B VRAM requirements | BF16 fits in ~12 GB; INT8 quantization → ~6 GB; regression heads < 100 MB |
| Echo image quality varies | VideoMAE [7] embeddings are resolution-agnostic (112×112 input) |
| Pediatric heart rates 100–160 BPM [10] | All training exclusively on pediatric data [6] |
| Clinical trust requires explainability | MedGemma provides natural language LV descriptions + reasoning; geometric EF provides physics-based cross-check |
| Privacy regulations (HIPAA, GDPR) | 100% local inference — no data transmitted |

#### Reproducibility

All code, trained checkpoints (10 `.pt` files + 2 calibration JSONs), training logs,
and hyperparameters are publicly available. The training pipeline (`train.sh`) reproduces
all 8 specialists from scratch. The dataset [6] is available from Stanford AIMI under a
research use agreement.

**Repository:** https://github.com/tinytimor/project_echo

---

### Experimental Design & Methodology

#### Why This Approach? The Failure Taxonomy That Drove Our Architecture

The experimental design of project_echo was shaped by **systematic failure analysis**.
We did not begin with a regression pipeline — we began with the simplest possible
approach (zero-shot VLM) and progressively pivoted as each method revealed its
limitations. This section documents the experimental progression, the statistical
rationale behind each design decision, and the quantitative/qualitative metrics
used for evaluation.

**Phase 1 — Direct VLM Prediction (6 experiments, all failed):**

We first hypothesized that MedGemma's medical vision encoders, combined with
LoRA fine-tuning or GRPO reinforcement learning, could learn to predict EF
as a text token. Six experiments across three training paradigms were conducted:

| Experiment | Training | Epochs | MAE | R² | Unique Predictions | Failure Mode |
|---|---|---|---|---|---|---|
| Zero-shot 4B | None | — | 9.33% | −0.22 | many | No echo understanding |
| LoRA SFT v1 | 10K samples | 3 | 7.33% | −0.17 | many | Clustered at population mean |
| LoRA SFT v2 | 10K samples | 5 | 10.46% | −0.40 | 5 | Mode collapse |
| GRPO v1 | RL reward | 500 steps | 6.95% | neg | 3 | Reward hacking |
| GRPO v3 | RL + diversity | 1000 steps | 7.12% | neg | 3 | Identical generations → σ(r)=0 → zero gradient |
| GRPO v4 | RL + hard negatives | 800 steps | 17.32% | −1.64 | many | Catastrophic divergence |

**Statistical root cause:** Cross-entropy loss over tokens computes
$L = -\sum_t \log p(y_t | y_{<t})$, where each digit is an independent
classification problem. The loss for predicting "60" vs "62" is identical to
"60" vs "38" — there is no ordinal gradient signal. Additionally, EF is
a **continuous scalar**, but VLM output is a **discrete token sequence**,
creating an impedance mismatch between the task and the output space.

**GRPO-specific failure:** In GRPO, the policy gradient is scaled by the
**standardized reward** $(r - \mu_r) / \sigma_r$. When all K generations for a
prompt produce identical tokens (mode collapse), $\sigma_r = 0$, yielding
$0/0$ → zero gradient → no learning signal. This was observed in v1–v3 where
the model collapsed to 3 unique EF values.

**Phase 2 — Frozen Encoder + Regression (current approach):**

The key insight was to **decompose the task**: use MedGemma's learned visual
representations as a feature backbone, but replace the token-generation head
with a Huber-loss regression head that preserves ordinal relationships.

$$L_\text{Huber}(\delta=5) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

Huber loss is quadratic for small errors (EF off by < 5 points — provides
strong gradient), and linear for large errors (robustness to outliers in the
heavily-skewed dataset). The $\delta=5$ threshold was chosen because a 5-point
EF error is at the boundary of clinical significance.

#### Dataset Characteristics & Class Imbalance

**EchoNet-Pediatric** [6] was selected because (a) it has verified ground-truth
EF from expert LV tracings, (b) MedGemma was never trained on it, and
(c) it is the largest labeled pediatric echo dataset available.

| Property | A4C | PSAX |
|---|---|---|
| Total videos | 3,284 | 4,526 |
| Train (folds 0–7) | 2,580 | 3,559 |
| Validation (fold 8) | 336 | 448 |
| Test (fold 9) | 368 | 519 |
| EF range | 7.0–73.0% | 4.1–73.0% |
| EF mean ± σ | 60.9 ± 10.5% | 61.3 ± 10.1% |
| Age range | 0–18 years | 0–18 years |
| Sex ratio (M/F) | 1,879/1,392 | 2,587/1,928 |

**Severe class imbalance** (a 10:1 ratio) is the defining challenge:

| EF Category | A4C Train | PSAX Train | Imbalance vs Normal |
|---|---|---|---|
| **Reduced** (< 45%) | 197 (7.6%) | 244 (6.9%) | ~10:1 |
| **Borderline** (45–55%) | 182 (7.1%) | 233 (6.5%) | ~11:1 |
| **Normal** (55–70%) | 1,943 (75.3%) | 2,733 (76.8%) | majority |
| **Hyperdynamic** (> 70%) | 258 (10.0%) | 349 (9.8%) | ~8:1 |

This imbalance is why VLM text generation collapsed to the population mean —
predicting "60%" for everything achieves MAE ~6% but R² = 0. Our mitigation
strategies are described in the training methodology below.

#### Feature Extraction: VideoMAE Embeddings

**Why VideoMAE over SigLIP (MedGemma's native encoder):**

| Feature | VideoMAE [7] | SigLIP (MedGemma) |
|---|---|---|
| Temporal awareness | ✅ Tube masking pre-training captures motion | ❌ Per-frame encoding, no temporal modeling |
| Pre-training data | Kinetics-400 (human action recognition) | Medical images (CXR, derm, path) |
| Input format | 16-frame video clip | Single image |
| Embedding dim | 768 | 768 |
| Domain relevance | Actions = temporal patterns = cardiac motion | Medical anatomy (no motion) |

VideoMAE was chosen because EF estimation is fundamentally a **temporal task** —
measuring how the LV contracts over a cardiac cycle. VideoMAE's tube-masking
pre-training learns spatiotemporal representations from video, capturing exactly
the kind of motion patterns that define EF. SigLIP processes frames independently,
discarding the inter-frame motion signal that is the primary discriminant.

**Extraction procedure:**
1. Load AVI video → extract all frames
2. Use expert tracing annotations (VolumeTracings.csv) to identify ED and ES frames
3. Uniformly sample 16 frames spanning the cardiac cycle (ED → ES → next ED)
4. Resize to 224×224, normalize with ImageNet statistics
5. Forward through frozen VideoMAE encoder (MCG-NJU/videomae-base, 86M params)
6. Spatially mean-pool patch tokens → 8 temporal positions × 768 dimensions
7. Save as `.pt` file per video (pre-computed for training efficiency)

#### Training Methodology

**Optimizer and schedule:**

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Weight decay regularization for small models |
| Learning rate | 1e-3 | Aggressive for small heads (< 22M params) |
| Weight decay | 1e-4 | Standard L2 regularization |
| Batch size | 64 | Full utilization of GPU memory |
| Max epochs | 100 | Generous ceiling; early stopping triggers earlier |
| Warmup | 5 epochs | Linear warmup to avoid early instability |
| Scheduler | Cosine annealing | Smooth LR decay; avoids plateau issues |
| Gradient clipping | Max norm 1.0 | Prevents exploding gradients with small models |

**Class imbalance mitigation (3 strategies):**

1. **Inverse-frequency weighted sampling** — `WeightedRandomSampler` with weights
   proportional to $1/n_{\text{class}}$, ensuring each batch sees roughly equal
   representation of reduced, borderline, normal, and hyperdynamic cases. Normalized
   weights: reduced ≈ 0.35, normal ≈ 0.03 — the sampler draws reduced cases ~10×
   more often than normal.

2. **Composite loss with clinical asymmetry:**

$$L_\text{total} = L_\text{Huber}(\delta=5) + 0.1 \cdot L_\text{ordinal} + 0.05 \cdot L_\text{asymmetric} + 0.01 \cdot L_\text{range} + 0.2 \cdot L_\text{boundary}$$

   - $L_\text{ordinal}$: Penalizes crossing EF category boundaries [45%, 55%, 70%]
     with margin=3.0. If the prediction lands on the wrong side of a boundary
     relative to ground truth, an additional penalty activates.
   - $L_\text{asymmetric}$: **Missing reduced EF (false normal) is 3.5× worse than
     calling normal as borderline.** A missed severely reduced EF could delay
     life-saving intervention. Missing hyperdynamic is penalized 2.0×.
   - $L_\text{range}$: Soft penalty for predictions outside [0, 100] EF range.
   - $L_\text{boundary}$: Pushes predictions toward extreme values when warranted
     (reduced < 50%, hyper > 70%), combating the model's tendency to regress to the mean.

3. **Gaussian embedding noise** (σ=0.01) during training: acts as data
   augmentation in the embedding space, improving generalization with no
   additional data required.

**Model selection criterion (early stopping):**

Model selection uses a **composite score** rather than MAE alone:

$$\text{score} = \text{MAE} - 5.0 \times \text{ClinAcc} - 3.0 \times \max(R^2, 0)$$

This means 1 percentage point of clinical accuracy improvement is valued
equivalent to 0.05% MAE reduction, and 0.10 R² improvement ≈ 0.30% MAE.
The best model minimizes this score, balancing regression precision with
clinical classification accuracy. Early stopping patience = 15 epochs.

**Training convergence (TCN A4C, representative):**

| Metric | Epoch 1 | Epoch 5 | Epoch 19 (best) | Epoch 34 (final) |
|---|---|---|---|---|
| Val MAE | 58.35% | 6.48% | **5.49%** | 5.59% |
| Val R² | −34.76 | 0.25 | **0.437** | 0.388 |
| Val ClinAcc | 5.4% | 69.4% | **76.2%** | 72.6% |
| Learning rate | 2e-4 | 1e-3 | 9.5e-4 | 7.9e-4 |

All 8 specialists converged within 19–50 epochs (well under the 100-epoch ceiling).

#### Quantitative Metrics

**Regression metrics (primary):**

| Metric | Definition | Why Used |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Primary — directly interpretable as "average error in EF percentage points" |
| **R²** | $1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | Measures variance explained; critical because negative R² = worse than predicting the mean |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | Penalizes large errors more heavily than MAE |
| **Within ±5%** | $\frac{1}{n}\sum\mathbb{1}[\|y-\hat{y}\| \leq 5]$ | Fraction of predictions within one clinically meaningful step |
| **Within ±10%** | $\frac{1}{n}\sum\mathbb{1}[\|y-\hat{y}\| \leq 10]$ | Fraction within two clinical steps (broader tolerance) |
| **Prediction diversity** | $\sigma(\hat{y})$ and count of unique values | Detects mode collapse — σ < 3.0 flags "possible collapse" |

**Clinical metrics (interpretive):**

| Metric | Definition | Why Used |
|---|---|---|
| **Clinical Accuracy** | Fraction matching the correct EF category (age-adjusted) | The metric clinicians actually care about — is the patient normal, borderline, or reduced? |
| **Abnormal Sensitivity** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ for non-normal EF | Critical safety metric — how many sick children does the system catch? |
| **Abnormal Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ for non-normal EF | Avoids unnecessary referrals (false positives) |
| **Per-category F1** | Harmonic mean of precision and recall per EF category | Handles class imbalance better than accuracy alone |
| **Error by EF range** | MAE computed within EF bins (< 30%, 30–45%, 45–55%, 55–70%, > 70%) | Reveals if the model is accurate across the full pathology spectrum or only for normal hearts |

**Z-score metrics (pediatric-specific):**

EF varies by age — a 50% EF is normal for a neonate but concerning for an
adolescent. We compute age-adjusted Z-scores using a pediatric nomogram
derived from 2,779 normal EchoNet-Pediatric patients (EF 55–73%):

$$\mu_\text{EF}(\text{age, sex, BSA}) = 64.78 - 0.13 \times \text{age} + 0.37 \times \text{male} + 0.66 \times \text{BSA}$$

$$Z = \frac{\text{EF} - \mu_\text{adjusted}}{\sigma}, \quad \sigma = 4.15\%$$

Z-score flags: **Critical** (Z ≤ −3.0), **Reduced** (−3.0 to −2.0),
**Borderline** (−2.0 to −1.5), **Normal** (−1.5 to +1.5),
**Hyperdynamic** (Z ≥ +2.0).

#### Qualitative Metrics: MedGemma VLM Validation

Beyond numerical accuracy, we evaluate the **qualitative clinical reasoning**
produced by MedGemma's VLM critic:

1. **LV Description Quality:** Does MedGemma accurately describe what it sees in the
   echo frames? We assess whether descriptions correctly identify chamber size
   (normal vs dilated), wall motion quality (normal vs reduced), and contractile
   function — despite never having been trained on echocardiographic images.

2. **Verdict Appropriateness:** When MedGemma issues AGREE, UNCERTAIN, or DISAGREE,
   is the verdict consistent with the visual evidence? A DISAGREE on a clearly
   normal-appearing heart with normal regression EF would indicate hallucination.

3. **Confidence Calibration:** After the VLM adjusts confidence (×1.10 for AGREE,
   ×0.85 for UNCERTAIN, ×0.60 for DISAGREE), is the resulting confidence correlated
   with actual prediction accuracy? Well-calibrated confidence enables clinicians
   to appropriately weight the AI's recommendation.

4. **Prompt Transparency:** Every VLM interaction is fully inspectable in the demo
   UI via "Show prompt" — the user can see exactly what information MedGemma
   received, enabling clinical audit and building trust.

#### Geometric EF: Physics-Based Cross-Check

The geometric EF pipeline provides a complementary, physics-based validation
that does not rely on learned regression patterns:

**Segmentation model:** DeepLabV3-MobileNetV3-Large, trained on expert LV
contours from VolumeTracings.csv (10 epochs, batch=32, lr=1e-3, AdamW,
class weights [0.3 background, 2.0 LV]).

**Calibration:** Raw area-based EF systematically overestimates clinical EF
because cross-sectional area change ≠ volumetric change. Linear calibration
fitted on the training set:

| View | Calibration Equation | IoU |
|---|---|---|
| A4C | $\text{EF}_\text{cal} = 0.378 \times \text{EF}_\text{area} + 46.66$ | 0.809 |
| PSAX | $\text{EF}_\text{cal} = 0.752 \times \text{EF}_\text{area} + 20.66$ | 0.828 |

**Graduated ensemble blending (clinically motivated):**

| Geometric EF Range | Regression Weight | Geometric Weight | Rationale |
|---|---|---|---|
| < 40% (structural) | 20% | **80%** | Geometric captures severe dysfunction directly |
| 40–55% (borderline) | 50% | 50% | Equal trust in ambiguous zone |
| ≥ 55% (normal) | **70%** | 30% | Regression more precise for normal variation |

These thresholds were **clinically motivated, not grid-searched** — they
reflect the clinical observation that area-based methods are most reliable
for detecting structural abnormalities (large LV dilation visible in
cross-section) but less precise for distinguishing 58% from 62% in the
normal range.

**Geometric results (PSAX, test set):**

| Method | MAE ↓ | Clinical Accuracy ↑ | Abnormal Sensitivity ↑ |
|---|---|---|---|
| Regression only | 5.69% | 85.0% | 40.8% |
| Geometric only | 5.35% | 81.9% | **65.8%** |
| Graduated ensemble | **4.97%** | **85.7%** | 47.4% |

The geometric pipeline achieves the **highest abnormal sensitivity** (65.8%)
of any single method — meaning it catches more children with reduced EF —
because it directly measures physical contraction rather than relying on
learned patterns that can be biased by the 75% normal class prevalence.

#### Confidence Scoring: Two Independent Signals

The confidence score combines two complementary signals via geometric mean:

1. **Consistency score** (inter-specialist agreement):
   $$\text{consistency} = \sigma\left(k \cdot (c - \sigma_\text{pred})\right), \quad k=0.30, \; c=3.0$$
   When all 4 specialists agree (σ → 0), consistency → 0.95.
   When they disagree by σ > 8%, consistency drops below 0.30.

2. **Z-score clarity** (distance from decision boundary):
   $$\text{z\_conf} = \sigma\left(0.5 \cdot (|Z| - 1.5)\right)$$
   Extreme Z-scores (clearly abnormal or clearly normal) yield high confidence.
   Z near ±1.5 (borderline) yields low confidence.

3. **Combined:**
   $$\text{overall} = \text{clip}\left(\sqrt{\text{consistency} \times \text{z\_confidence}},\; 0.01,\; 0.99\right)$$

#### Dual-View Fusion: Conservative Consensus

When both A4C and PSAX are available, project_echo fuses conservatively:

- **Primary view** = whichever predicts the **lower** (more pathological) EF.
  Clinical rationale: a cardiologist acts on the more concerning reading.
- **Fused EF** = confidence-weighted average:
  $$\text{EF}_\text{fused} = \frac{w_A \cdot \text{EF}_A + w_P \cdot \text{EF}_P}{w_A + w_P}$$
  where $w = \text{confidence}_\text{overall}$ for each view.
- **Cross-view disagreement** (|ΔEF| > 10%) triggers a confidence penalty:
  $$w_\text{penalty} = 1 - 0.4 \times \min\left(\frac{|\Delta\text{EF}|}{30}, 1\right)$$
  Up to −40% confidence reduction at |ΔEF| = 30%, flagging cases that need
  human review.

#### Ablation: Why an Ensemble of 4 Architectures?

The 4-architecture ensemble is not arbitrary — each architecture captures
different temporal features from the same VideoMAE embeddings:

| Architecture | Temporal Modeling | Strength | Weakness |
|---|---|---|---|
| **TCN** | Dilated causal convolutions | Multi-scale patterns; computationally efficient | Fixed receptive field |
| **Transformer** | Multi-head self-attention | Every frame attends to every other; captures global dynamics | More parameters (21.9M) |
| **Multi-Task** | Classification-regularized regression | Clinically constrained; prevents implausible predictions | Joint optimization challenges |
| **MLP** | None (mean-pooled) | Maximally simple; ensemble diversity baseline | Discards temporal order |

The ensemble provides two critical properties:
1. **Accuracy through diversity:** Different architectures make different errors.
   Weighted averaging cancels uncorrelated noise.
2. **Uncertainty quantification:** Inter-specialist agreement σ is a calibrated
   proxy for prediction reliability. Low σ (< 4%) → high confidence → reliable
   prediction. High σ (> 8%) → flag for human review.

---

### Why project_echo Fits Each Track

| Track | Fit |
|---|---|
| **Main Track** | Full end-to-end application: data → training → 8 specialists → VLM validation → demo UI → clinical output |
| **Agentic Workflow** | 4-stage pipeline where MedGemma VLM acts as the "Senior Attending" — receiving specialist roundtable predictions, visually inspecting echo frames, issuing clinical verdicts, and adjusting confidence based on agreement between its visual assessment and the regression ensemble |
| **Novel Task** | First application of MedGemma to echocardiography — a modality absent from its training data — achieving positive R² (0.54) where direct VLM prediction produced only negative R² |
| **Edge AI** | Entire pipeline runs on a single laptop with no internet; designed for $2k POCUS probe + laptop deployment in resource-limited settings |

---

### Links

- **Code Repository:** https://github.com/tinytimor/project_echo
- **Demo Video:** *(to be recorded)*
- **Technical Documentation:** See `agents.md` in the repository for full
  architecture history, model details, and lessons learned

---

### References

1. **CHD Prevalence (1.35 million children):** Gilboa SM, Devine OJ, Kucik JE,
   et al. "Congenital Heart Defects in the United States: Estimating the Magnitude
   of the Affected Population in 2010." *Circulation*. 2016;134(2):101–109.
   https://doi.org/10.1161/CIRCULATIONAHA.115.019307

2. **Pediatric Cardiology Fellowship (3+ years):** ACGME Program Requirements
   for Graduate Medical Education in Pediatric Cardiology.
   https://www.acgme.org/specialties/pediatric-cardiology/program-requirements-and-faqs-and-applications/

3. **HRSA Health Centers (~1,400 centers, 16,200+ sites):** HRSA Bureau of
   Primary Health Care, "About the Health Center Program," 2024.
   https://bphc.hrsa.gov/about-health-centers

4. **MedGemma Model Card (training modalities — no echocardiography):**
   Google, "MedGemma 4B IT," Hugging Face, 2025. Training data includes
   chest X-rays, dermatology, ophthalmology, and histopathology — no
   echocardiography.
   https://huggingface.co/google/medgemma-4b-it

5. **MedGemma Technical Report:** Sellergren A, Kazemzadeh S, Jaroensri T,
   et al. "MedGemma Technical Report." arXiv:2507.05201, 2025.
   https://arxiv.org/abs/2507.05201

6. **EchoNet-Pediatric Dataset & Paper:** Reddy C, Lopez L, Ouyang D,
   Zou JY, He B. "Video-Based Deep Learning for Automated Assessment of Left
   Ventricular Ejection Fraction in Pediatric Patients." *Journal of the
   American Society of Echocardiography*. 2023.
   Dataset: https://echonet.github.io/pediatric/
   Access: https://stanfordaimi.azurewebsites.net/

7. **VideoMAE (feature encoder):** Tong Z, Song Y, Wang J, Wang L.
   "VideoMAE: Masked Autoencoders are Data-Efficient Learners for
   Self-Supervised Video Pre-Training." *NeurIPS* 2022.
   https://arxiv.org/abs/2203.12602
   Model: https://huggingface.co/MCG-NJU/videomae-base
   Pre-trained on Kinetics-400 (human action recognition — no medical data).

8. **DeepLabV3 (LV segmentation):** Chen LC, Papandreou G, Schroff F,
   Adam H. "Rethinking Atrous Convolution for Semantic Image Segmentation."
   arXiv:1706.05587, 2017.
   https://arxiv.org/abs/1706.05587

9. **POCUS Probe Pricing (~$2,000):** Butterfly Network, Inc. Butterfly iQ3
   handheld ultrasound device. List price approximately $2,000–$3,000.
   https://www.butterflynetwork.com/

10. **Pediatric Heart Rate Norms (100–160 BPM):** Fleming S, Thompson M,
    Stevens R, et al. "Normal ranges of heart rate and respiratory rate in
    children from birth to 18 years of age." *Lancet*. 2011;377(9770):1011–1018.
    https://doi.org/10.1016/S0140-6736(10)62226-X

11. **Pediatric Echocardiography Quantification Guidelines:** Lopez L,
    Colan SD, Frommelt PC, et al. "Recommendations for Quantification
    Methods During the Performance of a Pediatric Echocardiogram."
    *J Am Soc Echocardiogr*. 2010;23(5):465–495.
    https://doi.org/10.1016/j.echo.2010.03.019

12. **Adult Cardiac Chamber Quantification Guidelines:** Lang RM, Badano LP,
    Mor-Avi V, et al. "Recommendations for Cardiac Chamber Quantification
    by Echocardiography in Adults." *J Am Soc Echocardiogr*. 2015;28(1):1–39.
    https://doi.org/10.1016/j.echo.2014.10.003

13. **Microsoft Research Healthcare Agent Orchestrator:** Lungren M.
    "Developing next-generation cancer care management with multi-agent
    orchestration." Microsoft Industry Blog, May 19, 2025.
    https://www.microsoft.com/en-us/industry/blog/healthcare/2025/05/19/developing-next-generation-cancer-care-management-with-multi-agent-orchestration/

14. **Healthcare Agent Orchestrator Architecture:** Gu Y (Aiden), Mandel J,
    Wei M. "Healthcare Agent Orchestrator: Multi-agent Framework for
    Domain-Specific Decision Support." Microsoft Tech Community, May 22, 2025.
    https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/healthcare-agent-orchestrator-multi-agent-framework-for-domain-specific-decision/4416668

15. **Google Research Personal Health Agent:** Xu X "Orson", Heydari A.
    "The anatomy of a personal health agent." Google Research Blog,
    September 30, 2025. Paper: https://arxiv.org/abs/2508.20148
    https://research.google/blog/the-anatomy-of-a-personal-health-agent/

16. **Google Cloud Agentic AI for Life Sciences:** Mehrotra P, Ledsam J.
    "Four agentic workflows you can build for life sciences for R&D."
    Google Cloud Blog, November 21, 2025.
    https://cloud.google.com/blog/topics/healthcare-life-sciences/agentic-ai-framework-in-life-sciences-for-rd

17. **EchoNet-Dynamic (Adult Echocardiography):** Ouyang D, He B, Ghorbani A,
    et al. "Video-based AI for beat-to-beat assessment of cardiac function."
    *Nature*. 2020;580:252–256.
    https://doi.org/10.1038/s41586-020-2145-8

18. **Transfer Learning for Pediatric Cardiac Assessment:** Adhikari A,
    Wesley GV III, Nguyen MB, Doan TT, et al. "Predicting cardiac magnetic
    resonance-derived ejection fraction from echocardiogram via deep learning
    approach in Tetralogy of Fallot." *Pediatric Cardiology*. 2025.
    https://doi.org/10.1007/s00246-025-03802-y
