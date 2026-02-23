# Checkpoints Directory

This directory contains all trained model weights for the project_echo pipeline.
Each model plays a specific role in the agentic ensemble — like specialists in a
cardiology department collaborating on a case.

---

## How the Models Work Together

```
AVI Video
    │
    ▼
VideoMAE Encoder (frozen, auto-downloaded)
    │
    ▼  Embeddings: (16 frames, 768 dims)
    │
    ├──→ TCN ──────────────┐
    ├──→ Temporal Transformer ──→ Weighted Ensemble EF ──→ Regression EF
    ├──→ Multi-Task ───────┘         (per view)              │
    ├──→ MLP ──────────────┘                                  │
    │                                                         │
    ▼                                                         ▼
DeepLabV3 Segmentation ──→ Geometric EF ──→ Graduated Blend ──→ Final EF
    │
    ▼
MedGemma 4B VLM ──→ AGREE / UNCERTAIN / DISAGREE ──→ Confidence Adjustment
```

---

## Regression Specialists (8 models)

All specialists consume frozen VideoMAE embeddings (768-dim, 16 frames) and
output a single scalar EF prediction. They are trained with Huber loss (δ=5.0).

### A4C View (Apical Four-Chamber) — 4 Specialists

#### `regression_videomae_tcn_a4c/` — **Pattern Matcher** (Primary A4C)

| Metric | Value |
|---|---|
| Architecture | Temporal Convolutional Network (TCN) |
| Parameters | ~1.36M |
| Val MAE | **5.49%** |
| Val R² | **0.4368** |
| Clinical Accuracy | **76.2%** |

**How it works:** 4 dilated residual Conv1d blocks (dilation 1→2→4→8) scan the
temporal embedding sequence for local contraction patterns. The exponentially
growing receptive field captures both frame-to-frame motion and full-cycle dynamics.
Best A4C model — excels at detecting periodic contraction patterns.

#### `regression_videomae_a4c/` — **Motion Analyst**

| Metric | Value |
|---|---|
| Architecture | Temporal Transformer (2-layer, 8-head) |
| Parameters | ~21.9M |
| Val MAE | 5.78% |
| Val R² | 0.3622 |
| Clinical Accuracy | 74.1% |

**How it works:** Positional encoding preserves frame order, then self-attention
lets every frame attend to every other frame — capturing long-range temporal
dependencies like the relationship between end-diastole and end-systole across
the full cardiac cycle.

#### `regression_videomae_multitask_a4c/` — **Guardrail Classifier**

| Metric | Value |
|---|---|
| Architecture | Multi-Task (Joint Regression + Classification) |
| Parameters | ~0.8M |
| Val MAE | 6.14% |
| Val R² | 0.3153 |
| Clinical Accuracy | 67.9% |

**How it works:** Shared backbone feeds two heads — a regression head for EF%
and a 4-class classifier (reduced / borderline / normal / hyperdynamic). The
classification head acts as a "guardrail": if the classifier says "reduced" but
regression says 62%, something is wrong. Joint training with
$L = L_{\text{Huber}} + 0.3 \times L_{\text{CE}}$ anchors the regression.

#### `regression_videomae_mlp_a4c/` — **Baseline Sonographer**

| Metric | Value |
|---|---|
| Architecture | Feed-forward MLP (3-layer) |
| Parameters | ~4.7M |
| Val MAE | 6.55% |
| Val R² | 0.2703 |
| Clinical Accuracy | 67.0% |

**How it works:** Mean-pools all 16 frames into a single 12,288-dim vector,
then 3 fully-connected layers with GELU activation and dropout. No temporal
modeling — serves as a "what if we ignore motion?" baseline. Its divergence
from temporal models flags unusual cases.

---

### PSAX View (Parasternal Short-Axis) — 4 Specialists

#### `regression_videomae_psax/` — **Motion Analyst** (Primary PSAX)

| Metric | Value |
|---|---|
| Architecture | Temporal Transformer (2-layer, 8-head) |
| Parameters | ~21.9M |
| Val MAE | **5.08%** |
| Val R² | **0.5363** |
| Clinical Accuracy | **74.8%** |

**How it works:** Same architecture as A4C Temporal Transformer. PSAX videos
show the LV as a circular cross-section — wall thickening is more uniform and
self-attention captures the symmetric contraction better than directional TCN
convolutions. Best PSAX model.

#### `regression_videomae_tcn_psax/` — **Pattern Matcher**

| Metric | Value |
|---|---|
| Architecture | Temporal Convolutional Network (TCN) |
| Parameters | ~1.36M |
| Val MAE | 5.14% |
| Val R² | 0.4988 |
| Clinical Accuracy | 74.6% |

**How it works:** Same TCN architecture as A4C. Very close to the Temporal
Transformer on PSAX — the circular LV geometry makes both temporal approaches
effective.

#### `regression_videomae_multitask_psax/` — **Guardrail Classifier**

| Metric | Value |
|---|---|
| Architecture | Multi-Task (Joint Regression + Classification) |
| Parameters | ~0.8M |
| Val MAE | 5.43% |
| Val R² | 0.4803 |
| Clinical Accuracy | 69.6% |

**How it works:** Same joint regression + classification architecture as A4C.
Performs notably better on PSAX than A4C — the cleaner circular geometry of
PSAX makes clinical category boundaries easier to learn.

#### `regression_videomae_mlp_psax/` — **Baseline Sonographer**

| Metric | Value |
|---|---|
| Architecture | Feed-forward MLP (3-layer) |
| Parameters | ~4.7M |
| Val MAE | 5.64% |
| Val R² | 0.4394 |
| Clinical Accuracy | 67.2% |

**How it works:** Same MLP baseline. Interestingly performs better on PSAX
than A4C (R² 0.44 vs 0.27), suggesting PSAX embeddings contain more
EF-informative spatial features even without temporal modeling.

---

## Segmentation Models (2 models)

### `lv_seg_deeplabv3.pt` — A4C LV Segmentation

| Metric | Value |
|---|---|
| Architecture | DeepLabV3-MobileNetV3-Large |
| Task | Binary LV endocardial segmentation |
| IoU | 0.809 |
| View | A4C |

**How it works:** Trained on expert LV contour tracings from VolumeTracings.csv.
Segments the left ventricle in each frame → measures LV area → identifies
end-diastolic (max area) and end-systolic (min area) frames → computes
area-based EF: $(A_{ED} - A_{ES}) / A_{ED} \times 100$.

### `lv_seg_psax_deeplabv3.pt` — PSAX LV Segmentation

| Metric | Value |
|---|---|
| Architecture | DeepLabV3-MobileNetV3-Large |
| Task | Binary LV endocardial segmentation |
| IoU | 0.828 |
| View | PSAX |

**How it works:** Same architecture, trained on PSAX contours. Higher IoU than
A4C because the circular PSAX cross-section is geometrically simpler to segment.

---

## Calibration Files (2 files)

### `a4c_geo_calibration.json` / `psax_geo_calibration.json`

Linear calibration parameters (slope + intercept) that map raw area-based
geometric EF to calibrated EF aligned with expert ground truth. Learned via
linear regression on the validation set.

The calibrated geometric EF is blended with regression EF using graduated
weighting:
- Geometric EF < 40%: 80% geometric (structural abnormality → trust geometry)
- 40–55%: 50/50 blend
- \> 55%: 70% regression (better precision in normal range)

---

## Files Per Checkpoint

Each regression checkpoint directory contains:

| File | Purpose |
|---|---|
| `best_model.pt` | Best model weights (selected by composite score) |
| `final_model.pt` | Last-epoch weights (not used in production) |
| `config.json` | Training hyperparameters |
| `training_history.json` | Full training history (loss, metrics per epoch) |
| `epoch_log.csv` | Per-epoch metrics in CSV format (for analysis/plotting) |

---

## Ensemble Weighting

The 4 specialists per view are combined via weighted average:

$$w_i = \frac{1}{\text{MAE}_i} \times (1 + R^2_i) \times (1 + \text{ClinAcc}_i)$$

This gives more weight to models with lower error, higher correlation, and
better clinical category accuracy. The weights are normalized to sum to 1.
