# project_echo 🫀

**EchoGuard-Peds: AI-Assisted Pediatric Cardiac Function Assessment with MedGemma**

> First-ever application of MedGemma to echocardiography — zero training data
> contamination, clinically impactful, privacy-first, runs entirely on a laptop.

---

## Overview

EchoGuard-Peds is an **agentic "second opinion"** system that estimates pediatric
ejection fraction (EF) from echocardiography videos using an 8-specialist Model
Garden, geometric EF via LV segmentation, and MedGemma 4B VLM visual validation.

### Why This Matters

- **1.35M+ children** live with congenital heart disease in the US
- Pediatric echo interpretation requires **specialized training** (3+ year fellowship)
- Rural/underserved areas face critical **workforce shortages**
- $150k+ echo machines have Auto-EF but **fail on fast-beating pediatric hearts**
- A **$2k handheld POCUS probe + laptop** running EchoGuard fills the gap

---

## Model Accuracy

All models trained exclusively on EchoNet-Pediatric data (7,810 videos).
No curriculum learning or adult pre-training was used.

### A4C View (Apical Four-Chamber)

| Role | Architecture | Val MAE ↓ | R² ↑ | Clinical Accuracy |
|------|-------------|-----------|------|-------------------|
| Pattern Matcher | **TCN** | **5.49%** | **0.4368** | **76.2%** |
| Motion Analyst | Temporal Transformer | 5.78% | 0.3622 | 74.1% |
| Guardrail | Multi-Task | 6.14% | 0.3153 | 67.9% |
| Baseline | MLP | 6.55% | 0.2703 | 67.0% |

### PSAX View (Parasternal Short-Axis)

| Role | Architecture | Val MAE ↓ | R² ↑ | Clinical Accuracy |
|------|-------------|-----------|------|-------------------|
| Motion Analyst | **Temporal Transformer** | **5.08%** | **0.5363** | **74.8%** |
| Pattern Matcher | TCN | 5.14% | 0.4988 | 74.6% |
| Guardrail | Multi-Task | 5.43% | 0.4803 | 69.6% |
| Baseline | MLP | 5.64% | 0.4394 | 67.2% |

### Geometric EF (DeepLabV3 Segmentation)

| View | Segmentation IoU | Graduated Ensemble MAE |
|------|-----------------|------------------------|
| A4C | 0.809 | — |
| PSAX | 0.828 | 4.97% |

### Why VLM Text Generation Failed (All Negative R²)

| Approach | MAE | R² | Verdict |
|---|---|---|---|
| VLM Zero-shot | 9.33% | -0.22 | Worse than mean |
| VLM LoRA SFT | 7.33% | -0.17 | Negative R² |
| VLM GRPO v1–v4 | 6.95–17.3% | all neg | Mode collapse |
| **VideoMAE TCN (A4C)** | **5.49%** | **+0.44** | ✅ |
| **VideoMAE Temporal (PSAX)** | **5.08%** | **+0.54** | ✅ |

**Root cause:** Cross-entropy loss treats "60%" and "62%" as equally distant as
"60%" and "banana". Huber loss provides proper ordinal regression signal.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    EchoGuard-Peds Pipeline                           │
│                                                                      │
│  LAYER 1 — MEASURE                                                   │
│  AVI → VideoMAE (MCG-NJU/videomae-base, frozen) → (16, 768)         │
│  → Model Garden: TCN + Temporal + MultiTask + MLP (4 per view)       │
│  → Weighted ensemble EF                                              │
│                                                                      │
│  LAYER 1.5 — GEOMETRIC VERIFICATION                                  │
│  AVI → DeepLabV3 segmentation → LV area → Calibrated EF             │
│  → Graduated blend with regression EF                                │
│                                                                      │
│  LAYER 2 — VALIDATE                                                  │
│  MedGemma 4B VLM ("Senior Attending")                                │
│  → AGREE / UNCERTAIN / DISAGREE + LV description + reasoning         │
│                                                                      │
│  LAYER 3 — SYNTHESIZE                                                │
│  Dual-View Fusion (A4C + PSAX) → Conservative consensus             │
│  → Age-adjusted Z-scores + BSA indexing + Clinical narrative          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Build the Virtual Environment

Requires **Python 3.10+** and an NVIDIA GPU with CUDA 12.x.

```bash
cd project_echo

# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install the package in editable mode (installs all dependencies from pyproject.toml)
pip install -e .

# Verify installation
python -c "import echoguard; print('echoguard installed successfully')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Key dependencies installed by `pip install -e .`:
- `torch`, `transformers`, `accelerate` — model inference
- `opencv-python`, `pillow` — video/image processing
- `numpy`, `pandas`, `scikit-learn` — data handling
- `fastapi`, `uvicorn` — demo API server

### 2. Download Models

```bash
# 1. Accept the MedGemma license: https://huggingface.co/google/medgemma-4b-it
# 2. Add your Hugging Face token to .env:
echo "HF_TOKEN=hf_your_token_here" >> .env

# 3. Run the download script (downloads MedGemma 4B + pre-caches VideoMAE):
bash download_models.sh

# Or skip the VLM if you only need regression inference:
bash download_models.sh --skip-vlm
```

See [local_models/README.md](local_models/README.md) for details on each model.

### 3. Download EchoNet-Pediatric Dataset

The dataset is hosted by Stanford AIMI and requires a Data Use Agreement:

1. Visit **https://stanfordaimi.azurewebsites.net/**
2. Search for **"EchoNet-Pediatric"**
3. Sign in with a **Microsoft account**
4. Accept the **Stanford Research Use Agreement** (DUA)
5. Click **"Export Dataset"** → copy the Azure SAS URL
6. Download:

```bash
# Install azcopy if needed:
# https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10

# Download the dataset (~2.4 GB)
azcopy copy "<SAS_URL>" ./data/echonet_pediatric --recursive

# Alternative: wget (if SAS URL points to a zip)
wget -O echonet_pediatric.zip "<SAS_URL>"
unzip echonet_pediatric.zip -d ./data/echonet_pediatric
```

Expected structure:
```
data/echonet_pediatric/
├── A4C/
│   ├── Videos/            (3,284 .avi files)
│   ├── FileList.csv       (FileName, EF, Sex, Age, Weight, Height, Split)
│   └── VolumeTracings.csv (FileName, X, Y, Frame — LV contour points)
└── PSAX/
    ├── Videos/            (4,526 .avi files)
    ├── FileList.csv
    └── VolumeTracings.csv
```

### 4. Train All Models

```bash
# Full pipeline: extract embeddings → train all 8 specialists → evaluate
bash train.sh

# Skip extraction (re-train with existing embeddings)
bash train.sh --skip-extract

# Train only specific views or model types
bash train.sh --views A4C
bash train.sh --models tcn temporal
bash train.sh --eval-only
```

### Training Reproduction Details

The default hyperparameters in `train.sh` are the **exact values** that produced
all 8 shipped checkpoints. Running `bash train.sh` with no arguments reproduces
the training (given the same data and random seed behavior).

| Hyperparameter | Value | CLI Flag |
|---|---|---|
| Epochs | 100 | `--epochs` |
| Learning Rate | 1e-3 | `--lr` |
| Batch Size | 64 | `--batch-size` |
| Early Stopping Patience | 15 epochs | `--patience` |
| Hidden Dim | 512 | `--hidden-dim` |
| Dropout | 0.3 | `--dropout` |
| Embedding Noise | 0.01 | `--noise` |
| Loss | Huber (δ=5.0) | — |
| Scheduler | Cosine Annealing | — |
| Weight Decay | 1e-4 | — |

**Best-model saving:** During training, a composite score selects the best
checkpoint (lower is better):

$$\text{score} = \text{MAE} - 5.0 \times \text{ClinAcc} - 3.0 \times \max(R^2, 0)$$

This jointly rewards low MAE, high clinical-category accuracy, and high R².
The best model is saved as `best_model.pt` inside each checkpoint directory;
a `final_model.pt` (last epoch) is also saved. Early stopping triggers after
15 epochs with no improvement in composite score.

Each checkpoint stores its full config, metrics, and model weights:
```python
torch.save({
    "epoch": best_epoch,
    "model_state_dict": ...,
    "val_mae": best_val_mae,
    "val_clin_acc": clin_acc,
    "val_r2": r2,
    "composite_score": composite_score,
    "config": config.__dict__,  # all hyperparameters
    "model_type": model_type,
}, "best_model.pt")
```

### 5. Run the Demo

```bash
# Start the FastAPI backend (from src/ directory)
cd src && uvicorn demo_api:app --host 0.0.0.0 --port 8000 --reload

# Or with PYTHONPATH set:
PYTHONPATH=src uvicorn src.demo_api:app --host 0.0.0.0 --port 8000 --reload

# Open http://localhost:8000 in your browser
```

---

## Dataset

### EchoNet-Pediatric

| Property | Value |
|---|---|
| Source | Stanford AIMI / Lucile Packard Children's Hospital |
| Videos | 7,810 labeled echocardiograms |
| Size | ~2.4 GB |
| Format | AVI, 112×112 pixels |
| Labels | EF, expert LV tracings at ED/ES |
| Ages | 0–18 years |
| Views | 3,284 A4C + 4,526 PSAX |
| Contamination | **NOT in MedGemma training data** ✅ |

### Split Mapping (10-Fold CV → 3-Way)

| Folds | Split | A4C | PSAX |
|---|---|---|---|
| 0–7 | TRAIN | 2,580 | ~3,560 |
| 8 | VAL | 336 | ~470 |
| 9 | TEST | 368 | ~496 |

---

## Project Structure

```
project_echo/
├── README.md                      # This file
├── agents.md                      # Full technical history & architecture docs
├── train.sh                       # Unified training pipeline
├── pyproject.toml                 # Package configuration
│
├── src/                           # All source code
│   ├── demo_api.py                # FastAPI backend (all endpoints)
│   ├── demo_frontend/
│   │   ├── index.html             # Frontend SPA
│   │   └── static/                # Static assets
│   └── echoguard/
│       ├── __init__.py
│       ├── config.py              # Configuration, EF norms, PROJECT_ROOT
│       ├── confidence.py          # Confidence scoring (MC-Dropout)
│       ├── zscore.py              # Pediatric Z-score calculation
│       ├── inference.py           # Inference engine — loads 8 specialists
│       ├── dual_view.py           # Dual-view conservative fusion
│       ├── vlm_critic.py          # MedGemma 4B VLM visual validation
│       ├── video_utils.py         # AVI frame extraction, tracing parser
│       └── regression/
│           ├── __init__.py
│           ├── model.py           # Base MLP model definitions
│           ├── model_garden.py    # TCN, Temporal, MultiTask architectures
│           ├── infer.py           # Inference wrapper
│           ├── geometric_ef.py    # DeepLabV3 segmentation + geometric EF
│           ├── extract_videomae.py# VideoMAE embedding extraction
│           ├── train.py           # Base MLP training loop
│           ├── train_garden.py    # Model Garden training
│           ├── evaluate.py        # Test set evaluation
│           └── evaluate_garden.py # Model Garden evaluation
│
├── local_models/
│   └── medgemma-4b/               # MedGemma 4B weights (download separately)
│
├── data/
│   ├── echonet_pediatric/         # Dataset (download from Stanford AIMI)
│   │   ├── A4C/                   # Videos/, FileList.csv, VolumeTracings.csv
│   │   └── PSAX/                  # Videos/, FileList.csv, VolumeTracings.csv
│   └── embeddings_videomae/       # Pre-extracted VideoMAE embeddings
│       ├── pediatric_a4c/         # .pt files + manifest.json
│       └── pediatric_psax/        # .pt files + manifest.json
│
└── checkpoints/
    ├── regression_videomae_tcn_a4c/         # TCN A4C  (MAE 5.49%, best A4C)
    ├── regression_videomae_a4c/             # Temporal A4C
    ├── regression_videomae_multitask_a4c/   # MultiTask A4C
    ├── regression_videomae_mlp_a4c/         # MLP A4C
    ├── regression_videomae_psax/            # Temporal PSAX (MAE 5.08%, best PSAX)
    ├── regression_videomae_tcn_psax/        # TCN PSAX
    ├── regression_videomae_multitask_psax/  # MultiTask PSAX
    ├── regression_videomae_mlp_psax/        # MLP PSAX
    ├── lv_seg_deeplabv3.pt                  # DeepLabV3 A4C (IoU 0.809)
    ├── lv_seg_psax_deeplabv3.pt             # DeepLabV3 PSAX (IoU 0.828)
    ├── a4c_geo_calibration.json             # A4C geometric calibration
    └── psax_geo_calibration.json            # PSAX geometric calibration
```

---

## Pediatric EF Norms (Age-Adjusted)

| Age Group | Normal EF | Borderline | Reduced |
|---|---|---|---|
| Neonate (0–28d) | ≥ 55% | 45–55% | < 45% |
| Infant (1m–1y) | ≥ 56% | 45–56% | < 45% |
| Toddler (1–3y) | ≥ 57% | 47–57% | < 47% |
| Child (3–12y) | ≥ 55% | 45–55% | < 45% |
| Adolescent (12–18y) | ≥ 52% | 42–52% | < 42% |

## Hardware Requirements

| Resource | Requirement |
|---|---|
| **GPU** | NVIDIA RTX 5090 (32 GB VRAM) or equivalent |
| **RAM** | 32 GB minimum |
| **Storage** | ~12 GB (models + dataset + checkpoints) |
| **CUDA** | 12.x with BF16 support |

## License

Research use only. EchoNet-Pediatric data subject to Stanford AIMI Research Use Agreement.
