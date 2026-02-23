# Local Models Directory

This folder holds large pre-trained model weights that are too big for git
(~8 GB+) and must be downloaded separately.

**Quick start:** Run the download script from the project root:
```bash
bash download_models.sh
```
It reads your `HF_TOKEN` from `.env` and handles everything below automatically.

---

## Required Models

### 1. MedGemma 4B (Required for VLM Validation — Layer 2)

**Google's medical vision-language model** used as the "Senior Attending" critic
that visually validates regression EF predictions.

- **Model:** `google/medgemma-4b-it`
- **Size:** ~8 GB
- **Purpose:** Reviews ED/mid/ES frames + regression EF → AGREE / UNCERTAIN / DISAGREE

#### Download Steps

1. **Accept the license** at https://huggingface.co/google/medgemma-4b-it
   (requires a Hugging Face account)

2. **Set your Hugging Face token** (from https://huggingface.co/settings/tokens):
   ```bash
   # Option A: environment variable
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

   # Option B: use .env file in project root
   source ../.env  # loads HF_TOKEN
   ```

3. **Download the model:**
   ```bash
   pip install huggingface_hub
   huggingface-cli download google/medgemma-4b-it \
       --local-dir ./medgemma-4b \
       --token $HF_TOKEN
   ```

#### Expected Structure
```
local_models/
└── medgemma-4b/
    ├── config.json
    ├── generation_config.json
    ├── model-00001-of-00002.safetensors
    ├── model-00002-of-00002.safetensors
    ├── model.safetensors.index.json
    ├── preprocessor_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── tokenizer.model
```

---

## Auto-Downloaded Models (No Action Required)

These models are downloaded **automatically** from Hugging Face / torchvision
the first time they are used. No manual steps needed.

### VideoMAE (MCG-NJU/videomae-base)
- **Size:** ~330 MB
- **Purpose:** Frozen video encoder for spatiotemporal embedding extraction
- **Used by:** `echoguard.regression.extract_videomae`
- **Downloaded to:** `~/.cache/huggingface/hub/`

### DeepLabV3-MobileNetV3 (torchvision)
- **Size:** ~40 MB (backbone weights)
- **Purpose:** LV segmentation backbone, fine-tuned on expert contours
- **Used by:** `echoguard.regression.geometric_ef`
- **Note:** The fine-tuned weights are in `checkpoints/lv_seg_*.pt` (already
  tracked in git). Only the backbone is auto-downloaded.

---

## Summary

| Model | Size | Download | Required For |
|---|---|---|---|
| MedGemma 4B | ~8 GB | **Manual** (see above) | VLM validation (Layer 2) |
| VideoMAE-base | ~330 MB | Automatic | Embedding extraction |
| DeepLabV3 backbone | ~40 MB | Automatic | Geometric EF segmentation |
