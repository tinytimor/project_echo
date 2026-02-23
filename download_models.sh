#!/usr/bin/env bash
# =============================================================================
# EchoGuard-Peds: Download Required Models
# =============================================================================
# Downloads MedGemma 4B and pre-caches VideoMAE embeddings encoder.
#
# Prerequisites:
#   1. Create a Hugging Face account: https://huggingface.co/join
#   2. Accept the MedGemma license: https://huggingface.co/google/medgemma-4b-it
#   3. Create an access token: https://huggingface.co/settings/tokens
#   4. Add your token to .env:  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
#
# Usage:
#   bash download_models.sh            # Download all models
#   bash download_models.sh --skip-vlm # Skip MedGemma (only pre-cache VideoMAE)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# ─── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; exit 1; }

# ─── Parse Arguments ────────────────────────────────────────────────────────
SKIP_VLM=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-vlm) SKIP_VLM=true; shift ;;
        -h|--help)
            echo "Usage: bash download_models.sh [--skip-vlm]"
            echo "  --skip-vlm   Skip MedGemma 4B download (only pre-cache VideoMAE)"
            exit 0
            ;;
        *) err "Unknown argument: $1" ;;
    esac
done

# ─── Load .env ───────────────────────────────────────────────────────────────
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
    log "Loading tokens from .env..."
    set -a
    source "$ENV_FILE"
    set +a
else
    warn "No .env file found at ${ENV_FILE}"
    echo "  Create one with:  echo 'HF_TOKEN=hf_your_token_here' > .env"
fi

# ─── Check HF_TOKEN ─────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    err "HF_TOKEN not set. Add it to .env or export it:
    
  1. Get a token at: https://huggingface.co/settings/tokens
  2. Accept the MedGemma license at: https://huggingface.co/google/medgemma-4b-it
  3. Add to .env:  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
fi

log "HF_TOKEN found (${HF_TOKEN:0:6}...)"

# ─── Check pip packages ─────────────────────────────────────────────────────
if ! python -c "import huggingface_hub" 2>/dev/null; then
    log "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          EchoGuard-Peds: Model Download Script          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── 1. Download MedGemma 4B ────────────────────────────────────────────────
MEDGEMMA_DIR="${PROJECT_ROOT}/local_models/medgemma-4b"

if [[ "$SKIP_VLM" == true ]]; then
    log "Skipping MedGemma 4B (--skip-vlm)"
elif [[ -d "$MEDGEMMA_DIR" && -f "$MEDGEMMA_DIR/config.json" ]]; then
    log "MedGemma 4B already downloaded at ${MEDGEMMA_DIR}"
else
    log "Downloading MedGemma 4B (~8 GB)..."
    log "  Model: google/medgemma-4b-it"
    log "  Destination: ${MEDGEMMA_DIR}"
    echo ""

    huggingface-cli download google/medgemma-4b-it \
        --local-dir "$MEDGEMMA_DIR" \
        --token "$HF_TOKEN" \
        || err "Failed to download MedGemma 4B.
  Make sure you have accepted the license at:
  https://huggingface.co/google/medgemma-4b-it"

    log "✓ MedGemma 4B downloaded successfully"
fi

# ─── 2. Pre-cache VideoMAE ──────────────────────────────────────────────────
log "Pre-caching VideoMAE encoder (MCG-NJU/videomae-base, ~330 MB)..."

python -c "
from transformers import VideoMAEModel, VideoMAEImageProcessor
print('  Downloading VideoMAE processor...')
VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
print('  Downloading VideoMAE model...')
VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')
print('  ✓ VideoMAE cached successfully')
" || warn "VideoMAE pre-cache failed (will download on first use)"

# ─── Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Model Download Complete!                   ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════╣${NC}"

if [[ "$SKIP_VLM" == false ]]; then
echo -e "${GREEN}║  ✓ MedGemma 4B       → local_models/medgemma-4b/       ║${NC}"
fi
echo -e "${GREEN}║  ✓ VideoMAE-base     → ~/.cache/huggingface/hub/       ║${NC}"
echo -e "${GREEN}║  • DeepLabV3 backbone will auto-download on first use  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Next steps:"
log "  1. Download the dataset — see data/README.md"
log "  2. Train models:  bash train.sh"
log "  3. Run the demo:  cd src && uvicorn demo_api:app --host 0.0.0.0 --port 8000"
