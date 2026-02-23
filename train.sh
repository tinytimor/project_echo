#!/usr/bin/env bash
# =============================================================================
# EchoGuard-Peds: Unified Training Pipeline
# =============================================================================
# Extracts VideoMAE embeddings, trains all 8 Model Garden specialists
# (TCN, Temporal, MultiTask, MLP × A4C, PSAX), and evaluates.
#
# Usage:
#   bash train.sh                          # Full pipeline
#   bash train.sh --skip-extract           # Skip embedding extraction
#   bash train.sh --views A4C              # Train only A4C models
#   bash train.sh --models tcn temporal    # Train only specific architectures
#   bash train.sh --eval-only              # Evaluate existing checkpoints
#   bash train.sh --epochs 200 --lr 5e-4   # Custom hyperparameters
# =============================================================================

set -euo pipefail

# ─── Resolve project root (works from any CWD) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# ─── Defaults ────────────────────────────────────────────────────────────────
VIEWS=("A4C" "PSAX")
MODELS=("tcn" "temporal" "multitask" "mlp")
SKIP_EXTRACT=false
EVAL_ONLY=false
EPOCHS=100
LR="1e-3"
BATCH_SIZE=64
PATIENCE=15
HIDDEN_DIM=512
DROPOUT=0.3
NOISE=0.01

DATA_DIR="${PROJECT_ROOT}/data/echonet_pediatric"
EMBEDDINGS_DIR="${PROJECT_ROOT}/data/embeddings_videomae"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
DEVICE="cuda"

# ─── Parse Arguments ────────────────────────────────────────────────────────
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-extract)    SKIP_EXTRACT=true; shift ;;
        --eval-only)       EVAL_ONLY=true; shift ;;
        --views)           shift; VIEWS=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do VIEWS+=("$1"); shift; done ;;
        --models)          shift; MODELS=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do MODELS+=("$1"); shift; done ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --lr)              LR="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --patience)        PATIENCE="$2"; shift 2 ;;
        --hidden-dim)      HIDDEN_DIM="$2"; shift 2 ;;
        --dropout)         DROPOUT="$2"; shift 2 ;;
        --noise)           NOISE="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ─── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       EchoGuard-Peds: Unified Training Pipeline         ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  Views:  ${NC}$(printf '%s ' "${VIEWS[@]}")                                  "
echo -e "${CYAN}║  Models: ${NC}$(printf '%s ' "${MODELS[@]}")                     "
echo -e "${CYAN}║  Epochs: ${NC}${EPOCHS}  LR: ${LR}  Batch: ${BATCH_SIZE}             "
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── Step 1: Extract VideoMAE Embeddings ─────────────────────────────────────
if [[ "$EVAL_ONLY" == false && "$SKIP_EXTRACT" == false ]]; then
    log "Step 1/3: Extracting VideoMAE embeddings..."
    for VIEW in "${VIEWS[@]}"; do
        log "  Extracting ${VIEW} embeddings..."
        python -m echoguard.regression.extract_videomae \
            --views "$VIEW" \
            --output-dir "$EMBEDDINGS_DIR" \
            --temporal
    done
    log "Embedding extraction complete ✓"
else
    log "Step 1/3: Skipping embedding extraction"
fi

# ─── Step 2: Train Model Garden ─────────────────────────────────────────────
if [[ "$EVAL_ONLY" == false ]]; then
    log "Step 2/3: Training Model Garden specialists..."

    for VIEW in "${VIEWS[@]}"; do
        VIEW_LOWER=$(echo "$VIEW" | tr '[:upper:]' '[:lower:]')

        for MODEL in "${MODELS[@]}"; do
            # Map model type to checkpoint directory name
            case "$MODEL" in
                temporal)  CKPT_SUFFIX="" ;;
                tcn)       CKPT_SUFFIX="_tcn" ;;
                multitask) CKPT_SUFFIX="_multitask" ;;
                mlp)       CKPT_SUFFIX="_mlp" ;;
                *)         CKPT_SUFFIX="_${MODEL}" ;;
            esac

            OUTPUT="${CHECKPOINT_DIR}/regression_videomae${CKPT_SUFFIX}_${VIEW_LOWER}"

            log "  Training ${MODEL} ${VIEW} → ${OUTPUT}"
            python -m echoguard.regression.train_garden \
                --view "$VIEW" \
                --model-type "$MODEL" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --output-dir "$OUTPUT" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --lr "$LR" \
                --patience "$PATIENCE" \
                --hidden-dim "$HIDDEN_DIM" \
                --dropout "$DROPOUT" \
                --noise "$NOISE" \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
                || { err "Training failed for ${MODEL} ${VIEW}"; continue; }

            log "  ✓ ${MODEL} ${VIEW} complete"
        done
    done

    log "Model Garden training complete ✓"
else
    log "Step 2/3: Skipping training (eval-only mode)"
fi

# ─── Step 3: Evaluate ───────────────────────────────────────────────────────
log "Step 3/3: Evaluating all specialists..."

for VIEW in "${VIEWS[@]}"; do
    log "  Evaluating ${VIEW} — all model types..."
    python -m echoguard.regression.evaluate_garden \
        --view "$VIEW" \
        --all \
        --embeddings-dir "$EMBEDDINGS_DIR" \
        --checkpoint-dir "$CHECKPOINT_DIR/regression" \
        --device "$DEVICE" \
        || warn "Evaluation failed for ${VIEW}"
done

# ─── Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║               Training Pipeline Complete!               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Checkpoints saved in: ${CHECKPOINT_DIR}/"
log "Run the demo:  cd src && uvicorn demo_api:app --host 0.0.0.0 --port 8000"
