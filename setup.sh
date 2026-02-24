#!/usr/bin/env bash
# =============================================================================
# project_echo: Universal Setup Script
# =============================================================================
# Automatically detects platform and installs everything:
#   - PyTorch (CUDA/MPS/CPU depending on platform)
#   - Project dependencies (pip install -e .)
#   - MedGemma 4B (full or 8-bit quantized)
#
# Usage:
#   bash setup.sh                    # Full precision (needs 10GB+ VRAM)
#   bash setup.sh --quantize         # 8-bit quantized (works on 8GB devices)
#   bash setup.sh --skip-vlm         # Skip MedGemma download
#   bash setup.sh --quantize --dry   # Show what would be done
#
# Supported platforms:
#   - NVIDIA Jetson (Orin Nano, AGX Orin, etc.)
#   - macOS (Apple Silicon M1/M2/M3)
#   - Linux with NVIDIA GPU
#   - Linux/macOS CPU-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; exit 1; }

# ─── Parse Arguments ─────────────────────────────────────────────────────────
QUANTIZE=false
SKIP_VLM=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quantize|-q) QUANTIZE=true; shift ;;
        --skip-vlm)    SKIP_VLM=true; shift ;;
        --dry)         DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quantize, -q   Download/use 8-bit quantized MedGemma (for 8GB devices)"
            echo "  --skip-vlm       Skip MedGemma download entirely"
            echo "  --dry            Dry run — show what would be done"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1" ;;
    esac
done

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          project_echo: Universal Setup Script                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── Detect Platform ─────────────────────────────────────────────────────────
detect_platform() {
    if [[ -f /etc/nv_tegra_release ]]; then
        echo "jetson"
    elif [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            echo "macos_arm"
        else
            echo "macos_x86"
        fi
    elif command -v nvidia-smi &>/dev/null; then
        echo "linux_cuda"
    else
        echo "linux_cpu"
    fi
}

PLATFORM=$(detect_platform)
log "Detected platform: ${PLATFORM}"

case "$PLATFORM" in
    jetson)
        log "  → NVIDIA Jetson (ARM64 + CUDA)"
        cat /etc/nv_tegra_release 2>/dev/null || true
        ;;
    macos_arm)
        log "  → macOS Apple Silicon (MPS acceleration)"
        ;;
    macos_x86)
        log "  → macOS Intel (CPU only)"
        ;;
    linux_cuda)
        log "  → Linux with NVIDIA GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        ;;
    linux_cpu)
        log "  → Linux CPU-only"
        ;;
esac

if [[ "$DRY_RUN" == true ]]; then
    log "[DRY RUN] Would set up for platform: $PLATFORM"
    log "[DRY RUN] Quantize: $QUANTIZE"
    log "[DRY RUN] Skip VLM: $SKIP_VLM"
    exit 0
fi

# ─── Create Virtual Environment ──────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    log "Creating virtual environment..."
    python3 -m venv .venv
else
    log "Virtual environment already exists"
fi

source .venv/bin/activate
pip install --upgrade pip wheel

# ─── Install PyTorch ─────────────────────────────────────────────────────────
install_pytorch() {
    case "$PLATFORM" in
        jetson)
            # Check if already installed with CUDA
            if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
                log "PyTorch with CUDA already installed"
                return
            fi
            
            log "Installing PyTorch for Jetson..."
            # JetPack 6.x wheels — update URL as needed
            # Check https://developer.nvidia.com/embedded/downloads for latest
            TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl"
            pip install --no-cache-dir "$TORCH_URL" || {
                warn "NVIDIA wheel failed, trying pip install torch..."
                pip install torch torchvision
            }
            
            # torchvision for Jetson
            pip install --no-cache-dir 'torchvision>=0.18.0' || {
                log "Building torchvision from source..."
                pip install --no-build-isolation 'git+https://github.com/pytorch/vision.git@v0.18.0'
            }
            ;;
            
        macos_arm|macos_x86)
            log "Installing PyTorch for macOS..."
            pip install torch torchvision
            ;;
            
        linux_cuda)
            log "Installing PyTorch with CUDA..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
            ;;
            
        linux_cpu)
            log "Installing PyTorch (CPU)..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
}

install_pytorch

# Verify PyTorch installation
log "Verifying PyTorch..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  Device: MPS (Apple Silicon)')
else:
    print('  Device: CPU')
"

# ─── Install Project Dependencies ────────────────────────────────────────────
log "Installing project dependencies..."
pip install -e .

# Install bitsandbytes for quantization (if supported)
if [[ "$QUANTIZE" == true ]]; then
    log "Installing bitsandbytes for 8-bit quantization..."
    case "$PLATFORM" in
        jetson|linux_cuda)
            pip install bitsandbytes
            ;;
        macos_arm)
            # bitsandbytes has experimental MPS support
            pip install bitsandbytes || warn "bitsandbytes may not fully support MPS yet"
            ;;
        *)
            warn "bitsandbytes requires CUDA, skipping on this platform"
            ;;
    esac
fi

# ─── Verify echoguard Installation ───────────────────────────────────────────
log "Verifying echoguard installation..."
python3 -c "
from echoguard.config import PROJECT_ROOT
from echoguard.inference import EchoGuardInference
print(f'  PROJECT_ROOT: {PROJECT_ROOT}')
print('  ✅ echoguard installed successfully!')
"

# ─── Download MedGemma ───────────────────────────────────────────────────────
if [[ "$SKIP_VLM" == true ]]; then
    log "Skipping MedGemma download (--skip-vlm)"
else
    # Load HF token
    if [[ -f ".env" ]]; then
        set -a
        source .env
        set +a
    fi
    
    if [[ -z "${HF_TOKEN:-}" ]]; then
        warn "No HF_TOKEN found in .env"
        echo ""
        echo "To download MedGemma, you need a Hugging Face token:"
        echo "  1. Create account: https://huggingface.co/join"
        echo "  2. Accept license: https://huggingface.co/google/medgemma-4b-it"
        echo "  3. Get token: https://huggingface.co/settings/tokens"
        echo "  4. Add to .env: echo 'HF_TOKEN=hf_xxx' >> .env"
        echo "  5. Re-run: bash setup.sh"
        echo ""
    else
        log "HF_TOKEN found: ${HF_TOKEN:0:10}..."
        
        MEDGEMMA_DIR="./local_models/medgemma-4b"
        
        if [[ -d "$MEDGEMMA_DIR" && -f "$MEDGEMMA_DIR/config.json" ]]; then
            log "MedGemma 4B already downloaded at $MEDGEMMA_DIR"
        else
            log "Downloading MedGemma 4B (~8 GB)..."
            pip install -q huggingface_hub
            
            huggingface-cli download google/medgemma-4b-it \
                --local-dir "$MEDGEMMA_DIR" \
                --token "$HF_TOKEN"
            
            log "✅ MedGemma 4B downloaded!"
        fi
        
        # Create quantization config if requested
        if [[ "$QUANTIZE" == true ]]; then
            log "Creating 8-bit quantization config..."
            cat > "$MEDGEMMA_DIR/quantization_config.json" << 'EOF'
{
    "load_in_8bit": true,
    "llm_int8_threshold": 6.0,
    "llm_int8_has_fp16_weight": false
}
EOF
            log "✅ Quantization config created"
            echo ""
            echo -e "${CYAN}To load MedGemma in 8-bit mode:${NC}"
            echo "  from echoguard.vlm_critic import VLMCritic"
            echo "  critic = VLMCritic(quantize='8bit')"
            echo "  critic.load()"
        fi
    fi
fi

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  cd src && uvicorn demo_api:app --host 0.0.0.0 --port 8000"
echo ""
if [[ "$QUANTIZE" == true ]]; then
    echo -e "${YELLOW}8-bit mode enabled — MedGemma will use ~4GB instead of ~10GB${NC}"
fi
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
