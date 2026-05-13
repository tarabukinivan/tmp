#!/bin/bash
# pod_setup_full_ft.sh -- One-shot setup for 8×H100 pod.
#
# Run this on a fresh RunPod 8×H100 instance to prepare for full fine-tune.
#
# Prerequisites the pod must already have:
#   - CUDA 12.x toolkit
#   - Python 3.10+
#   - Internet access (HuggingFace + pip)

set -e

echo "==============================================================="
echo "  Pod Setup for Full Fine-Tune (8×H100)"
echo "==============================================================="

# Check GPUs
echo "--- nvidia-smi ---"
nvidia-smi -L
NUM_GPU=$(nvidia-smi -L | wc -l)
if [[ "$NUM_GPU" -lt 8 ]]; then
    echo "WARNING: Expected 8 GPUs, found $NUM_GPU"
fi

# System packages
apt-get update -y >/dev/null 2>&1
apt-get install -y --no-install-recommends \
    tmux htop nvtop curl rsync >/dev/null 2>&1
echo "  apt: ok"

# Python deps
pip install --quiet --upgrade pip
# GPU info
echo "--- GPUs ---"
nvidia-smi -L
NUM_GPU=$(nvidia-smi -L | wc -l)
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_GB / 1024))
echo "GPUs: $NUM_GPU, VRAM/GPU: ${VRAM_GB} GB"

# System packages
echo ""
echo "--- apt deps ---"
apt-get update -y >/dev/null 2>&1
apt-get install -y --no-install-recommends \
    tmux htop nvtop curl rsync git ninja-build >/dev/null 2>&1
echo "  apt: ok"

# Python deps — stage 1: torch + core (must be done before flash-attn)
echo ""
echo "--- pip stage 1: torch + core ---"
pip install --quiet --upgrade pip
pip install --quiet --upgrade \
    "torch>=2.3" \
    "transformers>=4.45" \
    "datasets>=2.18" \
    "accelerate>=0.30" \
    "huggingface_hub" \
    "bitsandbytes>=0.43" \
    "safetensors" \
    "tqdm" \
    "aiohttp" \
    "requests" \
    "tiktoken" \
    "protobuf" \
    "sentencepiece" \
    "peft" \
    "packaging" \
    "ninja" \
    "wheel"
echo "  core: ok"

# Stage 2: flash-attn (optional, may fail on Blackwell B300)
echo ""
echo "--- pip stage 2: flash-attn (optional) ---"
# Build needs torch already installed → --no-build-isolation
if pip install --quiet --no-build-isolation "flash-attn>=2.5" 2>/dev/null; then
    python3 -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')"
else
    echo "  flash-attn install FAILED (common on B300 / new CUDA). Continuing without it."
    echo "  Our trainer auto-falls back to default attention. Slower but works."
fi

# Verify torch + GPU
echo ""
echo "--- Verify ---"
python3 -c "
import torch
print(f'torch:  {torch.__version__}')
print(f'cuda:   {torch.cuda.is_available()}')
print(f'device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')
print(f'gpus:   {torch.cuda.device_count()}')
"
python3 -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"


# Download Moonlight base
BASE_DIR=${BASE_DIR:-/root/Moonlight-16B-A3B-Instruct}
if [[ ! -d "$BASE_DIR" ]] || [[ -z "$(ls -A $BASE_DIR/*.safetensors 2>/dev/null)" ]]; then
    echo ""
    echo "Downloading Moonlight base to $BASE_DIR..."
    hf download moonshotai/Moonlight-16B-A3B-Instruct \
        --local-dir "$BASE_DIR"
    echo "  Moonlight: downloaded"
else
    echo "  Moonlight: already present at $BASE_DIR"
fi

mkdir scripts
wget -O /root/scripts/launch_full_ft_single.sh https://raw.githubusercontent.com/tarabukinivan/tmp/refs/heads/main/launch_full_ft_single.sh
wget -O /root/scripts/train_full_ft_single.py https://raw.githubusercontent.com/tarabukinivan/tmp/refs/heads/main/train_full_ft_single.py

# Symlink eval cache + scripts (assume scp'd to /root/)
mkdir -p /root/distil/datasets_v5 /root/distil/scripts /root/distil/configs
for f in teacher_cache_eval.jsonl sft_full_ft.jsonl sft_val.jsonl; do
    if [[ -f "/root/$f" ]] && [[ ! -f "/root/distil/datasets_v5/$f" ]]; then
        ln -sf "/root/$f" "/root/distil/datasets_v5/$f"
        echo "  linked: $f"
    fi
done

# Check disk space
echo ""
echo "--- Disk space ---"
df -h /root | tail -1

# Check VRAM available
echo ""
echo "--- Per-GPU VRAM ---"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader

echo ""
echo "==============================================================="
echo "  Setup complete. Next:"
echo "    cd /root/distil"
echo "    bash scripts/launch_full_ft.sh"
echo "==============================================================="
