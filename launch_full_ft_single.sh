#!/bin/bash
# launch_full_ft_single.sh -- Launch full fine-tune on a SINGLE GPU.
#
# Auto-detects VRAM and chooses optimizer:
#   B300 (288GB): AdamW FP32 (best quality)
#   B200 (180GB): AdamW FP32 (tight, fallback to 8-bit if OOM)
#   H200 (141GB): AdamW 8-bit (paged)
#
# Usage:
#   bash scripts/launch_full_ft_single.sh

set -e

BASE_MODEL=${BASE_MODEL:-/root/Moonlight-16B-A3B-Instruct}
DATASET=${DATASET:-/root/sft_train.jsonl}
VAL_DATASET=${VAL_DATASET:-/root/sft_val.jsonl}
OUTPUT=${OUTPUT:-/root/distil-full-ft}

MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-16}
EPOCHS=${EPOCHS:-2}
LR=${LR:-5e-6}
SAVE_EVERY=${SAVE_EVERY:-500}
EVAL_EVERY=${EVAL_EVERY:-500}

mkdir -p "$OUTPUT" /root/logs

# Pre-flight
if [[ ! -d "$BASE_MODEL" ]]; then
    echo "ERROR: base model not found at $BASE_MODEL"
    echo "Download: huggingface-cli download moonshotai/Moonlight-16B-A3B-Instruct --local-dir $BASE_MODEL"
    exit 1
fi
if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found at $DATASET"
    exit 1
fi

# Auto-detect VRAM for optimizer choice
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_GB / 1024))
echo "Detected GPU VRAM: ${VRAM_GB} GB"
if [[ -z "$OPTIMIZER" ]]; then
    if [[ "$VRAM_GB" -ge 240 ]]; then
        OPTIMIZER="adamw"         # B300, plenty of room
    elif [[ "$VRAM_GB" -ge 150 ]]; then
        OPTIMIZER="adamw"         # B200, fits FP32
    else
        OPTIMIZER="adamw_8bit"    # H200 or smaller
    fi
fi
echo "Optimizer: $OPTIMIZER"

VAL_ARG=""
if [[ -f "$VAL_DATASET" ]]; then
    VAL_ARG="--val-dataset $VAL_DATASET"
fi

echo "==============================================================="
echo "  Full Fine-Tune (single GPU)"
echo "==============================================================="
echo "  GPU VRAM      : ${VRAM_GB} GB"
echo "  base_model    : $BASE_MODEL"
echo "  dataset       : $DATASET"
echo "  output        : $OUTPUT"
echo "  optimizer     : $OPTIMIZER"
echo "  batch         : $BATCH_SIZE × grad_accum $GRAD_ACCUM"
echo "                  → effective: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  max_seq_len   : $MAX_SEQ_LEN"
echo "  epochs        : $EPOCHS  lr: $LR"
echo "==============================================================="

LOG="/root/logs/full_ft_single_$(date +%Y%m%d_%H%M%S).log"

python3 -u scripts/train_full_ft_single.py \
    --base-model "$BASE_MODEL" \
    --dataset "$DATASET" \
    $VAL_ARG \
    --output "$OUTPUT" \
    --max-seq-len $MAX_SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --save-every $SAVE_EVERY \
    --eval-every $EVAL_EVERY \
    2>&1 | tee "$LOG"
