#!/bin/bash
set -e

# ─── Configuration ──────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
GPU_UTIL=0.85
MAX_MODEL_LEN=2048

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── Find LoRA checkpoint paths ─────────────────────────────────
# LLaMAFactory outputs to outputs/{task}_lora/checkpoint-* directories
# Auto-discover the latest checkpoint for each adapter

JUDGE_PATH=$(ls -td "${BASE_DIR}/outputs/judge_lora/checkpoint-*" 2>/dev/null | head -1)
POET_PATH=$(ls -td "${BASE_DIR}/outputs/poet_lora/checkpoint-*" 2>/dev/null | head -1)

# Fallback: check if adapter was saved directly (no checkpoint subdirectory)
if [ -z "$JUDGE_PATH" ]; then
    if [ -f "${BASE_DIR}/outputs/judge_lora/adapter_config.json" ]; then
        JUDGE_PATH="${BASE_DIR}/outputs/judge_lora"
    fi
fi
if [ -z "$POET_PATH" ]; then
    if [ -f "${BASE_DIR}/outputs/poet_lora/adapter_config.json" ]; then
        POET_PATH="${BASE_DIR}/outputs/poet_lora"
    fi
fi

if [ -z "$JUDGE_PATH" ]; then
    echo "ERROR: No judge LoRA checkpoint found in outputs/judge_lora/"
    echo "Please run: python scripts/train_lora.py --task judge"
    exit 1
fi

if [ -z "$POET_PATH" ]; then
    echo "ERROR: No poet LoRA checkpoint found in outputs/poet_lora/"
    echo "Please run: python scripts/train_lora.py --task poet"
    exit 1
fi

echo "=== Starting vLLM Server with Multi-LoRA ==="
echo "Base model: $MODEL"
echo "Judge LoRA: $JUDGE_PATH"
echo "Poet LoRA:  $POET_PATH"
echo "Port: $PORT"
echo ""

# ─── Launch vLLM ───────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 32 \
    --max-cpu-loras 4 \
    --dtype auto \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --lora-modules "judge=$JUDGE_PATH" "poet=$POET_PATH"

# Note: vllm serve blocks until the server is stopped.
# To use in background: run this script with & or use nohup.
