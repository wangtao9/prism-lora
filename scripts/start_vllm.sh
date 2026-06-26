#!/bin/bash
set -e

# ─── Configuration (from configs/config.yaml) ──────────────────
# Read values via Python one-liner for portable YAML parsing.
# Environment variables override YAML values.
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse model path: env var > yaml > fallback
if [ -n "$PRISM_BASE_MODEL" ]; then
    MODEL="$PRISM_BASE_MODEL"
else
    MODEL="$(cd "$BASE_DIR" && python3 -c "
import yaml, os
with open('configs/config.yaml') as f: cfg=yaml.safe_load(f)
print(cfg['base_model'])
")"
fi

# Parse vllm port
if [ -n "$PRISM_VLLM_PORT" ]; then
    VLLM_PORT="$PRISM_VLLM_PORT"
else
    VLLM_PORT="$(cd "$BASE_DIR" && python3 -c "
import yaml
with open('configs/config.yaml') as f: cfg=yaml.safe_load(f)
print(cfg['vllm_port'])
")"
fi

GPU_UTIL="$(cd "$BASE_DIR" && python3 -c "
import yaml
with open('configs/config.yaml') as f: cfg=yaml.safe_load(f)
print(cfg['vllm_gpu_util'])
")"

MAX_MODEL_LEN="$(cd "$BASE_DIR" && python3 -c "
import yaml
with open('configs/config.yaml') as f: cfg=yaml.safe_load(f)
print(cfg['vllm_max_model_len'])
")"

# ─── Find LoRA checkpoint paths ─────────────────────────────────
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
echo "Port: $VLLM_PORT"
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
    --port "$VLLM_PORT" \
    --lora-modules "judge=$JUDGE_PATH" "poet=$POET_PATH"

# Note: vllm serve blocks until the server is stopped.
# To use in background: run this script with & or use nohup.