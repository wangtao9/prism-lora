#!/bin/bash
set -e

# Resolve BASE_DIR from script location
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Prism-LoRA vLLM Multi-Adapter Server ==="
echo "BASE_DIR: ${BASE_DIR}"

# Check adapter directories exist
JUDGE_ADAPTER="${BASE_DIR}/adapters/judge/adapter_config.json"
POET_ADAPTER="${BASE_DIR}/adapters/poet/adapter_config.json"

if [ ! -f "${JUDGE_ADAPTER}" ]; then
    echo "ERROR: Judge adapter not found at ${JUDGE_ADAPTER}"
    echo "Please run training first: python scripts/train_lora.py configs/judge_lora.yaml"
    exit 1
fi

if [ ! -f "${POET_ADAPTER}" ]; then
    echo "ERROR: Poet adapter not found at ${POET_ADAPTER}"
    echo "Please run training first: python scripts/train_lora.py configs/poet_lora.yaml"
    exit 1
fi

echo "Adapter check passed:"
echo "  - judge: ${JUDGE_ADAPTER}"
echo "  - poet: ${POET_ADAPTER}"

echo "Starting vLLM server on http://0.0.0.0:8000 ..."
echo ""

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-lora \
    --lora-modules judge="${BASE_DIR}/adapters/judge" poet="${BASE_DIR}/adapters/poet" \
    --max-lora-rank 8 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --host 0.0.0.0
