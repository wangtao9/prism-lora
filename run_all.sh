#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================";
echo "  Prism-LoRA: LoRA + vLLM Multi-Adapter";
echo "=========================================";
echo "";

# Step 1: 合成训练数据
echo "=== Step 1: Generating training data ==="
python scripts/prepare_data.py
echo ""

# Step 2: 训练 Judge LoRA
echo "=== Step 2: Training Judge LoRA ==="
python scripts/train_lora.py --task judge
echo ""

# Step 3: 训练 Poet LoRA
echo "=== Step 3: Training Poet LoRA ==="
python scripts/train_lora.py --task poet
echo ""

# Step 4: 启动 vLLM 多适配器服务（后台运行）
echo "=== Step 4: Starting vLLM multi-adapter server ==="
echo "Starting vLLM in background..."
bash scripts/start_vllm.sh &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM ready (max 300s, check every 5s with curl)
echo "Waiting for vLLM to be ready..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "Error: vLLM server did not start within $MAX_WAIT seconds"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi
echo ""

# Step 5: 运行评测
echo "=== Step 5: Running evaluation ==="
python scripts/evaluate.py
echo ""

# Step 6: 生成对比报告
echo "=== Step 6: Generating comparison report ==="
python scripts/evaluate.py --report
echo ""

# cleanup: stop vLLM
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
echo ""

echo "========================================="
echo "  All steps completed!"
echo "  Report: results/comparison.md"
echo "========================================="
