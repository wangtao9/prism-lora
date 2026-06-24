#!/bin/bash
# Stop the vLLM multi-adapter server

PIDS=$(ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No vLLM server found."
    exit 0
fi

for PID in $PIDS; do
    echo "Killing vLLM server (PID $PID)..."
    kill "$PID" 2>/dev/null
done

# Wait briefly for graceful shutdown
sleep 2

# Force kill if still running
for PID in $PIDS; do
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing PID $PID..."
        kill -9 "$PID" 2>/dev/null
    fi
done

echo "vLLM server stopped."