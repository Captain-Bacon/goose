#!/bin/bash
# Start vllm-mlx server for Goose
# Usage: ./start-mlx-server.sh [model]
#   model: 35b (default), 27b, 35b-8bit
#
# Kills any existing vllm-mlx process before starting.

set -euo pipefail

VENV="$(dirname "$0")/vllm-mlx-env"
PORT=5757
MODEL_BASE="$HOME/.cache/lm-studio/models/mlx-community"

# Model selection
case "${1:-35b}" in
  35b)
    MODEL="$MODEL_BASE/Qwen3.5-35B-A3B-4bit"
    echo "Starting Qwen 3.5 35B-A3B 4-bit (fast, MoE 3B active)"
    ;;
  27b)
    MODEL="$MODEL_BASE/Qwen3.5-27B-4bit"
    echo "Starting Qwen 3.5 27B 4-bit (dense, slower but smarter)"
    ;;
  35b-8bit)
    MODEL="$MODEL_BASE/Qwen3.5-35B-A3B-8bit"
    echo "Starting Qwen 3.5 35B-A3B 8-bit (MoE, higher quality)"
    ;;
  *)
    echo "Unknown model: $1"
    echo "Options: 35b (default), 27b, 35b-8bit"
    exit 1
    ;;
esac

# Kill existing vllm-mlx process if running
if pgrep -f "vllm-mlx serve" > /dev/null 2>&1; then
  echo "Stopping existing vllm-mlx server..."
  pkill -f "vllm-mlx serve"
  sleep 1
fi

echo "Serving on port $PORT"
"$VENV/bin/vllm-mlx" serve "$MODEL" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --port "$PORT"
