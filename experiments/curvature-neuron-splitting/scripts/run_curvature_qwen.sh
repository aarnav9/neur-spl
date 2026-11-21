#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_DATA="${2:-$NEUR_SPL_ROOT/subsets/dev_v1/curvature_64.jsonl}"
NEUR_SPL_LAYER="${3:-12}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/curvature_proxy_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/curvature_proxy_${NEUR_SPL_RUN_TAG}.log"
NEUR_SPL_CHECKPOINT="$NEUR_SPL_ROOT/runs/curvature_proxy_${NEUR_SPL_RUN_TAG}.checkpoint.pt"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting Qwen curvature scan"
echo "Model:  $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Data:   $NEUR_SPL_DATA"
echo "Layer:  $NEUR_SPL_LAYER"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"
echo "Resume: $NEUR_SPL_CHECKPOINT"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.curvature_proxy \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --output "$NEUR_SPL_RESULT" \
  --checkpoint "$NEUR_SPL_CHECKPOINT" \
  --resume \
  --layer "$NEUR_SPL_LAYER" \
  --batch-size 1 \
  --max-length 384 \
  --probes 8 \
  --top-k 64 \
  --seed 42 \
  2>&1 | tee -a "$NEUR_SPL_LOG"

echo "Qwen curvature scan finished"
echo "Result: $NEUR_SPL_RESULT"
