#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_SCREENING="${2:?Pass the Qwen screening JSON as argument 2}"
NEUR_SPL_DATA="${3:-$NEUR_SPL_ROOT/subsets/dev_v1/curvature_64.jsonl}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/curvature_refine_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/curvature_refine_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting Qwen Lanczos curvature refinement"
echo "Model:  $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Data:   $NEUR_SPL_DATA"
echo "Screen: $NEUR_SPL_SCREENING"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.curvature_refine \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --screening "$NEUR_SPL_SCREENING" \
  --output "$NEUR_SPL_RESULT" \
  --candidate-count 64 \
  --iterations 20 \
  --batch-size 1 \
  --max-length 384 \
  --seed 314159 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Qwen Lanczos curvature refinement finished"
echo "Result: $NEUR_SPL_RESULT"
