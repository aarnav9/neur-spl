#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_EPSILON="${2:-0.08}"
NEUR_SPL_CONTROL_SEEDS="${3:-10}"
NEUR_SPL_REFINEMENT="${4:-$NEUR_SPL_ROOT/runs/curvature_refine_qwen_layer12_refine.json}"
NEUR_SPL_DATA="${5:-$NEUR_SPL_ROOT/subsets/dev_v1/validation_128.jsonl}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/split_controls_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/split_controls_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting locked Qwen curvature-split controls"
echo "Model:   $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Refine:  $NEUR_SPL_REFINEMENT"
echo "Data:    $NEUR_SPL_DATA"
echo "Epsilon: $NEUR_SPL_EPSILON"
echo "Seeds:   $NEUR_SPL_CONTROL_SEEDS"
echo "Arms:    $((2 + 3 * NEUR_SPL_CONTROL_SEEDS))"
echo "Result:  $NEUR_SPL_RESULT"
echo "Log:     $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.split_controls \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --refinement "$NEUR_SPL_REFINEMENT" \
  --output "$NEUR_SPL_RESULT" \
  --top-k 8 \
  --epsilon "$NEUR_SPL_EPSILON" \
  --control-seeds "$NEUR_SPL_CONTROL_SEEDS" \
  --batch-size 2 \
  --max-length 384 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Locked Qwen curvature-split controls finished"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"
