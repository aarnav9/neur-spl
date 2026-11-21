#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_EPSILON="${2:-0.08}"
NEUR_SPL_DATA="${3:-$NEUR_SPL_ROOT/subsets/general_retention_v1/eval_192.jsonl}"
NEUR_SPL_REFINEMENT="${4:-$NEUR_SPL_ROOT/runs/curvature_refine_qwen_layer12_refine.json}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/general_retention_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/general_retention_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting paired Qwen general-retention evaluation"
echo "Model:   $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Data:    $NEUR_SPL_DATA"
echo "Refine:  $NEUR_SPL_REFINEMENT"
echo "Epsilon: $NEUR_SPL_EPSILON"
echo "Result:  $NEUR_SPL_RESULT"
echo "Log:     $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.evaluate_general_retention \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --refinement "$NEUR_SPL_REFINEMENT" \
  --output "$NEUR_SPL_RESULT" \
  --top-k 8 \
  --epsilon "$NEUR_SPL_EPSILON" \
  --batch-size 2 \
  --max-length 512 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Paired Qwen general-retention evaluation finished"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"
