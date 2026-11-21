#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_REFINEMENT="${2:-$NEUR_SPL_ROOT/runs/curvature_refine_qwen_layer12_refine.json}"
NEUR_SPL_DATA="${3:-$NEUR_SPL_ROOT/subsets/dev_v1/curvature_64.jsonl}"
NEUR_SPL_EPSILONS="${4:-0,0.005,0.01,0.02,0.04,0.08,0.12}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/split_line_search_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/split_line_search_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting Qwen split-direction epsilon line search"
echo "Model:  $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Refine: $NEUR_SPL_REFINEMENT"
echo "Data:   $NEUR_SPL_DATA"
echo "Eps:    $NEUR_SPL_EPSILONS"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.split_line_search \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --refinement "$NEUR_SPL_REFINEMENT" \
  --output "$NEUR_SPL_RESULT" \
  --top-k 8 \
  --epsilons "$NEUR_SPL_EPSILONS" \
  --batch-size 2 \
  --max-length 384 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Qwen split-direction epsilon line search finished"
echo "Result: $NEUR_SPL_RESULT"
