#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_REFINEMENT="${2:-$NEUR_SPL_ROOT/runs/curvature_refine_first_refine.json}"
NEUR_SPL_DATA="${3:-$NEUR_SPL_ROOT/subsets/micro_v1/curvature_16.jsonl}"
NEUR_SPL_EPSILONS="${4:-0,0.025,0.05,0.1,0.2}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/split_line_search_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/split_line_search_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting split-direction epsilon line search"
echo "Refine: $NEUR_SPL_REFINEMENT"
echo "Data:   $NEUR_SPL_DATA"
echo "Eps:    $NEUR_SPL_EPSILONS"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.split_line_search \
  --model "$NEUR_SPL_ROOT/models/SmolLM2-135M-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --refinement "$NEUR_SPL_REFINEMENT" \
  --output "$NEUR_SPL_RESULT" \
  --top-k 8 \
  --epsilons "$NEUR_SPL_EPSILONS" \
  --batch-size 2 \
  --max-length 384 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Split-direction epsilon line search finished"
echo "Result: $NEUR_SPL_RESULT"
