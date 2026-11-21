#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_REFINEMENT="${2:-$NEUR_SPL_ROOT/runs/curvature_refine_qwen_layer12_refine.json}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/split_sanity_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/split_sanity_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting exact Qwen split sanity check"
echo "Model:  $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Refine: $NEUR_SPL_REFINEMENT"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.split_sanity \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --refinement "$NEUR_SPL_REFINEMENT" \
  --top-k 8 \
  --output "$NEUR_SPL_RESULT" \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Exact Qwen split sanity check finished"
echo "Result: $NEUR_SPL_RESULT"
