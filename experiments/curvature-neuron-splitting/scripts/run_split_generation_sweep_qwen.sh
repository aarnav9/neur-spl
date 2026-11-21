#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_EPSILONS="${2:-0.005,0.01,0.02,0.04,0.08}"
NEUR_SPL_OFFSET="${3:-0}"
NEUR_SPL_LIMIT="${4:-32}"
NEUR_SPL_BASELINE="${5:-$NEUR_SPL_ROOT/runs/qwen_baseline_qwen_validation128.json}"
NEUR_SPL_REFINEMENT="${6:-$NEUR_SPL_ROOT/runs/curvature_refine_qwen_layer12_refine.json}"
NEUR_SPL_DATA="${7:-$NEUR_SPL_ROOT/subsets/dev_v1/validation_128.jsonl}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/split_generation_sweep_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/split_generation_sweep_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting split-only Qwen generation epsilon sweep"
echo "Model:    $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Data:     $NEUR_SPL_DATA"
echo "Refine:   $NEUR_SPL_REFINEMENT"
echo "Baseline: $NEUR_SPL_BASELINE"
echo "Eps:      $NEUR_SPL_EPSILONS"
echo "Slice:    offset=$NEUR_SPL_OFFSET limit=$NEUR_SPL_LIMIT"
echo "Result:   $NEUR_SPL_RESULT"
echo "Log:      $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.split_generation_sweep \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --refinement "$NEUR_SPL_REFINEMENT" \
  --baseline-result "$NEUR_SPL_BASELINE" \
  --output "$NEUR_SPL_RESULT" \
  --benchmark gsm8k \
  --top-k 8 \
  --epsilons "$NEUR_SPL_EPSILONS" \
  --offset "$NEUR_SPL_OFFSET" \
  --limit "$NEUR_SPL_LIMIT" \
  --batch-size 8 \
  --max-input-tokens 512 \
  --max-new-tokens 128 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Split-only Qwen generation epsilon sweep finished"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"
