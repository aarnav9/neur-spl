#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_LIMIT="${2:-128}"
NEUR_SPL_DATA="${3:-$NEUR_SPL_ROOT/subsets/dev_v1/validation_128.jsonl}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/qwen_baseline_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/qwen_baseline_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting Qwen validation baseline"
echo "Model:  $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Data:   $NEUR_SPL_DATA"
echo "Limit:  $NEUR_SPL_LIMIT"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.evaluate \
  --benchmark gsm8k \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --output "$NEUR_SPL_RESULT" \
  --limit "$NEUR_SPL_LIMIT" \
  --seed 42 \
  --batch-size 8 \
  --max-input-tokens 512 \
  --max-new-tokens 128 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Qwen validation baseline finished"
echo "Result: $NEUR_SPL_RESULT"
