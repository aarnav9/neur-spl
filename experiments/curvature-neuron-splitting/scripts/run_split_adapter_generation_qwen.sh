#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:?usage: $0 RUN_TAG ADAPTER [BASELINE] [DATA]}"
NEUR_SPL_ADAPTER="${2:?usage: $0 RUN_TAG ADAPTER [BASELINE] [DATA]}"
NEUR_SPL_BASELINE="${3:-$NEUR_SPL_ROOT/runs/qwen_baseline_qwen_validation128.json}"
NEUR_SPL_DATA="${4:-$NEUR_SPL_ROOT/subsets/dev_v1/validation_128.jsonl}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/split_adapter_generation_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/split_adapter_generation_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting trained-adapter Qwen generation evaluation"
echo "Estimated runtime on this Mac: roughly 5-20 minutes for 128 examples, depending on output length"
echo "Model:    $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Data:     $NEUR_SPL_DATA"
echo "Adapter:  $NEUR_SPL_ADAPTER"
echo "Baseline: $NEUR_SPL_BASELINE"
echo "Result:   $NEUR_SPL_RESULT"
echo "Log:      $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.evaluate_split_generation \
  --benchmark gsm8k \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --adapter "$NEUR_SPL_ADAPTER" \
  --baseline-result "$NEUR_SPL_BASELINE" \
  --output "$NEUR_SPL_RESULT" \
  --batch-size 8 \
  --max-input-tokens 512 \
  --max-new-tokens 128 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Trained-adapter Qwen generation evaluation finished"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"
