#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_EPSILON="${2:-0.08}"
NEUR_SPL_OBJECTIVE="${3:-mixed}"
NEUR_SPL_ANSWER_FRACTION="${4:-0.5}"
NEUR_SPL_DATA="${5:-$NEUR_SPL_ROOT/subsets/dev_v1/train_512.jsonl}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/localized_train_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_ADAPTER="$NEUR_SPL_ROOT/runs/localized_train_${NEUR_SPL_RUN_TAG}.adapter.pt"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/localized_train_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting localized Qwen split training"
echo "Estimated runtime on this Mac: roughly 60-80 minutes for 512 examples"
echo "Model:      $NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct"
echo "Data:       $NEUR_SPL_DATA"
echo "Refine:     $NEUR_SPL_ROOT/runs/curvature_refine_qwen_layer12_refine.json"
echo "Epsilon:    $NEUR_SPL_EPSILON"
echo "Objective:  $NEUR_SPL_OBJECTIVE"
echo "Answer mix: $NEUR_SPL_ANSWER_FRACTION"
echo "Result:     $NEUR_SPL_RESULT"
echo "Adapter:    $NEUR_SPL_ADAPTER"
echo "Log:        $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.train_localized_split \
  --model "$NEUR_SPL_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --data "$NEUR_SPL_DATA" \
  --refinement "$NEUR_SPL_ROOT/runs/curvature_refine_qwen_layer12_refine.json" \
  --output "$NEUR_SPL_RESULT" \
  --adapter-output "$NEUR_SPL_ADAPTER" \
  --top-k 8 \
  --epsilon "$NEUR_SPL_EPSILON" \
  --objective "$NEUR_SPL_OBJECTIVE" \
  --answer-fraction "$NEUR_SPL_ANSWER_FRACTION" \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 0.0003 \
  --weight-decay 0 \
  --max-grad-norm 1 \
  --max-length 384 \
  --checkpoint-every-steps 10 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Localized Qwen split training finished"
echo "Result:  $NEUR_SPL_RESULT"
echo "Adapter: $NEUR_SPL_ADAPTER"
echo "Log:     $NEUR_SPL_LOG"
