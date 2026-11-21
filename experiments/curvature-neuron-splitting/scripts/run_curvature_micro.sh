#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
NEUR_SPL_RESULT="$NEUR_SPL_ROOT/runs/curvature_proxy_${NEUR_SPL_RUN_TAG}.json"
NEUR_SPL_LOG="$NEUR_SPL_ROOT/runs/curvature_proxy_${NEUR_SPL_RUN_TAG}.log"

mkdir -p "$NEUR_SPL_ROOT/runs"
export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Starting curvature scan"
echo "Result: $NEUR_SPL_RESULT"
echo "Log:    $NEUR_SPL_LOG"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.curvature_proxy \
  --model "$NEUR_SPL_ROOT/models/SmolLM2-135M-Instruct" \
  --data "$NEUR_SPL_ROOT/subsets/micro_v1/curvature_16.jsonl" \
  --output "$NEUR_SPL_RESULT" \
  --layer 15 \
  --batch-size 2 \
  --max-length 384 \
  --probes 8 \
  --top-k 32 \
  --seed 42 \
  2>&1 | tee "$NEUR_SPL_LOG"

echo "Curvature scan finished"
echo "Result: $NEUR_SPL_RESULT"

