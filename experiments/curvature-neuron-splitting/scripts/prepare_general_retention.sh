#!/usr/bin/env bash
set -euo pipefail

NEUR_SPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUR_SPL_PER_TASK="${1:-64}"
NEUR_SPL_OUTPUT="$NEUR_SPL_ROOT/subsets/general_retention_v1/eval_$((NEUR_SPL_PER_TASK * 3)).jsonl"

export PYTHONPATH="$NEUR_SPL_ROOT/src"
export HF_HOME="$NEUR_SPL_ROOT/.hf-cache"
export PYTHONUNBUFFERED=1

echo "Preparing general-retention datasets"
echo "Tasks:    HellaSwag, ARC-Challenge, BoolQ"
echo "Per task: $NEUR_SPL_PER_TASK"
echo "Output:   $NEUR_SPL_OUTPUT"

"$NEUR_SPL_ROOT/.venv/bin/python" -u -m neurspl.prepare_general_retention \
  --raw-dir "$NEUR_SPL_ROOT/datasets/general_retention" \
  --output "$NEUR_SPL_OUTPUT" \
  --examples-per-task "$NEUR_SPL_PER_TASK" \
  --seed 42

echo "General-retention datasets ready"
echo "Output: $NEUR_SPL_OUTPUT"
echo "Manifest: ${NEUR_SPL_OUTPUT%.jsonl}.manifest.json"
