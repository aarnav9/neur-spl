# Curvature-guided neuron splitting

Experiments with function-preserving SwiGLU neuron duplication, negative-curvature
selection, and localized recovery training on SmolLM2-135M and Qwen2.5-0.5B.
This project lives in `experiments/curvature-neuron-splitting/` in the shared
`aarnav9/neur-spl` repository, alongside the separate Cut-to-Grow project.

The completed experiments show lower teacher-forced math loss, but **no
statistically credible GSM8K accuracy improvement**. Large separation steps and
the first mixed-objective localized-training method degraded generation accuracy.
See [the cumulative results ledger](results.md) for the evidence, negative
results, and development/holdout distinctions.

## Contents

- `src/neurspl/`: splitting, curvature estimation, evaluation, and localized training.
- `scripts/`: foreground experiment commands with saved results and logs.
- `tests/`: unit tests using small tensors and fixtures.
- `subsets/`: fixed development examples and selection manifests.
- `runs/`: saved JSON results and text logs, including refined split directions.
- [NOTES.md](NOTES.md): research notes and experimental decisions.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): the initial experiment plan.
- [SOURCES.json](SOURCES.json): recorded source revisions for the initial assets.
- [SIZE_SUMMARY.json](SIZE_SUMMARY.json): sizes of the initial downloaded assets,
  not the size of this Git folder.

## Setup and quick validation

From the shared repository root:

```bash
cd experiments/curvature-neuron-splitting
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

If working from a standalone copy of this folder, start at the `python3.11`
command. The unit tests do not require model weights or full datasets.

## Local experiment assets

The original local workspace contains these downloaded assets, which are
excluded from Git:

- `models/Qwen2.5-0.5B-Instruct/` from `Qwen/Qwen2.5-0.5B-Instruct`.
- `models/SmolLM2-135M-Instruct/` from `HuggingFaceTB/SmolLM2-135M-Instruct`.
- Complete GSM8K and Hendrycks MATH repositories under `datasets/`.
- The first 50,000 valid records from OpenMathInstruct-1's corrected-solutions
  training file at `datasets/openmathinstruct_50k/train.jsonl`.
- General-retention source data downloaded by `scripts/prepare_general_retention.sh`.

Restore the assets required by a script at these relative paths before running
it; use the pinned revisions in `SOURCES.json` where recorded. The run scripts
locate this folder relative to themselves, use its `.venv`, and load models
offline. Original benchmark test splits were retained separately from the
OpenMath training sample. Consult `results.md` before using any subset as a
holdout: several named test/validation subsets have already been inspected.

Run longer experiments in a foreground Terminal using the scripts in `scripts/`.
They write into `runs/`; choose a fresh run tag to preserve archived results.
Model weights, binary checkpoints/adapters, full datasets, virtual environments,
caches, and temporary files are excluded. Archived JSON and text logs retain
the original local paths as provenance; those paths are not portable commands.
