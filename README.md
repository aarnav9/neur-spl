# Cut-to-Grow

Research prototype for treating a sparse neural network as a task-conditioned
transport graph. The idea is to protect bottleneck edges, remove redundant
capacity, and grow new connections across weak cuts while keeping the parameter
budget fixed.

Current status: the graph solver, CIFAR ResNet-18, deterministic dense trainer,
transport monitor, branch-local min-cut scorer, channel-deletion probe, and
score-correlation report are implemented and tested. The first declared
CIFAR-100 seed has reached the epoch-20 checkpoint. Its exhaustive structured
diagnostic is evidence for a useful flow signal over magnitude, but gate-level
Group Fisher is a stronger and cheaper `conv1` catastrophic-deletion score. A
subsequent frozen `conv2` holdout found no significant incremental benefit from
an equal-rank Fisher-flow hybrid, so dynamic Cut-to-Grow training is currently
blocked rather than assumed to work. Measured-skip, local coarse-class, and
global class-sink flow redesigns were also implemented; discovery checks show
that scalar max-flow treats residual shortcuts as semantic substitutes and does
not justify fresh deletion labels. A true 20-commodity shared-capacity LP was
also tested and assigned zero marginal value to every learned `conv1` group,
confirming that the blocker is the scalar representation rather than the flow
solver. A task-conditioned PSD replacement based on per-example mask-gradient
outer products is now implemented. Its ridge-leverage score did not improve
single-group catastrophic-deletion ranking over Group Fisher on the existing
epoch-20 tables; the remaining defensible question is joint spectral subset
preservation, not another fitted scalar hybrid. That joint test has now also
finished: unweighted leverage failed to beat Fisher consistently, while
rescaled leverage sampling won strongly at 50% group sparsity but regressed at
25% and collapsed at 75%. It failed the frozen multi-sparsity gate, so this
branch is stopped rather than tuned around the favorable middle point.

## Layout

```text
research_blueprint.md   Question, method, experiments, gates, and prior work
implementation.md       What the current code does and what to build next
code/                   Python implementation, pilot config, scripts, and tests
pull/                   Data and external-code manifest; payloads appear on demand
```

## Quick check

```bash
make test
make toy
make validate-config
make list-data
```

The toy example removes a redundant edge, grows an edge across the weak cut,
keeps the same number of active edges, and raises max-flow from `5` to `7`.

For a clean PyTorch environment:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e 'code[hf]'
make test PYTHON=.venv/bin/python
make pull-pilot PYTHON=.venv/bin/python
```

The dataset is the standard CIFAR-100 benchmark. The canonical source is the
University of Toronto; the pinned `uoft-cs/cifar100` Hugging Face mirror is used
for faster retrieval. Development uses a stratified 45,000/5,000 split and the
official test loader is intentionally absent.

Train a declared seed, probe a checkpoint, and summarize it with:

```bash
.venv/bin/python code/scripts/train_pilot.py
.venv/bin/python code/scripts/probe_checkpoint.py runs/dense_cifar100_seed_20260824/checkpoint_epoch_020.pt
.venv/bin/python code/scripts/analyze_probes.py runs/dense_cifar100_seed_20260824/checkpoint_epoch_020_probes.csv
```

Run a matched structured-group diagnostic with `--deletion-group-size`. Multiple
`--reuse-probes-from` tables can be supplied to consolidate recoverable shards
without repeating deletion measurements.

```bash
.venv/bin/python code/scripts/probe_checkpoint.py \
  runs/dense_cifar100_seed_20260824/checkpoint_epoch_020.pt \
  --device mps --deletion-group-size 16
```

Taylor and gate-level Group Fisher can be added to an existing deletion table
without repeating any interventions:

```bash
.venv/bin/python code/scripts/rescore_probes.py \
  runs/dense_cifar100_seed_20260824/checkpoint_epoch_020.pt \
  runs/dense_cifar100_seed_20260824/checkpoint_epoch_020_group16_conv1_exhaustive.csv \
  --device mps --calibration-batches 8
```

Matrix-Fisher traces and global ridge-leverage scores can likewise reuse a
cached deletion table:

```bash
.venv/bin/python code/scripts/rescore_spectral_fisher.py \
  runs/dense_cifar100_seed_20260824/checkpoint_epoch_020.pt \
  runs/dense_cifar100_seed_20260824/checkpoint_epoch_020_group16_conv1_exhaustive_strong_baselines.csv \
  --device cpu --num-workers 0 --calibration-batches 8
```

The frozen exact-budget joint-mask protocol is reproducible with:

```bash
.venv/bin/python code/scripts/evaluate_joint_spectral_masks.py \
  runs/dense_cifar100_seed_20260824/checkpoint_epoch_020.pt \
  --data-root pull/raw --device mps --num-workers 0 \
  --score-batches 8 --bn-batches 8 \
  --sparsity 0.25 --sparsity 0.50 --sparsity 0.75 \
  --spectral-replicates 3
```

The probe table now contains magnitude, local task transport, coarsened-group
marginal-flow damage, transport-allocated marginal flow, and cut-incidence
scores alongside measured deletion damage. Reports include layer-macro
Spearman, top-5%-damage AUROC and average precision, an absolute two-point
accuracy-damage endpoint, paired bootstrap intervals, and a within-layer
shuffle control. The flow graph currently covers each two-convolution residual
branch. It deliberately excludes the skip path until that path has a measured
capacity rather than an arbitrary constant.
