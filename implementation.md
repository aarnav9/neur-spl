# Implementation Notes

The repository currently contains a small research oracle, not a full training
system.

## Code layout

```text
code/src/cut_to_grow/capacity.py        Framework-free capacity definitions
code/src/cut_to_grow/flow.py            Exact directed max-flow/min-cut solver
code/src/cut_to_grow/policy.py          Fixed-budget remove-and-grow oracle
code/src/cut_to_grow/torch_capacity.py  PyTorch Linear/Conv2d monitor
code/src/cut_to_grow/model.py           CIFAR ResNet-18 and channel groups
code/src/cut_to_grow/pilot.py           Data split, loaders, training, checkpoints
code/src/cut_to_grow/diagnostics.py     Channel ablation and correlation metrics
code/src/cut_to_grow/block_flow.py      Coarsened residual-branch flow scores
code/src/cut_to_grow/global_flow.py     Global coarse-class commodity graph
code/src/cut_to_grow/multicommodity.py  Shared-capacity concurrent-flow LP
code/scripts/run_toy.py                 Deterministic toy demonstration
code/scripts/pull_data.py               CIFAR downloader and manifest listing
code/scripts/train_pilot.py              Dense CIFAR training entrypoint
code/scripts/probe_checkpoint.py         Score-versus-deletion-damage table
code/scripts/rescore_probes.py           Taylor/Fisher scores for cached probes
code/scripts/analyze_probes.py           Pooled and layer-macro correlations
code/scripts/test_conditional_flow.py    Locked conv2 holdout decision
code/scripts/rescore_skip_aware.py       Measured residual-bypass flow
code/scripts/rescore_class_flow.py       Local coarse-class flow ensemble
code/scripts/rescore_global_class_flow.py Global class-sink flow graph
code/scripts/rescore_concurrent_flow.py  Concurrent-flow dual deletion score
code/scripts/validate_config.py         Pilot-config checks
code/configs/pilot.yaml                 Declared development settings and gates
code/tests/                              Unit tests for the implemented pieces
pull/manifest.yaml                       Dataset and upstream-baseline inventory
```

The nested Python layout is retained because these files already contain real,
tested implementation. Empty result, report, artifact, reference, and scratch
directories are intentionally not created in advance.

## Implemented behavior

`flow.py` uses Dinic's algorithm to return maximum flow, one corresponding
minimum cut, the source-reachable residual set, and per-edge flows.

`policy.py` is deliberately exact and slow. It protects the current cut,
recomputes flow after each eligible deletion, removes the least damaging edge,
then evaluates inactive edges across the post-removal cut. It preserves the
number of active edges exactly. This is an oracle for testing the idea before
building approximations.

`torch_capacity.py` records
`E[|activation * weight * downstream-gradient|]` using forward and backward
hooks. It supports `Linear` and ordinary non-grouped `Conv2d` layers and keeps
EMA scores on CPU. The probe runner now connects those scores to explicit
ResNet convolution/BatchNorm channel groups. A second monitor scores the exact
contiguous activation masks used by the structured deletion intervention. For
per-example mask gradient

```text
g(i,G) = sum[c in G,h,w] activation(i,c,h,w) * d loss(i) / d activation(i,c,h,w)
```

it reports first-order Taylor as `|mean_i g(i,G)|` and gate-level Group Fisher
as `mean_i g(i,G)^2`. The constant factor sometimes used in Fisher definitions
does not affect rankings. This is a matched activation-gate comparator, not a
claim to implement every dependency grouping in the full Group Fisher pruning
algorithm.

`pilot.py` creates the fixed stratified 45,000/5,000 CIFAR-100 development
split, trains the CIFAR ResNet-18, and saves only the declared early/middle/late
checkpoints. It never constructs an official-test loader. The Hugging Face
mirror is pinned and saved into a local offline dataset after its first pull.

`diagnostics.py` selects channels across both magnitude and transport rankings,
temporarily zeroes each channel at its BatchNorm gate, and records immediate
selection-loss damage. `analyze_probes.py` reports tie-aware pooled and
equal-layer-weighted Spearman correlations, tail AUROC and average precision,
paired stratified-bootstrap score differences, and a within-layer shuffle null.

`block_flow.py` converts the per-kernel transport tensor for both convolutions
in a BasicBlock into a three-level input→hidden→output graph. Channels are
coarsened in fixed groups of 16, exact maximum flow is computed, and each hidden
or output group is removed in turn to measure its exact normalized flow damage.
That group value is reported directly and allocated back to individual channels
in proportion to local transport. Cut-incidence scores are reported the same
way. This exposes whether the group-level topology contributes information
beyond the local channel score.

## Commands

```bash
make test             # unit tests
make toy              # expected flow: 5 -> 7 at a fixed edge budget
make validate-config  # validate the declared pilot
make list-data        # show data and external inputs without downloading
```

To install and pull the development data:

```bash
.venv/bin/pip install -e 'code[hf]'
make pull-pilot PYTHON=.venv/bin/python
```

The downloader creates `pull/raw/` only when it is actually used. ImageNet is
listed in the manifest but requires a licensed manual copy. External baseline
repositories are discovery entries and must be pinned to commits before a
formal comparison.

## Current empirical state and next implementation

The first declared seed reached epoch 20 at 54.62% selection accuracy. The
initial per-channel diagnostic rejected flow as a general within-layer saliency
score. A matched, exhaustive size-16 diagnostic then measured every one of the
120 hidden `conv1` groups. Six groups caused at least a two-point accuracy drop.
The cached intervention table has now been rescored with strong gradient
baselines using 1,024 calibration examples.

| Score | 2-point-drop AUROC | Average precision | Rank needed to cover all 6 | Within-layer-rank AUROC |
|---|---:|---:|---:|---:|
| Group Fisher | 0.996 | 0.944 | 9 | 0.795 |
| Flow marginal / cut incidence | 0.959 | 0.674 | 17 | 0.731 |
| Taylor | 0.940 | 0.702 | 39 | 0.664 |
| Magnitude | 0.934 | 0.382 | 32 | 0.623 |

Group Fisher minus flow was `+0.0365` AUROC with a paired stratified-bootstrap
95% interval of `[+0.0029, +0.0848]`; its average-precision difference was
`+0.2705 [ +0.0278, +0.5462 ]`. The flow-versus-magnitude AUROC difference was
only `+0.0249 [-0.0556, +0.1184]`. Flow retains information beyond layer
allocation—the within-layer shuffle null gave `p=0.0224`—but Group Fisher was
stronger there too (`p=0.0012`). The primary general-ranking metric also still
rejects flow: layer-macro Spearman was 0.021, versus 0.286 Taylor, 0.230
magnitude, and 0.120 Group Fisher.

The cost result points the same way. Taylor and Group Fisher were collected
together in 4.07 seconds. The CPU-stabilized transport plus flow pass took
35.87 + 0.77 = 36.64 seconds. These are single-machine diagnostic timings, not
portable hardware benchmarks, but flow is not earning its extra cost here.

The frozen `conv2` structural holdout has also finished. It contained eight
two-point failures among 120 untouched deletions. The fixed equal-percentile
Fisher-flow hybrid scored 0.992 AUROC versus 0.979 Fisher, but its paired gain
was only `+0.0134 [-0.0089, +0.0446]`. That missed both the zero-exclusion and
`+0.05` gates, so the conditional test failed. Magnitude alone reached 0.989
AUROC and protected all eight failures in its top 11 groups, versus 12 hybrid,
19 Fisher, and 20 flow.

All catastrophic `conv2` deletions occurred in layers 1 and 2. Flow's
within-layer shuffle test was not significant (`p=0.0854`) and its layer-macro
Spearman was 0.038. The apparently excellent pooled AUROC is therefore mostly
a layer-allocation result, precisely the failure mode declared in the research
blueprint.

The current decision is:

1. preserve both exhaustive tables and the frozen failed holdout without
   post-hoc hybrid tuning;
2. stop before the dynamic topology optimizer—the current score does not
   justify its cost;
3. treat skip-aware or class-conditioned multi-commodity flow as a new method,
   not a reinterpretation of this test, and freeze it before using new labels;
4. use a new checkpoint or seed for any redesigned method; and
5. pursue a pruning-criterion benchmark paper only after adding multiple
   checkpoints, architectures, and datasets—the current single-checkpoint
   negative result is useful research evidence but not yet a paper claim.

## Skip-aware and class-conditioned redesign outcome

Two prespecified redesign directions were implemented after the frozen holdout,
but evaluated only on the already-seen epoch-20 deletion tables. No fresh
labels were spent because neither cleared a development sanity gate.

The measured-skip graph inserts distinct hookable gates on the branch, shortcut,
and post-addition tensors. Gate capacity is
`EMA E[mean_spatial |activation * per-example-gradient|]`; projection shortcuts
retain their measured convolution transport. In exact toy graphs, a shortcut
that saturates measured downstream demand correctly makes branch deletion have
zero scalar-flow damage.

That behavior is precisely why the neural result fails. On the 120 `conv1`
discovery groups, 99 received zero skip-aware marginal value. Catastrophic
AUROC fell to 0.408 and average precision to 0.050. On `conv2`, skip-aware flow
reached 0.971 AUROC and 0.787 AP, but did not beat Fisher or magnitude and its
layer-macro Spearman was 0.010.

The coarse-class variant uses the official 20 CIFAR-100 superclasses as flow
commodities. Class-conditional gate demand reweights shared convolution
capacity, and normalized deletion damage is aggregated by worst class, top-3
mean, and class mean. It is finite and nondegenerate, but adds almost nothing:

| Discovery table | Ordinary flow AUROC | Best local class-flow AUROC | Group Fisher AUROC |
|---|---:|---:|---:|
| `conv1` | 0.9591 | 0.9649 | 0.9956 |
| `conv2` | 0.9632 | 0.9598 | 0.9788 |

For `conv1`, the class-mean improvement over ordinary flow was only `+0.0044`
and the Fisher hybrid became worse. For `conv2`, class conditioning reduced
AUROC. Calibration plus local commodity solving took about 40 seconds, versus
4 seconds for collecting Taylor and Fisher together.

A final global graph connected the stem through all residual additions to
coarse-class classifier sinks, including measured identity and projection
shortcuts. The exact solver took 47 seconds for a one-batch `conv1` smoke score,
then assigned numerical zero to 118 of 120 groups and produced AUROC 0.491.

Because independent class flows can each reuse the shortcut, a genuine
shared-capacity maximum-concurrent-flow LP was also implemented. Twenty coarse
class commodities maximize their common minimum flow over 6,600 shared edges;
SciPy/HiGHS solves the sparse LP in 2.88 seconds. Capacity shadow prices are
converted into a normalized deletion value for every group. With the full
eight-batch calibration, however, all 120 `conv1` dual deletion values were
exactly zero. The optimum routes through residual shortcuts and is bottlenecked
outside every learned hidden branch. This is structural degeneracy, not
one-batch noise.

This exposes the central limitation of the current theory: scalar capacity
makes two paths interchangeable when they can carry the same amount, while a
learned residual transformation and an identity shortcut are not semantically
interchangeable. Class labels at the sink do not repair the missing feature
identity. Literal max-flow/min-cut on this scalar mapping is therefore
falsified as a pruning score at this checkpoint. A future method would need
vector- or representation-aware compatibility, not another scalar capacity
normalization.

## Matrix-Fisher spectral pilot

The first representation-aware replacement is now implemented in
`spectral_fisher.py`. For each structured gate group `g`, it retains the signed
per-example mask-gradient vector `v_g` and defines a rank-one PSD capacity

```text
C_g = v_g v_g^T / n
C   = sum_g C_g
```

All residual-branch groups share the same calibration-example coordinate
system, so aligned responses are explicitly redundant rather than merely
equal-capacity paths. `trace(C_g)` reproduces matched Group Fisher. The new
score is the ridge leverage

```text
tau_g = v_g^T (C + lambda I)^-1 v_g / n
```

computed through the much smaller group Gram matrix. The default ridge is the
mean Group Fisher trace, fixed without deletion labels. Collection takes one
ordinary backward pass per calibration batch, keeps the matrix calculation on
CPU, fails on non-finite loss or gradients, and does not materialize an
`examples x examples` operator.

The exact eight-batch, 1,024-example epoch-20 discovery result is negative for
single-group catastrophe ranking:

| Table | Spectral Fisher AUROC / AP | Group Fisher AUROC / AP | Best prior comparator |
|---|---:|---:|---:|
| `conv1` | 0.980 / 0.786 | 0.996 / 0.944 | Group Fisher |
| `conv2` | 0.959 / 0.693 | 0.979 / 0.754 | Magnitude: 0.989 / 0.861 |

The PSD trace agrees with the existing Group Fisher implementation to within
`9.5e-5`, validating the collection path. Calibration took 18--20 seconds on
CPU and the 240-group spectral solve took at most 0.04 seconds. The leverage
score is highly correlated with Group Fisher (`rho` about 0.97--0.98) but its
redundancy discount demotes some individually catastrophic groups.

This does not yet falsify spectral subset selection. Leverage scores are
designed to preserve a *sum* of PSD contributions under joint sparsification,
usually with rescaling of retained terms; individual unrescaled deletion AUROC
is not that objective. The next valid experiment, if pursued, is therefore a
frozen joint-mask test at several sparsities against Fisher and magnitude—not
another hybrid fitted to these deletion labels. Reweighting and BatchNorm
handling must be declared before looking at new joint-mask outcomes.

### Frozen joint-mask result

That joint test is now complete. `spectral_pruning.py` implements exact-budget
dependent rounding with leverage inclusion probabilities, the classical
`1/sqrt(q_g)` retained-vector rescaling, temporary multi-gate masks, common
BatchNorm-statistics recalibration, and a direct ridge-whitened operator-error
measurement. The practical arm retains the globally highest leverage groups
without rescaling. Fisher and magnitude use the same global group budget.

The gate was fixed before evaluation: after identical BN recalibration, one
spectral arm had to beat Fisher by at least 0.5 accuracy point at two of 25%,
50%, and 75% group sparsity, with no regression worse than 0.5 point at the
third. Three fixed dependent-rounding replicates were used for the rescaled
arm. The official CIFAR-100 test set remained locked.

| Group sparsity | Magnitude | Fisher | Spectral top-k | Rescaled spectral mean ± SD |
|---:|---:|---:|---:|---:|
| 25% | 32.92% | 54.38% | 54.82% | 54.04% ± 0.84% |
| 50% | 17.96% | 31.36% | 29.42% | **36.68% ± 0.75%** |
| 75% | 12.76% | 15.26% | 15.36% | 10.63% ± 0.72% |

The dense checkpoint scored 54.62% immediately and 60.20% after the same
1,024-example running-statistics replacement. The unweighted spectral arm
differed from Fisher by `+0.44`, `-1.94`, and `+0.10` points. The rescaled arm
differed by `-0.34`, `+5.32`, and `-4.63` points. Neither cleared the frozen
gate.

The operator diagnostics make the isolated 50% result less persuasive as a
spectral claim. Rescaling kept the Fisher trace close to one in expectation
(observed mean trace ratios 1.001, 0.996, and 0.986), but its ridge-whitened
spectral error was 0.319, 0.620, and 1.338 as sparsity increased. At 50%, where
network accuracy improved sharply, its operator error was worse than Fisher
top-k's 0.506. At 75%, both operator preservation and accuracy failed. Thus the
50% gain is not presently explained by superior preservation of the declared
operator and cannot justify tuning a sparsity-specific method.

Per the frozen decision rule, task-conditioned Fisher-operator sparsification
stops here as a proposed general pruning method. The implementation and the
isolated 50% observation are retained as negative evidence, but no new seed,
checkpoint, or official-test evaluation is authorized by this result.
