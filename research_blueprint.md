# Cut-to-Grow Research Blueprint

Declared: `2026-08-24`  
Status: graph-level prototype plus one negative strong-baseline checkpoint result

## Objective

Test whether a fixed-budget sparse neural network can learn a better topology
by modeling channels or connection groups as a task-conditioned transport
network. Minimum-cut edges are bottlenecks to protect—not edges to delete.

The paper-worthy question is deliberately narrow:

> Can task-conditioned minimum cuts guide online removal and regrowth of
> hardware-structured neural capacity at acceptable end-to-end cost?

## Why this may help

Magnitude pruning assumes a small weight is unimportant. That can fail when a
small connection is the only useful entrance to a downstream feature, while a
large connection lies on a redundant path. The proposed method estimates each
group's functional transport using activations, weights, and downstream loss
gradients, then reasons about the global paths those groups support.

This is not a claim that neural networks literally conserve flow. Signed
weights, cancellation, nonlinearities, batch normalization, and residual paths
all break that analogy. The score is useful only if it predicts interventions.

## Proposed method

At a topology update, coarsen the model into a directed graph. Nodes are feature
groups and edges are channels, constant-fan-in blocks, N:M groups, attention
heads, or MLP dimensions. Scalar-weight graphs are diagnostic only.

For an edge `e=(u,v)`, start with the task-conditioned capacity

```text
c_e = EMA E[|a_u W_e delta_v|]
```

and compare it against squared/Fisher transport, signed transport, magnitude,
and forward-only activation scores. Cross-layer scores require declared
normalization; the default candidate divides by the geometric mean of incoming
and outgoing capacity.

For each class or clustered gradient direction:

1. Estimate edge capacities.
2. Compute source-to-output maximum flow and its minimum cut.
3. Protect edges that repeatedly occur in critical cuts.
4. Remove active groups with low counterfactual flow damage and low loss
   saliency.
5. Grow inactive high-gradient groups that cross the newly weak cut.
6. Preserve the active group budget exactly and continue training.

The exact diagnostic quantities are

```text
deletion value(e) = F(mask) - F(mask without e)
growth value(e)   = F(mask with e) - F(mask)
```

Exact recomputation is an oracle for small graphs. A scalable version would
use cut frequency, residual capacity, warm starts, sampled commodities, or a
critic trained to predict deletion damage.

## Hypotheses

- **H1:** Flow scores rank measured deletion damage better than magnitude,
  activation-weighted magnitude, `|w*grad|`, or diagonal Fisher.
- **H2:** Minimum-cut membership finds low-magnitude groups whose removal is
  unusually damaging.
- **H3:** At equal sparsity and topology-update budget, cut-guided regrowth
  improves on gradient-only dynamic sparse training.
- **H4:** Structured grouping eventually converts the algorithmic result into
  measured latency or memory savings.

## Phase A: falsify the signal first

Use CIFAR-100 with a CIFAR-style ResNet-18. Split the official training data
deterministically into `45,000` fit and `5,000` development-selection examples;
keep the official test set locked. Use seeds `20260824`, `20260825`, and
`20260826`.

At early, middle, and late checkpoints:

1. sample output-channel groups across layers and score quantiles;
2. delete one group at a time without repair;
3. measure the immediate development loss and accuracy change;
4. restore the checkpoint; and
5. correlate each pre-deletion score with measured damage.

Primary metric: layer-macro Spearman correlation. Secondary metrics: AUROC for
the top 5% most damaging deletions, precision at `k`, calibration, scorer time,
and monitor overhead.

Proceed only if flow improves Spearman correlation by at least `0.10` over the
strongest non-flow comparator, or improves critical-deletion AUROC by `0.05`,
at two of the three training stages. If within-layer shuffling keeps the gain,
the edge-selection claim fails; at most it is a layer-allocation result.

## Phase B: dynamic sparse training

Only after Phase A passes, compare fixed masks, gradual magnitude pruning, SET,
RigL, Structured RigL, and Cut-to-Grow at `80%`, `90%`, and `95%` diagnostic
sparsity. Match initialization, layer budgets, optimizer, schedule, training
steps, number of topology changes, and tuning budget.

Scale the method only if it achieves one of these:

- at least `0.5` accuracy point over the strongest tuned dynamic baseline; or
- matched accuracy with topology scoring at least `10x` cheaper.

The paired effect must exceed two standard errors, at least two sparsity points
must move in the same direction, and topology updates must consume no more than
`15%` of pilot training time.

## Phase C: paper-level evidence

Freeze the method before moving beyond development data. A strong paper should
include a second CNN or a small Vision Transformer, one non-CIFAR benchmark,
full accuracy-versus-sparsity curves, end-to-end training cost, peak memory,
and measured structured inference latency. CIFAR-only evidence is a pilot, not
a broad performance claim.

## Required controls

- dense and equivalently smaller dense networks;
- random masks with the same layer allocation;
- magnitude, activation-weighted magnitude, `|w*grad|`, and Fisher;
- SNIP, GraSP, and SynFlow for initialization diagnostics;
- fixed sparse masks, SET, RigL, and Structured RigL for dynamic training;
- within-layer score shuffling, score inversion, and layer-mean replacement;
- direct versus effective sparsity; and
- theoretical FLOPs versus measured runtime.

## Closest prior work and novelty boundary

- [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635): iterative
  magnitude pruning and rewinding, rather than online sparse-to-sparse growth.
- [SynFlow](https://arxiv.org/abs/2006.05467): flow-like products that avoid
  layer collapse, but data-free and not min-cut-guided dynamic growth.
- [RigL](https://proceedings.mlr.press/v119/evci20a.html): the closest dynamic
  baseline; magnitude removal and missing-gradient growth.
- [Spectral graph sparsification](https://proceedings.mlr.press/v221/laenen23a.html)
  and [mask topology](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd5404354496e39d37b7947d8a0d7b72-Abstract-Conference.html): strong evidence
  that topology already matters.
- [Neural-curvature pruning](https://openreview.net/forum?id=kwACVY73Ug): the
  closest bottleneck-preservation precedent.
- [Structured RigL](https://arxiv.org/abs/2305.02299): the hardware-relevant
  structured dynamic baseline.
- [Taylor pruning](https://openreview.net/forum?id=SJGCiw5gl): first-order
  activation-gradient importance for structured feature-map removal.
- [Group Fisher pruning](https://proceedings.mlr.press/v139/liu21ab.html):
  per-example squared shared-mask gradients with structural dependency groups.
- [FlowPrune](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c2cd7fa98591516c025d7632b2586ca1-Abstract-Conference.html): max-flow/min-cut
  on attention interpretation graphs, not model-weight training.

Graph-aware pruning, bottleneck preservation, and max-flow analysis are not
new by themselves. The defensible gap is intervention-validated,
task-conditioned min-cut guidance for online structured removal and regrowth,
with the scoring cost included.

## Failure modes

- Flow may only rediscover layer allocation.
- Absolute capacity may hide cancellation; test signed and squared variants.
- Aggregate flow may sacrifice minority classes; report class-conditioned and
  worst-class retained flow.
- Residual paths may dominate; test normalization and block-level supernodes.
- Exact flow may be too expensive; profile before proposing a learned critic.
- Unstructured zeros may compress without accelerating hardware; do not claim
  latency until structured measurements exist.

## Immediate work

Implemented: CIFAR ResNet-18 group extraction, deterministic training and
checkpoint code, the pinned 45,000/5,000 data split, exhaustive size-16
`conv1` deletion probes at epoch 20, magnitude/local-transport/flow reporting,
matched Taylor and gate-level Group Fisher comparators, paired uncertainty, and
within-layer shuffle controls.

At this checkpoint, flow detects catastrophic deletions better than magnitude
but loses to Group Fisher: AUROC 0.959 versus 0.996 and average precision 0.674
versus 0.944. The paired Group-Fisher-minus-flow AUROC interval is entirely
positive. Flow also costs about nine times as much to score in the current
CPU-stabilized implementation. This fails the declared standalone comparison
at epoch 20 and blocks Phase B for now.

A leakage-resistant conditional test was then frozen before acquiring the 120
`conv2` deletion labels. The fixed Fisher-flow hybrid improved AUROC over
Fisher by only `+0.0134 [-0.0089, +0.0446]`, below the declared `+0.05` gate.
Magnitude was also competitive with the hybrid, and within-layer controls
showed that pooled performance was dominated by shallow-layer allocation. The
conditional test therefore failed and Phase B remains blocked.

Any next attempt must introduce a genuinely redesigned score—such as measured
skip-path capacity or class-conditioned multi-commodity flow—freeze it before
new labels, and evaluate on a new checkpoint or seed. The present branch-local
flow score should not be advanced to dynamic sparse training.

Both suggested redesigns were subsequently implemented and checked only on the
existing discovery labels. Measured shortcut capacity collapsed 99 of 120
`conv1` scores to zero and yielded 0.408 catastrophic AUROC. A 20-commodity
coarse-class ensemble improved ordinary `conv1` flow by only 0.004 AUROC,
worsened `conv2`, and remained slower and weaker than Group Fisher. Extending
the commodities to a global stem-to-class graph collapsed 118 of 120 `conv1`
scores and yielded chance-level AUROC.

A final true multi-commodity control used a sparse maximum-concurrent-flow LP:
20 coarse-class flows compete for 6,600 shared edge capacities, and group value
is derived from the capacity duals. The complete eight-batch result assigned
zero value to all 120 `conv1` groups. Shared capacity therefore does not repair
the semantic-substitution problem; the LP places its active bottleneck on the
shortcut/classifier route.

No new deletion labels were acquired. The scalar-flow program is now blocked
at the representation itself: residual identity and learned transformation
paths are capacity-substitutable in max-flow but not functionally
substitutable in the network. Continuing would require a new vector-valued or
feature-compatibility theory and a new preregistration, not tuning this score.

A first PSD replacement has now been implemented and evaluated only on the
already-seen epoch-20 tables. It represents each structured group by the outer
product of its signed per-example mask-gradient vector. Group Fisher is the
trace of this matrix; ridge leverage measures the group's unique direction in
the global empirical Fisher operator. This directly removes scalar-path
substitutability and costs one backward pass per calibration batch.

As an individual-deletion score it did not beat the trace baseline: catastrophic
AUROC/AP were 0.980/0.786 versus Fisher's 0.996/0.944 on `conv1`, and
0.959/0.693 versus 0.979/0.754 on `conv2`. These are discovery results, not a
fresh confirmation test. Spectral sparsification is fundamentally a joint
operator-preservation claim, so the only justified continuation is to freeze a
subset-selection and retained-group reweighting rule, then test complete masks
at unseen sparsity settings/checkpoints. A single-group leverage hybrid tuned
on these failures is explicitly out of scope.

The prespecified joint-mask experiment has now also failed its continuation
gate. After common 1,024-example BatchNorm recalibration, deterministic
unweighted leverage differed from Fisher by `+0.44`, `-1.94`, and `+0.10`
accuracy points at 25%, 50%, and 75% group sparsity. Exact-budget dependent
leverage sampling with `1/sqrt(q_g)` activation rescaling differed by `-0.34`,
`+5.32`, and `-4.63` points, averaged over three fixed mask replicates. It won
strongly only at the middle sparsity and failed badly at the highest.

The rescaled masks preserved total Fisher trace, but not the full operator:
ridge-whitened spectral error rose from 0.319 to 0.620 to 1.338. In particular,
the 50% accuracy win occurred despite worse operator error than Fisher top-k,
so it does not validate the proposed mechanism. The frozen requirement of a
0.5-point win at two sparsities with no greater than 0.5-point regression was
not met. This research branch therefore stops without new checkpoint, seed, or
test-set evaluation. Any later revival requires a substantively different
operator and a new preregistration, not selecting the favorable 50% result.
