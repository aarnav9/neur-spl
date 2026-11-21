# Neuron split research notes

Canonical completed-result ledger: `results.md`. Keep hypotheses and extended
design/literature notes here, but record every completed, failed, null, or
claim-changing experimental outcome in `results.md` throughout the project.

## Working hypothesis

Duplicate selected SwiGLU FFN neurons with an exactly function-preserving split,
then separate the sibling neurons along a negative-curvature direction and train
only the small expanded subnetwork. The goal is to allocate capacity where the
task loss indicates that one neuron is locally insufficient.

## SmolLM2-135M proof-of-mechanism result

- Model: `SmolLM2-135M-Instruct`
- Layer: 15
- Split neurons: 8 of 1,536 FFN neurons
- Added parameters: 13,824 of 134,515,008 (`0.0103%`)
- The exact split changed NLL only at approximately `1e-7` numerical scale.
- The refined splitting curvature predicted the observed small-epsilon loss
  change closely on the 16-example curvature set.
- The curvature-selected split lowered NLL on disjoint validation data and on a
  locked 16-example test set.
- Locked `epsilon=0.65` test NLL: `1.31975473 -> 1.22046543`, a `0.09928930`
  reduction and approximately `9.45%` perplexity reduction.
- At `epsilon=0.65`, no matched random control among 30 arms approached the
  curvature result. The best random-arm NLL improvement was `0.000304`, versus
  `0.075017` for curvature on validation.

## Important negative benchmark result

The loss improvement did not translate into GSM8K greedy-decoding accuracy.

- Locked test: original `1/16`, curvature split `0/16`.
- Validation generation sweep: original `2/32`.
- `epsilon` 0.05, 0.1, and 0.2 retained `2/32` but added no correct answers.
- `epsilon` 0.4 and 0.65 fell to `0/32`.
- Prediction changes increased with epsilon: 0, 2, 5, 13, 25, and 30 of 32
  examples for epsilon 0, 0.05, 0.1, 0.2, 0.4, and 0.65 respectively.

Interpretation: the present curvature objective is full gold-solution NLL under
teacher forcing. It can improve the probability of reference reasoning tokens
while destabilizing autoregressive greedy trajectories. The current evidence
supports curvature-guided target-domain likelihood improvement, not improved
math-solving accuracy.

## Next experimental phase

1. Use a small curvature initialization (`epsilon=0.05` or `0.1`) that remains
   near the original function.
2. Train only the duplicated neuron rows and duplicated output columns, keeping
   the rest of the model frozen.
3. Use an answer-aligned objective, or compare it directly with full-solution
   NLL, so the optimized loss better matches exact-answer evaluation.
4. Run matched exact-split and random-direction controls with the same number of
   trainable coordinates and optimizer steps.
5. Finish the development pipeline on SmolLM, then move to the local
   `Qwen2.5-0.5B-Instruct` model, whose nontrivial baseline accuracy can resolve
   benchmark changes.
6. Use a fresh held-out GSM8K sample for the next locked test. The existing
   `micro_v1/test_16.jsonl` has now been inspected and must not be reused for
   model selection.

## Qwen2.5-0.5B development result

- Baseline greedy GSM8K accuracy on `dev_v1/validation_128.jsonl`: `35/128`
  (`27.34%`).
- Layer 12 refined split: 8 of 4,864 FFN neurons, adding 21,504 parameters to
  494,032,768 (`0.00435%`).
- The exact function-preserving split changes logits only at expected float32
  numerical scale (maximum absolute difference `2.00e-5`).
- On the disjoint 128-example development validation set, every tested nonzero
  epsilon from `0.005` through `0.12` improved teacher-forced solution NLL.
- Best tested value `epsilon=0.12`: mean NLL `0.66247095 -> 0.62526288`, a
  reduction of `0.03720808` (`5.62%` of baseline NLL); perplexity
  `1.93958 -> 1.86874`, a `3.65%` reduction.
- An extended calibration-only line search produced a clear finite optimum
  among the tested values: NLL was `0.65313933` at `epsilon=0.12`, reached its
  minimum of `0.65040537` at `epsilon=0.16`, rose to `0.66813659` at
  `epsilon=0.20`, and degraded sharply at `epsilon=0.30` and `0.40`.
- Lock `epsilon=0.16` for the next validation NLL and greedy-generation tests.
  Do not tune it further using validation results.
- The locked `epsilon=0.16` then generalized on the 128-example development
  validation set: mean NLL `0.66247095 -> 0.62267835`, a reduction of
  `0.03979260` (`6.01%` of baseline NLL); perplexity `1.93958 -> 1.86391`, a
  `3.90%` reduction. This is slightly stronger than the previously measured
  validation result at `epsilon=0.12`.
- The local quadratic curvature prediction ceases to describe the loss at
  moderate/large epsilon: it predicts continued improvement while measured NLL
  reverses after `0.16`. This is expected because curvature only describes the
  neighborhood of the exact split.
- The development validation set has now informed model selection and must not
  be reported as the final untouched test. After the method and epsilon are
  fixed, evaluate once on a fresh, excluded GSM8K test sample.

## Qwen locked-step generation result

The locked `epsilon=0.16` likelihood gain did not translate into greedy GSM8K
accuracy. Accuracy fell from `35/128` (`27.34%`) to `20/128` (`15.62%`): 13
examples remained correct, 22 changed from correct to wrong, and 7 changed from
wrong to correct. The paired exact McNemar p-value is `0.00813`, so the observed
degradation is unlikely to be a chance imbalance among the changed answers.

This is not primarily an output-formatting or truncation failure. The split
model emitted an explicit final-answer marker more often (`39` versus `19`) and
hit the 128-token cap less often (`43` versus `53`), but changed 117 of 128
extracted predictions and 127 of 128 completions. Together with the SmolLM
result, this is replicated evidence that lowering teacher-forced gold-solution
NLL along the negative-curvature split direction can destabilize free-running
reasoning and reduce exact-answer accuracy.

Before localized training, screen small epsilons on only the first 32 development
validation examples. Select by accuracy, breaking ties by fewer changed
predictions and then smaller epsilon, and evaluate the selected value once on
the remaining 96 examples. This separates behavioral-step selection from its
next accuracy evaluation.

The completed 32-example behavioral screen selected `epsilon=0.08`. Baseline
accuracy was `7/32`. Epsilons `0.005`, `0.01`, `0.016`, and `0.02` all retained
`7/32`; `0.04` fell to `6/32`; and `0.08` reached `8/32`, with 3 wrong-to-correct
and 2 correct-to-wrong transitions. The `0.08` tuning result is only a net gain
of one example (paired exact McNemar `p=1.0`), so it is a candidate rather than
evidence of improvement. Lock it and evaluate once on the remaining 96 examples,
whose original-model baseline is `28/96`.

On that untouched 96-example accuracy slice, the locked `epsilon=0.08` split
scored `29/96` versus the original model's `28/96`. There were 10
wrong-to-correct and 9 correct-to-wrong transitions, with exact paired McNemar
`p=1.0`. Thus the small step preserves accuracy and has a directionally positive
one-example difference, but provides no statistical evidence of an accuracy
gain. Combining the tuning and evaluation slices descriptively gives `37/128`
versus `35/128` (13 gains and 11 losses), but that combined number is optimistic
because the first 32 examples selected epsilon.

## Math conditioning versus math specificity

The current neurons and split directions are **math-conditioned**, not yet
demonstrated to be math-specific. They were selected using the second derivative
of teacher-forced GSM8K full-solution NLL on 64 examples. This proves that the
selected parameter directions are locally useful for that objective, but an FFN
neuron can be polysemantic and participate in math, language, knowledge, and
other behaviors.

Add two complementary experiments:

1. A curvature-specificity matrix: derive neurons/directions separately on math
   and general-domain calibration data, then evaluate each direction on both
   losses. Report top-k neuron overlap, rank correlation, direction cosine for
   shared neurons, and the 2-by-2 cross-domain loss effects. This directly tests
   whether the math-derived signal is specialized.
2. A general-capability retention suite: compare original and split models on
   lightweight, fixed multiple-choice subsets such as HellaSwag, ARC-Challenge,
   BoolQ, and a non-math MMLU slice, plus general-text NLL. This measures
   collateral damage; it does not replace GSM8K as the target metric.

The first 192-example paired retention screen at locked `epsilon=0.08` found no
general-capability degradation. Overall accuracy was `96/192 -> 100/192`, with 9
wrong-to-correct and 5 correct-to-wrong transitions (`p=0.42395`). All three
tasks moved in the same positive direction: ARC-Challenge `33/64 -> 34/64`,
BoolQ `36/64 -> 38/64`, and HellaSwag `27/64 -> 28/64`. These are fixed
development subsets and the changes are not statistically significant, so the
result supports retention and a possible broad effect, not a claimed general
benchmark improvement. A standardized full-suite evaluation remains necessary
for a paper headline.

Cross-domain directional curvature shows that the eight GSM8K-derived directions
are not purely math-specific. Their math curvature sums to `-50.8552`, while
their equally weighted general multiple-choice curvature sums to `-21.0564`;
7 of 8 directions are negative on the general mixture. The effect is strongly
task dependent: ARC-Challenge is locally adverse (`+34.6590`), BoolQ strongly
beneficial (`-74.7595`), and HellaSwag beneficial (`-23.0689`). Math versus
general curvature rank among the eight neurons has essentially zero Spearman
correlation (`-0.0238`). The best interpretation is polysemantic, broadly
splittable capacity with domain-dependent effects, rather than a discrete set of
math neurons.

At `epsilon=0.08`, the general quadratic prediction has the correct aggregate
sign: predicted correct-answer NLL delta `-0.06738`, measured `-0.04039`. It also
gets each task's sign right, but substantially overestimates the magnitude,
again showing that curvature is a local guide rather than an accurate finite-step
loss model.

## Qwen matched-control result at epsilon 0.08

The 128-example solution-NLL improvement survives all three matched control
families. Original mean NLL was `0.66247095`; the curvature split reached
`0.62924725`, a delta of `-0.03322370` and a `3.27%` perplexity reduction. The
exact split itself changed NLL by only `+9.02e-8`.

- Same neurons with random directions: mean delta `+9.97e-7`; best of 10
  `-9.87e-5`; `0/10` beat curvature.
- Random neurons with curvature directions: mean delta `-0.0021690`; best of 10
  `-0.0078586`; `0/10` beat curvature.
- Random neurons with random directions: mean delta `+1.60e-6`; best of 10
  `-1.63e-5`; `0/10` beat curvature.

The curvature effect is `15.32x` the magnitude of the average random-location
curvature effect and `4.23x` the best such seed. Each family-level randomization
tail estimate is at its 10-control resolution limit, `(0 + 1) / (10 + 1) =
0.0909`; this should be reported as `0/10` rather than as conventional
significance. The result is strong causal evidence that both neuron location and
direction matter for the differentiable likelihood effect. It still does not
establish an exact-answer accuracy gain.

For an infinitesimal symmetric split, the first derivative cancels and negative
curvature predicts lower calibration loss as
`delta L ~= 0.5 * curvature * epsilon^2`. At `epsilon=0.016`, the Qwen top-eight
curvature sum predicts approximately `-0.00651` NLL locally, and the measured
`0.01`/`0.02` values support an NLL improvement in that neighborhood. There is
still no monotonic guarantee for greedy exact-answer accuracy, which is a
discrete autoregressive metric rather than the differentiable objective used to
select the split.

## Later inverse operation: merge/compression

Investigate merging sibling or nearby neurons when the loss curvature along the
antisymmetric separation coordinate indicates that collapsing them is locally
safe. A merge criterion should use the predicted loss increase near the merged
state, not merely a negative first derivative: sibling convergence and safe
collapse are different conditions. Compare curvature-guided merging with weight
similarity, activation similarity, magnitude pruning, and second-order pruning.
Novelty must be checked against neuron merging, model morphism, Optimal Brain
Damage/Surgeon-style pruning, and modern LLM structured-pruning literature before
making a paper claim.

### Concrete curvature-guided merge formulation

For a candidate pair of SwiGLU neurons `i,j`, write their gate/up weights as a
center plus a difference and introduce an explicit collapse coordinate:

`g_i(e)=g_c+e*d_g`, `g_j(e)=g_c-e*d_g`,
`u_i(e)=u_c+e*d_u`, `u_j(e)=u_c-e*d_u`.

The existing pair is represented near `e=1`; at `e=0` the gate/up weights are
identical and the neurons can be replaced by one neuron whose down-projection
column is `v_i+v_j`. For exact siblings with equal gate/up weights this merge is
function preserving. For arbitrary existing neurons, the center and collapse
path may need an outgoing-weight-aware fit because unequal down columns break
the simple symmetric first-derivative cancellation.

Candidate generation can use activation similarity and gate/up/down weight
similarity, but similarity alone must not authorize pruning. Estimate the loss
along the collapse coordinate and merge only when the measured or conservatively
bounded increase from `e=1` to `e=0` is small. Around a genuinely symmetric
merged state, nonnegative/near-zero splitting curvature means separation is not
locally useful and supports merging; negative splitting curvature means the pair
benefits from remaining separated and must not be collapsed. Because an
arbitrary pair's full collapse is a finite move, validate promising pairs at
several points such as `e=1,0.5,0` rather than trusting a local Taylor expansion
alone.

A training-time version should parameterize pairs directly as center and
difference, regularize selected difference vectors toward zero, and perform an
exact structural merge once the difference norm and calibrated loss bound are
both below thresholds. Localized distillation/recovery training can follow each
batch of merges.

For Qwen layer 12, one SwiGLU neuron contains `3*896=2,688` structural parameters
(one gate row, one up row, and one down column). Merging eight pairs would remove
exactly the 21,504 parameters added by the current eight-neuron split. This makes
a fixed-width merge-split reallocation experiment especially clean: merge the
eight lowest-cost redundant pairs and spend that capacity on the eight most
negative-curvature splits, then compare against pruning-only, splitting-only,
and random merge-split controls under the same parameter count.

### Provisional novelty audit (2026-08-29)

Keep this fork, but do not describe the broad ingredients as novel. The search
found extensive precedent for each ingredient in isolation:

- Splitting Steepest Descent (Liu, Wu, and Wang, NeurIPS 2019) derives a
  second-order splitting matrix and uses its most negative eigenvalue/eigenvector
  to choose and perturb neurons, but only grows networks:
  https://arxiv.org/abs/1910.02366
- Neuron Merging (Kim et al., NeurIPS 2020) merges similarity-selected neurons
  and compensates in the following layer; it is data-free and does not use the
  splitting coordinate or its curvature:
  https://arxiv.org/abs/2010.13160
- Optimal Neuron Merging (Goldberg et al., ICPRAI 2022) gives a reconstruction
  error bound and optimizes the prune-and-merge assignment, rather than using
  loss curvature on a reversible split/merge path:
  https://doi.org/10.1007/978-3-031-09037-0_56
- RED/RED++ merges redundant neurons exposed through weight hashing and
  similarity, without the proposed loss-curvature stability test:
  https://arxiv.org/abs/2110.01397
- Taylor, Hessian, and Fisher pruning already use first- or second-order loss
  information to delete parameters or neurons; curvature-based pruning alone is
  therefore not a novelty claim. A representative neuron-level Taylor method is:
  https://arxiv.org/abs/1906.10771
- Dynamic Sparse Reparameterization (Mostafa and Wang, ICML 2019) prunes and
  reallocates individual nonzero weights under a fixed parameter budget, but it
  does not perform dense-neuron merging and curvature-selected neuron splitting:
  https://proceedings.mlr.press/v97/mostafa19a.html
- DOTResize (Verma, Murray, and Duh, 2025) is the closest LLM-specific merging
  method found. It reduces transformer width by optimal-transport reprojection
  from activation signatures, not by reversing splitting curvature:
  https://arxiv.org/abs/2507.04517
- CoCurve (Gong et al., 2026) is the closest recent curvature-pruning paper. It
  uses Fisher off-diagonal interactions to choose groups of attention/FFN units
  to remove under a budget; it does not merge two neurons along a collapse
  coordinate, test splitting stability, or re-spend the unit on a split:
  https://arxiv.org/abs/2607.17568

No searched paper was found that combines all of the following: (1) a reversible
center/difference coordinate connecting two distinct gated-FFN neurons to one
merged neuron, (2) pair selection and merge acceptance based on the task loss
along that coordinate, (3) the splitting curvature at/near the collapsed state
as a local certificate that the merge will not immediately benefit from
re-separation, and (4) reallocation of each freed dense FFN neuron to a
negative-curvature split elsewhere in the LLM. This precise combination is the
provisionally novel contribution.

The safest eventual claim is therefore **bidirectional curvature-guided dense
FFN capacity reallocation**, not “curvature-based pruning” or “neuron merging.”
A compression-only result remains useful, but is more crowded and would need to
beat strong merging and structured-pruning baselines. The merge-split result is
the more defensible research contribution because it unifies the inverse
operations under one structural coordinate and can improve allocation without
increasing the neuron or parameter count. This audit is evidence from a focused
search, not a proof that no uncatalogued or newly released paper exists; repeat
the search immediately before submission.
