# Curvature-guided neuron splitting: cumulative results ledger

Last updated: 2026-08-29

This is the canonical running record for the entire experiment and chat. Update
it after every completed run, failed run that changes the method, statistical
conclusion, data-split decision, or implementation change that affects how a
result should be interpreted. Preserve negative and null results alongside
positive results. `notes.md` may contain longer hypotheses and literature notes,
but claims about what was actually run and found belong here.

## Current bottom line

Curvature-guided symmetric splitting produces a large, reproducible reduction in
teacher-forced target-domain NLL using only eight added SwiGLU neurons. Matched
controls on both SmolLM2-135M and Qwen2.5-0.5B show that the effect requires both
the selected neuron locations and the selected negative-curvature directions.

That likelihood result has **not become a demonstrated math-accuracy gain**.
Large separation steps significantly damage Qwen greedy GSM8K accuracy; a smaller
locked step preserves it and moves accuracy by only `+1/96`, with `p=1.0`.
The first localized mixed-objective sibling-training method is decisively
negative: it reduced paired validation accuracy from `35/128` to `6/128`
(`p=4.18e-7`) and taught the model to emit a terse final answer without doing the
reasoning requested by the prompt. This mixed answer-only/full-solution objective
is retired; no final holdout should be spent on it.

## Claim status

### Supported by current evidence

- Exact neuron duplication with halved outgoing weights is function-preserving to
  expected float32 numerical precision.
- Negative splitting curvature predicts local teacher-forced NLL improvement.
- The finite-step NLL improvement generalizes from curvature/calibration data to
  disjoint development validation data on both tested model sizes.
- Qwen's `epsilon=0.08` NLL effect is substantially larger than matched random
  location and random direction controls.
- The selected Qwen directions are math-conditioned but not purely math-specific;
  they have mixed, task-dependent effects on general-domain objectives.
- At `epsilon=0.08`, the tested general-capability development suite shows no
  detected collateral accuracy loss.
- The v1 mixed localized-training objective causes severe free-running behavioral
  collapse and should not be used as evidence that localized split training
  improves reasoning.

### Not supported yet

- No statistically credible GSM8K exact-answer improvement has been shown.
- No final untouched Qwen test result has been run after freezing a trained
  method.
- The present fixed development retention suite is not a standardized paper
  headline.
- The selected neurons should not be described as discrete "math neurons."
- Broad neuron splitting, curvature-based pruning, or neuron merging are not by
  themselves novel contributions.

## Experimental conventions and environment

- Primary device: Apple MPS.
- Qwen inference/training dtype: `torch.float32`. A float16 diagnostic exhibited
  numerical/repeated-token instability, so float16 results are not used as the
  trusted comparison path.
- Generation is greedy with the fixed concise math system prompt and exact-answer
  extraction used by `src/neurspl/evaluate.py`.
- NLL means token-weighted teacher-forced cross entropy on assistant targets.
- Negative NLL deltas are improvements.
- Paired accuracy comparisons use the exact two-sided McNemar/sign test on
  discordant examples.
- Random-control tail estimates use the plus-one convention. With 10 controls,
  `0/10` beating the treatment has a minimum family-level tail fraction of
  `1/11 = 0.0909`; report the actual `0/10` count and do not call that family
  conventionally significant.

## Models and data status

### Models

- `SmolLM2-135M-Instruct`: 30 layers, hidden size 576, SwiGLU width 1,536,
  134,515,008 parameters before splitting.
- `Qwen2.5-0.5B-Instruct`: 24 layers, hidden size 896, SwiGLU width 4,864,
  494,032,768 parameters before splitting.

### Development subsets

- `subsets/micro_v1`: early SmolLM mechanism development.
- `subsets/dev_v1/train_512.jsonl`: localized-training data.
- `subsets/dev_v1/curvature_64.jsonl`: Qwen curvature selection/refinement.
- `subsets/dev_v1/validation_128.jsonl`: Qwen development validation. It has now
  been used for NLL line searches, behavioral epsilon selection, accuracy
  evaluation, and controls; it is fully development data.
- `subsets/general_retention_v1/eval_192.jsonl`: fixed development suite with 64
  ARC-Challenge, 64 BoolQ, and 64 HellaSwag examples.

### Holdout hygiene

- `subsets/micro_v1/test_16.jsonl` has been inspected and is retired from future
  model selection or confirmatory claims.
- The first 32 Qwen validation examples selected `epsilon=0.08`; the remaining 96
  were untouched for that immediate evaluation only. Subsequent work has used the
  complete 128-example set, so none of it is now a final holdout.
- After the localized-training method, objective, learning rate, step count, and
  initialization controls are fixed, create one fresh excluded GSM8K holdout and
  evaluate it once.

## SmolLM2-135M proof of mechanism

### Structure and exactness

- Split layer: 15.
- Selected neurons: `[49, 322, 381, 492, 934, 1025, 1350, 1429]`.
- Added neurons: 8 of 1,536.
- Added parameters: 13,824, or approximately `0.0103%`.
- Exact-split NLL differences were approximately `1e-7`.

### Local and held-out NLL behavior

- Curvature/calibration 16 examples, original NLL `1.07572187`:
  - `epsilon=0.025`: delta `-0.00020265`
  - `epsilon=0.05`: delta `-0.00080910`
  - `epsilon=0.10`: delta `-0.00319264`
  - `epsilon=0.20`: delta `-0.01210485`
- Disjoint validation 32 examples, original NLL `1.25834763`:
  - `epsilon=0.025`: delta `-0.00022719`
  - `epsilon=0.05`: delta `-0.00090597`
  - `epsilon=0.10`: delta `-0.00358572`
  - `epsilon=0.20`: delta `-0.01375485`
- Locked 16-example test at `epsilon=0.65`: NLL
  `1.31975473 -> 1.22046543`, delta `-0.09928930`, approximately `9.45%`
  perplexity reduction.

Artifacts:

- `runs/split_line_search_first_line_search.json`
- `runs/split_line_search_heldout_validation.json`
- `runs/split_line_search_locked_test.json`

### Matched controls at epsilon 0.65

- Curvature split validation delta: `-0.07501727`.
- Best of 10 random-neuron/curvature-direction controls: `-0.00030423`.
- Best of 10 same-neuron/random-direction controls: `-0.00027286`.
- Best of 10 random-neuron/random-direction controls: `-0.00006152`.
- `0/10` controls in every family beat the curvature arm.

Artifact: `runs/split_controls_validation_controls.json`.

### Negative generation result

- Locked 16-example test: original `1/16`, split `0/16`.
- Validation 32-example sweep: original `2/32`.
- `epsilon=0.05`, `0.10`, and `0.20` retained `2/32` but added no correct
  answers.
- `epsilon=0.40` and `0.65` fell to `0/32`.
- Changed predictions rose with epsilon: `0, 2, 5, 13, 25, 30` for
  `0, .05, .10, .20, .40, .65`.

Interpretation: teacher-forced solution likelihood can improve while
free-running reasoning trajectories become less accurate.

Artifacts:

- `runs/split_generation_locked_test_generation.json`
- `runs/split_generation_sweep_validation_generation_sweep.json`

## Qwen2.5-0.5B curvature selection and exact split

### Baseline

- Greedy GSM8K development accuracy: `35/128 = 27.34375%`.
- Artifact: `runs/qwen_baseline_qwen_validation128.json`.

### Curvature screening and refinement

- Split layer: 12.
- Curvature data: 64 examples and 7,827 target tokens.
- Random-probe screen: 4,774 of 4,864 neurons had a negative minimum proxy;
  screen-best neuron was 1161.
- Lanczos-refined top eight, in refined rank order:
  `[4613, 657, 529, 4705, 2966, 1161, 1031, 1304]`.
- Sorted structural indices:
  `[529, 657, 1031, 1161, 1304, 2966, 4613, 4705]`.
- Sum of refined minimum Ritz values: `-50.8552351`.

Artifacts:

- `runs/curvature_proxy_qwen_layer12_scan_b1.json`
- `runs/curvature_refine_qwen_layer12_refine.json`

### Exact structural expansion

- Width: `4864 -> 4872`.
- Added parameters: 21,504, or `0.00435275%` of 494,032,768.
- Maximum absolute logit difference: `2.0027e-5`.
- Mean absolute logit difference: `2.0895e-6`.
- Logits are allclose at absolute tolerance `1e-4`.
- Validation exact-split NLL delta: `+9.01765e-8`.

Artifact: `runs/split_sanity_qwen_layer12_top8.json`.

## Qwen finite-step NLL results

### Curvature/calibration line search

Original mean NLL: `0.68389240`.

| Epsilon | Mean NLL | Delta |
| ---: | ---: | ---: |
| 0.005 | 0.68326503 | -0.00062737 |
| 0.010 | 0.68149040 | -0.00240201 |
| 0.020 | 0.67592430 | -0.00796810 |
| 0.040 | 0.66444061 | -0.01945179 |
| 0.080 | 0.65374341 | -0.03014899 |
| 0.120 | 0.65313933 | -0.03075307 |
| 0.160 | 0.65040537 | -0.03348703 |
| 0.200 | 0.66813659 | -0.01575581 |
| 0.300 | 0.96488446 | +0.28099206 |
| 0.400 | 1.87024896 | +1.18635656 |

The measured curve has a finite optimum around `epsilon=0.16` among tested
values. The local quadratic prediction is useful only near zero and wrongly
predicts continued improvement at moderate/large epsilon.

Artifacts:

- `runs/split_line_search_qwen_calibration_line_search.json`
- `runs/split_line_search_qwen_calibration_extended.json`

### Development validation NLL

Original mean NLL: `0.66247095`; original perplexity approximately `1.93958`.

- `epsilon=0.08`: NLL `0.62924721`, delta `-0.03322375`; perplexity ratio
  `0.967322`, a `3.27%` reduction.
- `epsilon=0.12`: NLL `0.62526288`, delta `-0.03720808`; perplexity reduction
  approximately `3.65%`.
- Locked calibration-selected `epsilon=0.16`: NLL `0.62267835`, delta
  `-0.03979260`; perplexity `1.93958 -> 1.86391`, a `3.90%` reduction.

Artifacts:

- `runs/split_line_search_qwen_validation_line_search.json`
- `runs/split_line_search_qwen_validation_locked016.json`

## Qwen generation behavior

### Large locked step: significant degradation

At calibration-selected `epsilon=0.16`:

- Accuracy: `35/128 -> 20/128` (`27.34% -> 15.62%`).
- Both correct: 13.
- Original-only correct: 22.
- Split-only correct: 7.
- Exact paired McNemar `p=0.00813`.
- Extracted predictions changed on 117/128 examples; completions changed on
  127/128.
- This was not primarily a formatting or truncation failure: final-answer markers
  increased (`19 -> 39`) and max-token truncations decreased (`53 -> 43`).

Artifact: `runs/split_generation_qwen_validation_eps016.json`.

### Small-step behavioral selection

The first 32 development examples were used to select a behaviorally safer step:

- Baseline: `7/32`.
- `epsilon=.005, .01, .016, .02`: each `7/32`.
- `epsilon=.04`: `6/32`.
- `epsilon=.08`: `8/32`, with 3 gains and 2 losses; `p=1.0`.
- Selection rule: accuracy, then fewer changed predictions, then smaller epsilon.
- Locked choice: `epsilon=0.08`.

On the immediately untouched remaining 96 examples:

- Accuracy: `28/96 -> 29/96`.
- Wrong-to-correct: 10; correct-to-wrong: 9.
- Exact paired McNemar `p=1.0`.
- Descriptive combined result: `35/128 -> 37/128`, but the first 32 selected
  epsilon, so this combined result is optimistic and not confirmatory.

Artifacts:

- `runs/split_generation_sweep_qwen_small_eps_tune32.json`
- `runs/split_generation_sweep_qwen_eps008_eval96.json`

## Qwen matched controls at locked epsilon 0.08

Original validation NLL: `0.6624709532`.

- Exact split delta: `+9.01765e-8`.
- Curvature split NLL: `0.6292472504`.
- Curvature delta: `-0.0332237028`.
- Curvature perplexity ratio: `0.96732214`.

| Control family | Mean delta | Best delta | Improved / 10 | Beat curvature |
| --- | ---: | ---: | ---: | ---: |
| Same neurons, random direction | +0.000000997 | -0.000098683 | 5 | 0/10 |
| Random neurons, curvature directions | -0.002168998 | -0.007858554 | 8 | 0/10 |
| Random neurons, random directions | +0.000001604 | -0.000016328 | 4 | 0/10 |

- Curvature is `15.32x` the magnitude of the average random-location curvature
  effect.
- Curvature is `4.23x` the best random-location curvature seed.
- It is approximately `336.7x` the best same-neuron random-direction effect and
  `2034.7x` the best random-both effect.
- Each family is resolution-limited to a plus-one tail fraction of `1/11`.
- Descriptively, `0/30` pooled controls beat curvature, but the three families
  are not automatically exchangeable; do not substitute a pooled formal p-value
  without prespecifying that test.

Interpretation: both selected location and direction causally matter for the
teacher-forced NLL effect. Randomly placing the curvature directions sometimes
helps slightly, consistent with broadly useful/polysemantic directions, but it
does not approach the selected pairing.

Artifact: `runs/split_controls_qwen_eps008_validation_controls.json`.

## General-capability retention at epsilon 0.08

Scoring used mean log likelihood of the correct answer letter on a fixed
192-example development suite.

| Task | Original | Split | Gains | Losses | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | 33/64 | 34/64 | 2 | 1 | 1.0 |
| BoolQ | 36/64 | 38/64 | 4 | 2 | 0.6875 |
| HellaSwag | 27/64 | 28/64 | 3 | 2 | 1.0 |
| Overall | 96/192 | 100/192 | 9 | 5 | 0.42395 |

- Overall predictions changed on 20/192 examples.
- Correct-answer mean log-likelihood deltas, where positive means the split
  raised gold-answer log likelihood: ARC `-0.002949`, BoolQ `+0.119659`,
  HellaSwag `+0.004458`, overall `+0.040389`.
- Mean answer-margin deltas: ARC `-0.07431`, BoolQ `+0.011355`, HellaSwag
  `-0.001473`, overall `-0.021476`.

Interpretation: no tested general task shows an accuracy decline, and the total
is directionally positive but nonsignificant. This supports retention, not a
claim of general benchmark improvement.

Artifact: `runs/general_retention_qwen_eps008_general.json`.

## Math conditioning versus specificity

The GSM8K-derived directions were evaluated on the general correct-answer-letter
objective:

- Math curvature sum: `-50.8552351`.
- General mixture curvature sum: `-21.0564461`.
- Seven of eight neurons have negative aggregate general curvature; neuron 2966
  is slightly positive (`+0.1597294`).
- Per-task general curvature sums:
  - ARC-Challenge: `+34.6590157` (locally adverse)
  - BoolQ: `-74.7594910` (strongly favorable)
  - HellaSwag: `-23.0688572` (favorable)
- Math/general Pearson correlation across the eight neurons: approximately
  `0.320`.
- Math/general Spearman rank correlation: `-0.02381`.
- At `epsilon=.08`, predicted aggregate correct-answer NLL delta was
  `-0.0673806`; measured was `-0.0403893`, with the same aggregate sign.
- Predicted versus measured task deltas had the correct sign for all three tasks
  but substantially overestimated magnitude:
  - ARC `+0.110909` predicted vs `+0.002949` measured
  - BoolQ `-0.239230` predicted vs `-0.119659` measured
  - HellaSwag `-0.073820` predicted vs `-0.004458` measured

Interpretation: these are polysemantic, broadly splittable directions with
domain-dependent effects, not a clean set of math-specific neurons. Curvature is
a local sign/selection guide, not a reliable finite-step magnitude predictor.

Artifact: `runs/cross_domain_curvature_qwen_math_directions_general.json`.

## Localized sibling training v1

The first localized-training run completed successfully.

- Initialize the eight Qwen pairs at the behaviorally safer curvature step
  `epsilon=0.08`.
- Keep the 494M-parameter base model frozen.
- Architectural expansion remains exactly 21,504 parameters.
- Train 43,008 coordinates: gate rows, up rows, and down columns for both members
  of all eight sibling pairs.
- The implementation physically removes these coordinates from the frozen MLP
  and reintroduces them as the trainable adapter, so there is no hidden duplicate
  parameter copy.
- Objective: one epoch on `train_512`, randomly mixing 50% full-solution
  targets and 50% `Final answer: <answer>` targets.
- Optimizer: AdamW, learning rate `3e-4`, no weight decay, microbatch 1,
  gradient accumulation 8, max gradient norm 1, max length 384.
- Adapter checkpoint every 10 optimizer steps.
- All 20 current unit tests pass, including function preservation, separated
  forward equivalence, gradient isolation, exact parameter count, and adapter
  checkpoint round trip.

### Completed run

- Run tag: `qwen_eps008_mixed_localized_v1`.
- Status: complete, 64/64 optimizer steps and 512/512 examples.
- Actual target mix: 258 answer-only and 254 full-solution examples
  (`50.39%` answer-only).
- Total target tokens: 33,266.
- Token-weighted mixed training NLL: `0.53291952`.
- First 16 steps' token-weighted NLL: `0.57929255`.
- Last 16 steps' token-weighted NLL: `0.49172662`.
- The lower late-run value is suggestive of learning but is not a controlled
  before/after comparison because each step contains different examples and a
  different answer/solution mix.
- Pre-clipping gradient norm: minimum `0.6335`, mean `1.1210`, median `0.9785`,
  maximum `3.6664`.
- Gradient norm exceeded the threshold of 1 on 30/64 steps and was clipped; no
  NaN, divergence, or failed step occurred.
- Initial mean sibling distances: gate `0.112414`, up `0.113805`, down `0`.
- Final mean sibling distances: gate `0.204849`, up `0.203684`, down `0.173726`.
- Final maximum sibling distances: gate `0.388704`, up `0.418676`, down
  `0.266375`.
- Parameter displacement norms from initialization: gate rows `0.458915`, up
  rows `0.425036`, down columns `0.566075`.
- Adapter size on disk: 175,346 bytes.
- Actual wall time: 4,051.74 seconds (`67m 31.7s`). Step times were highly
  variable: median `53.2s`, mean `61.8s`, maximum `259.4s`.

Interpretation: optimization was numerically stable and produced real sibling
specialization, including divergence of the outgoing columns that were exactly
identical at initialization. Training loss alone did not establish improvement:
the paired generation result below shows that the learned specialization was
behaviorally harmful.

Artifacts:

- `runs/localized_train_qwen_eps008_mixed_localized_v1.json`
- `runs/localized_train_qwen_eps008_mixed_localized_v1.adapter.pt`
- `runs/localized_train_qwen_eps008_mixed_localized_v1.log`

Completed training command:

```bash
./scripts/run_train_localized_qwen.sh qwen_eps008_mixed_localized_v1 0.08 mixed 0.5
```

Completed paired generation command:

```bash
./scripts/run_split_adapter_generation_qwen.sh \
  qwen_eps008_mixed_localized_v1 \
  runs/localized_train_qwen_eps008_mixed_localized_v1.adapter.pt
```

### Paired generation result: decisive failure

On the same 128-example Qwen GSM8K development validation set:

| Metric | Original | Trained split adapter |
| --- | ---: | ---: |
| Correct | 35/128 | 6/128 |
| Accuracy | 27.34375% | 4.6875% |
| Mean generated tokens | 95.01 | 11.38 |
| Median generated tokens | 110.5 | 7 |
| Outputs containing `Final answer:` | 19/128 | 127/128 |
| Outputs reaching 128-token cap | 53/128 | 1/128 |

- Accuracy delta: `-29/128 = -22.65625` percentage points.
- Paired outcomes: 3 both correct, 32 original-only correct, 3
  adapter-only correct, and 90 both wrong.
- Exact two-sided McNemar p-value: `4.17698e-7`.
- Extracted predictions changed on 124/128 examples; completions changed on all
  128.
- Adapter completions averaged 25.55 characters versus 292.49 for the original.
- 123/128 adapter completions began immediately with `Final answer:`. Of the 32
  baseline-correct answers lost by the adapter, the dominant pattern was a terse,
  unsupported wrong answer.
- The three adapter-only gains were examples `gsm8k:train:3637`,
  `gsm8k:train:7160`, and `gsm8k:train:4358`; they do not offset the 32 losses.
- Adapter generation took 202.27 seconds; total run time with model setup was
  205.89 seconds (`3m 25.9s`). The short time is itself a symptom of the much
  shorter completions.
- This is also far below the untrained `epsilon=0.08` split's descriptive
  `37/128` combined development result. That static result is optimistically
  selected, but it makes clear that localized training—not exact splitting or
  the initial curvature step—introduced the catastrophic change.

Interpretation: the most likely mechanism is objective-induced output-mode
collapse. Although answer-only examples contributed relatively few target tokens,
half of prompts in the mixed objective consistently rewarded `Final answer:` as
the first response tokens. That strong prefix signal overrode the prompt's request
for a short solution. Stable optimization and falling training NLL therefore
measured successful fitting of a harmful target mixture, not improved reasoning.

Decision:

- Retire the v1 `mixed` objective with any nonzero answer-only fraction.
- Do not run the planned matched 67-minute controls for this failed objective;
  they would test a training recipe we no longer intend to claim.
- Do not spend a fresh final GSM8K holdout on this adapter.
- A v2 attempt, if pursued, must train only on complete solution trajectories,
  emphasize the final-answer portion within those trajectories rather than
  replacing the trajectory, use a smaller update, retain step-specific
  checkpoints, and include a small free-running behavioral sentinel capable of
  stopping a run before another full epoch collapses.

### Proposed localized-training v2 safety protocol (not yet run)

This is a development plan, not a result.

1. **Trajectory-preserving target:** every target should contain a concise
   reasoning trajectory followed by `Final answer: <answer>`. Remove answer-only
   examples entirely. If the answer deserves extra emphasis, weight the final
   marker and answer tokens within that same sequence (initial proposal: `2-4x`)
   rather than presenting `Final answer:` as the first assistant tokens. Audit
   tokenization so truncation never removes the final-answer segment.
2. **Smaller trust-region update:** start at learning rate `3e-5`, ten times below
   v1, with a lower gradient-norm cap (initial proposal: `0.5`) and an anchor to
   the curvature-initialized adapter. Calibrate the anchor coefficient in a short
   smoke test rather than prespecifying an unmeasured value. Treat total parameter
   displacement and free-running behavior—not the fact that only 43,008
   coordinates are trainable—as the actual update-size constraints.
3. **Short gate before a full epoch:** train only 16 optimizer steps / 128 examples
   first. Do not authorize the 64-step / 512-example run unless the short run
   passes the behavioral sentinel and improves a held-out teacher-forced solution
   metric. Expected training time is roughly one quarter of v1, around 17 minutes,
   plus sentinel generation time; this remains a user-run foreground job.
4. **Retained checkpoints:** save immutable adapters at steps 0, 4, 8, and 16
   (for example `.step0004.adapter.pt`) plus a manifest. The current trainer saves
   every 10 steps to the same path and therefore overwrites the early states.
   Adapter-only checkpoints suffice for evaluation; optimizer and RNG state should
   also be saved if exact resumption is required.
5. **Free-running sentinel:** use a fixed 16-example slice from the already-spent
   development validation set, preferably stratified as eight baseline-correct
   and eight baseline-wrong examples. At steps 0, 4, 8, and 16, greedily generate
   with the real evaluation prompt and record accuracy, paired gains/losses,
   generated-token median, fraction beginning immediately with `Final answer:`,
   marker presence, and truncation. Cached baseline records avoid loading a second
   reference model.
6. **Prespecified stop rules:** hard-stop on an unmistakable mode collapse such
   as more than 50% of sentinel outputs beginning immediately with
   `Final answer:` or median completion length falling below 25% of its step-0
   value. Also reject a checkpoint if it has at least two more correct-to-wrong
   than wrong-to-correct transitions on the balanced sentinel. These thresholds
   would have caught v1 decisively; they should be frozen before v2 runs.
7. **Escalation rule:** if a checkpoint passes the sentinel, select among safe
   checkpoints using held-out teacher-forced solution NLL, then evaluate only the
   selected checkpoint on all 128 development examples. Extend to 64 steps only
   after that full paired evaluation shows no accuracy regression. Matched
   initialization/location controls and a fresh final holdout remain downstream
   of a successful frozen recipe.

Current implementation gap: `train_localized_split.py` already supports the
solution-only objective and an initialization-anchor penalty, but it does not yet
support answer-segment token weighting, immutable step-numbered checkpoints,
sentinel generation, prespecified early stopping, or exact optimizer/RNG resume.

Artifacts:

- `runs/split_adapter_generation_qwen_eps008_mixed_localized_v1.json`
- `runs/split_adapter_generation_qwen_eps008_mixed_localized_v1.log`

## Recommended clean zero-training demonstration (not yet run)

The preferred next behavioral experiment is a frozen-model BLiMP minimal-pair
test on `SmolLM2-135M-Instruct`, initially reusing layer 15 and the existing split
machinery. BLiMP asks whether the model assigns greater probability to the
grammatical member of each short sentence pair. It removes autoregressive
reasoning, prompt-following, answer extraction, and training-loop confounds while
retaining a discrete paired accuracy endpoint.

Proposed protocol:

1. Stratify BLiMP into disjoint curvature, epsilon-validation, and locked-test
   items across its grammatical phenomena. Preserve a large test set and never
   choose phenomena based on test results.
2. Use pairwise logistic loss on the probability margin
   `log p(grammatical) - log p(ungrammatical)` as the curvature objective. Report
   the benchmark-standard grammatical-preference accuracy, mean margin, pairwise
   loss, and wrong-to-correct versus correct-to-wrong flips.
3. Compute the curvature-selected split once, verify the epsilon-zero duplicate
   is function-preserving, select one finite epsilon on validation, then evaluate
   the locked test exactly once. There are zero optimizer/training steps, though
   curvature estimation and epsilon selection must still be described honestly
   as supervised, data-driven model surgery.
4. Run the existing three matched control families at the same added parameter
   count and perturbation norm. Also include an unsplit matched one-shot weight
   perturbation if a fair version can be prespecified, so the conclusion concerns
   expansion rather than merely task-informed weight editing.
5. Primary success requires lower locked-test pairwise loss and a favorable paired
   confidence interval; the stronger behavioral result requires positive net
   accuracy flips, ideally with a significant exact McNemar test. If only smooth
   loss/margin improves, claim only objective-aligned held-out likelihood gain.

Potential headline, only if supported: curvature-guided function-preserving
expansion of eight frozen-LLM FFN neurons improves held-out grammatical preference
after one symmetric structural displacement, with no optimizer steps. LAMBADA
last-word prediction is a reasonable second test but has longer contexts; general
held-out text perplexity is the cleanest smooth-objective replication but would
not add a discrete behavioral endpoint beyond the NLL evidence already obtained.

## Provisional paper direction: splitting instability of gated neurons

This is a candidate theorem and paper design, not yet a proved or experimentally
validated result. Model size alone cannot supply novelty. Function-preserving
duplication, negative splitting-curvature selection, the optimal symmetric
two-child eigendirection, and additive top-k splitting are already central results
of Splitting Steepest Descent; Net2Net and Network Morphism already cover
function-preserving expansion. A simple perceptron/MLP demonstration would
therefore be an illustration or replication, not the contribution. A recent ICLR
2026 submission, *Expand Neurons, Not Parameters*, also studies neuron splitting
on small symbolic, MLP, CNN, and CLIP tasks, making a generic "splitting helps
small networks" claim especially weak.

The more promising mathematical distinction is the multiplicative structure of a
GLU/SwiGLU neuron. For the simplified gated unit

`h(x; g, u) = phi(g^T x) (u^T x)`, with `theta = (g, u)`, its parameter Hessian is

```text
          [ phi''(g^T x)(u^T x) xx^T    phi'(g^T x) xx^T ]
H_theta = [                                                    ].
          [ phi'(g^T x) xx^T                         0         ]
```

After multiplying by the downstream loss/backpropagation scalar and averaging
over data, the corresponding splitting matrix retains the block form

```text
S = [ A   B  ].
    [ B^T 0  ]
```

This is the same coordinate implemented by `curvature_refine.py`: for token or
example position `t`, let `delta_t` be the loss gradient with respect to the
selected neuron's scalar pre-`down_proj` activation, `a_t = g^T x_t`, and
`b_t = u^T x_t`. With fixed half-outgoing columns and gate/up split directions
`v,w`,

```text
A = (1/T) sum_t delta_t phi''(a_t) b_t x_t x_t^T
B = (1/T) sum_t delta_t phi'(a_t)     x_t x_t^T
```

and the exact symmetric child parameterization
`(g_+,u_+)=(g,u)+epsilon(v,w)`,
`(g_-,u_-)=(g,u)-epsilon(v,w)` gives

```text
L_split(epsilon) - L(0)
  = (epsilon^2 / 2) [v;w]^T S [v;w] + O(epsilon^4),
```

assuming sufficient smoothness. The fourth-order remainder, rather than a
generic third-order remainder, follows because swapping the two children maps
`epsilon` to `-epsilon` without changing the network, so the loss is even in
`epsilon`. The deep network's loss Hessian first contributes at fourth order
because the symmetric split changes the layer output only at second order.

If `B != 0`, this symmetric matrix is indefinite. Choose left/right singular
vectors `p, q` with `p^T B q = s > 0`; on the two-dimensional subspace spanned by
`(p, 0)` and `(0, q)`, the restricted matrix is `[[a, s], [s, 0]]`, whose smaller
eigenvalue is

`(a - sqrt(a^2 + 4 s^2)) / 2 < 0`.

Subject to verifying the exact deep-network splitting-matrix assumptions, this
would imply that multiplicative gated neurons are generically splitting-unstable
whenever their gate/value cross block is nonzero. It could also explain why
`4,774/4,864` Qwen layer-12 neurons had negative screening proxies. A focused
search has not yet found this exact GLU/SwiGLU splitting theorem, but absence from
that search is not proof of novelty; the derivation and literature audit must be
checked rigorously before it becomes a paper claim.

The first formal milestone is therefore a code-aligned conditional theorem:

> For the fixed-half-outgoing symmetric split above, if the cross block `B` is
> nonzero, the splitting matrix has a strictly negative eigenvalue and there is
> an `epsilon_0 > 0` such that a sufficiently small nonzero split lowers the
> empirical loss. The theorem must list the smoothness assumptions, normalization,
> bias treatment, and the degenerate cases `B=0` explicitly.

Only after proving this conditional statement should the stronger word
"generically" be formalized, for example by showing that `B=0` requires a
measure-zero cancellation under specified data/backpropagation assumptions.

Recommended paper structure if the theorem survives:

1. Derive the exact deep-network GLU/SwiGLU splitting matrix and prove the
   generic-indefiniteness result, including all degenerate cases and the effect of
   the outgoing projection.
2. Give a scalable estimator/solver for selecting gated neurons and their
   split directions in pretrained networks.
3. Validate the predicted `epsilon^2` local loss curve with a tiny smooth gated
   network, an ordinary non-gated hidden unit as a comparison, and a pure linear
   perceptron as a negative control. A duplicated linear perceptron remains linear
   and has no useful within-unit second derivative, so it should not be the main
   demonstration.
4. Use the existing SmolLM/Qwen frozen NLL results as realistic-scale validation,
   plus a clean zero-training behavioral test such as the prespecified BLiMP
   minimal-pair experiment. Small models are sufficient because the theorem and
   method—not parameter count—would carry the novelty.
5. Include matched random location/direction controls and a matched unsplit
   one-shot parameter edit. The latter is essential because Splitting Steepest
   Descent notes that an ordinary parameter displacement changes loss at first
   order away from a stationary point, whereas a symmetric split begins at second
   order.

A suitable toy figure would train a one-hidden-unit gated classifier or regressor
on a smooth two-mode 2D problem, freeze it, split once, and plot the measured
loss against the theorem's quadratic prediction with zero post-split optimizer
steps. Baseline training is allowed; the clean claim is that the structural edit
itself uses no post-split training. This figure should explain the theorem, not be
marketed as evidence that neuron splitting was newly invented.

## Mechanistic interpretation retained so far

For an infinitesimal symmetric split, first-order effects cancel. Along the
antisymmetric sibling coordinate,

`delta L ~= 0.5 * curvature * epsilon^2`.

Negative curvature therefore identifies a local direction in which duplicating
and separating a neuron can lower the chosen differentiable loss. This explains
the small-epsilon NLL results. It does not imply monotonic finite-step behavior
or improved greedy accuracy, because greedy exact-answer accuracy is a discrete,
free-running property and the selection objective has been teacher-forced NLL.

## Merge/compression fork and provisional novelty conclusion

A natural inverse operation collapses a sibling pair along its center/difference
coordinate and sums the outgoing columns when their gate/up weights coincide.
For arbitrary neurons, finite collapse needs outgoing-aware fitting and measured
loss checks; local curvature alone is insufficient for a large merge.

The most defensible provisional research direction is **bidirectional
curvature-guided dense FFN capacity reallocation**:

1. Find merge candidates using redundancy signals.
2. Accept merges only when calibrated loss along the collapse path is small and
   splitting curvature near the collapsed state does not favor immediate
   re-separation.
3. Spend the freed dense-neuron capacity on the most negative-curvature splits
   elsewhere.
4. Compare at fixed neuron/parameter count against pruning-only, splitting-only,
   similarity merging, and random merge-split controls.

For Qwen layer 12, one structural SwiGLU neuron contains `3 * 896 = 2,688`
parameters. Merging eight pairs would free exactly the 21,504 parameters used by
the current eight splits, enabling a clean fixed-width experiment.

Prior-art conclusion from the focused audit in `notes.md`:

- Splitting Steepest Descent already uses negative splitting curvature to grow
  networks.
- Neuron Merging, Optimal Neuron Merging, RED/RED++, Taylor/Hessian/Fisher
  pruning, dynamic sparse reparameterization, DOTResize, and CoCurve cover major
  neighboring ingredients.
- No searched work was found combining a reversible dense gated-FFN merge/split
  coordinate, task-loss-based merge acceptance, splitting-curvature stability,
  and fixed-budget reallocation of freed neurons to negative-curvature splits.
- This is provisional, not proof of novelty; repeat the search immediately before
  submission.

## Update log

- 2026-08-29: Recorded a provisional theorem-led paper direction. The Hessian of
  a multiplicative GLU/SwiGLU unit has block form `[[A, B], [B^T, 0]]`, which is
  indefinite whenever `B != 0`; if the exact deep-network splitting assumptions
  check out, gated neurons are generically splitting-unstable. This could be the
  mathematical novelty, with tiny networks as theorem illustrations and
  SmolLM/Qwen/BLiMP as validation. Generic splitting or a perceptron toy alone is
  not novel given Splitting Steepest Descent and related expansion work.
- 2026-08-29: Recorded the decisive negative paired generation result for the
  localized mixed-objective adapter: `35/128 -> 6/128`, 32 losses versus 3 gains,
  exact McNemar `p=4.18e-7`. Output-length diagnostics show an answer-immediately
  mode (`123/128` completions begin with `Final answer:`; median 7 tokens), so the
  mixed answer-only/full-solution objective is retired and no final holdout or
  matched long-run controls will be spent on it.
- 2026-08-29: Recorded completed localized mixed-objective training v1. The run
  finished 64/64 steps without numerical failure, broke sibling symmetry, and
  took 67.5 minutes.
- 2026-08-29: Created this canonical ledger and backfilled all completed SmolLM,
  Qwen, control, generation, retention, cross-domain, data-hygiene, implementation,
  and novelty conclusions available in the workspace and chat.
