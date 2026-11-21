# Initial experiment plan

## Phase 0: frozen baseline

Evaluate the untouched `Qwen2.5-0.5B-Instruct` model on deterministic subsets of
GSM8K and Hendrycks MATH. Each run saves its exact prompts, selected example IDs,
raw completions, parsed answers, correctness, token counts, runtime, seed, and
generation settings.

The initial generative metric uses exact numeric matching for GSM8K and normalized
boxed-answer matching for MATH. A publication run should additionally use a
standard symbolic-equivalence grader for MATH.

## Phase 1: splitting sanity check

Implement an exactly function-preserving split for one Qwen SwiGLU FFN neuron:
duplicate the corresponding `gate_proj` and `up_proj` rows and divide the matching
`down_proj` column between the two copies. Verify logits before and after splitting.

## Phase 2: curvature selection

Estimate the second directional derivative of supervised math loss along symmetric
split directions. Split neurons with the most negative curvature, then compare with
random selection and weight-magnitude selection.

## Phase 3: localized recovery training

Train only the split parameters on OpenMathInstruct examples. Compare all methods
using identical examples, optimizer steps, seeds, and evaluation prompts.

