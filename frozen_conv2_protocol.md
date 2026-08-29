# Frozen Conv2 Conditional-Flow Test

Frozen: `2026-08-25`, before any `conv2` deletion labels were generated or
inspected.

## Question

Does branch-local marginal flow add out-of-sample catastrophic-deletion signal
beyond gate-level Group Fisher?

This is a structural holdout at the same model checkpoint and on the same
development split. It is not an independent-model replication. The previously
inspected 120 `conv1` deletions are discovery data; all 120 size-16 `conv2`
groups are the untouched holdout.

## Frozen inputs

- checkpoint SHA-256:
  `6f871d7888db0fff3a517d8eae7fb05049afca8617bfbdfd1bf8da47ecea46fc`
- discovery table SHA-256:
  `0f448e1029ebacc5d49825d99a2ae44ab9af41354b43c79e86c349290afc088c`
- calibration: the first eight deterministic fit-loader batches, 1,024
  examples;
- intervention set: every contiguous size-16 group at each of the eight
  residual-branch `bn2` gates; and
- outcome set: the fixed 5,000-example selection split.

## Scores and fixed hybrid

`F` is gate-level Group Fisher. `Q(F)` is its tie-aware global percentile rank
over the 120 holdout groups. `Q(V)` is the corresponding percentile rank of
branch-local marginal-flow damage. No outcome labels enter either transform.

The only hybrid tested is fixed before label generation:

```text
H = 0.5 * Q(F) + 0.5 * Q(V)
```

No coefficient, layer weight, threshold, or score definition will be selected
using the holdout damage labels.

## Endpoints and decision

The primary label is an immediate accuracy drop of at least `0.02`. The primary
endpoint is paired `AUROC(H) - AUROC(F)`, with a stratified paired-bootstrap 95%
interval using 5,000 replicates.

The conditional-flow test passes only if:

1. the holdout contains at least five positive and five negative deletions;
2. observed AUROC improvement is at least `+0.05`; and
3. the paired interval's lower bound is above zero.

If the class-count requirement fails, the tail endpoint is inconclusive rather
than a pass or failure. Prespecified secondary results are paired average
precision, rank needed to protect every positive, layer-macro Spearman against
loss damage, and the within-layer shuffle null. They cannot override a failed
primary endpoint.

## Execution and recoverability

Deletion evaluation may be split by layers without changing scores, examples,
or endpoints. Each shard is written separately and the canonical table is
assembled by the existing exact-key cache, so completed interventions are not
repeated. Taylor may be reported as a diagnostic but is not part of the frozen
conditional claim.

The final table must contain exactly these group counts:

```text
layer1.0.conv2: 4    layer1.1.conv2: 4
layer2.0.conv2: 8    layer2.1.conv2: 8
layer3.0.conv2: 16   layer3.1.conv2: 16
layer4.0.conv2: 32   layer4.1.conv2: 32
```

The locked analyzer is `code/scripts/test_conditional_flow.py`. It refuses
partial tables and non-`conv2` group layouts.

## Outcome

The locked test was executed once after all four recoverable shards were
complete. The canonical holdout contained 120 groups, eight positive deletions,
and 112 negatives, so the tail endpoint was feasible.

| Score | 2-point-drop AUROC | Average precision | Rank covering all 8 |
|---|---:|---:|---:|
| Fisher-flow hybrid | 0.9922 | 0.9127 | 12 |
| Group Fisher | 0.9788 | 0.7535 | 19 |
| Marginal flow | 0.9632 | 0.6324 | 20 |
| Magnitude diagnostic | 0.9888 | 0.8614 | 11 |

The primary hybrid-minus-Fisher AUROC difference was `+0.0134`, with paired
95% interval `[-0.0089, +0.0446]`. It did not reach the frozen `+0.05` effect
gate and its interval crossed zero. The average-precision difference was
`+0.1592 [-0.0952, +0.4044]`. Therefore `primary_pass = false`.

The protection-rank improvement is a secondary hint, not evidence that can
override the failed endpoint. All eight catastrophic groups occurred in
layers 1 and 2. Marginal flow did not significantly outperform its
within-layer shuffle null (`p=0.0854`), and its layer-macro Spearman was only
0.038. This indicates that most of its high pooled AUROC came from layer
allocation rather than useful within-layer group selection.

Canonical artifact SHA-256 values:

- deletion table:
  `de7adc946911bb57b812a68009907df2fc5446bbf74c97872abcc0af0dd429f0`
- strong-baseline table:
  `9cf3445b41204d0ad6d0644be447e2e7fe676010f6db5a5859939d4b82a49b08`
