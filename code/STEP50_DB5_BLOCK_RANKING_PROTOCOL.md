# Step50 DB5 Pose-Block Ranking Protocol

Status: locked before primary prediction

Date: 2026-08-12

## Question

Can a three-parameter pose metric improve cluster naming by maximizing the correct
Hungarian assignment rank on DB5 legal proxy episodes?

## Scope

- NinaPro DB5 s1-s10 only; DB1 is forbidden.
- Fit exposure: target A1-A12+B5/B6.
- Evaluated target classes: B1/B2/B3/B4/B7.
- Other-nine DB5 pose donors, complete five-class query batch, and six trials per
  class remain fixed.
- Step48's truth-blind stable cluster-parent policy is frozen.

## Metric

The pose representation has three 22-coordinate blocks: window mean, window
standard deviation, and late-minus-early change. For each block, Step50 computes a
cluster-by-class cosine score matrix. A nonnegative weight vector
`w = [w_mean, w_std, w_delta]`, with weights summing to one, combines the matrices.

Weights are searched on the simplex at increments of `0.05` (231 candidates). For
every legal proxy episode, all exact class permutations are enumerated. The ranking
loss is the negative log probability of the oracle-optimal legal assignment under
temperature `0.1`, plus `0.01 * ||w - [1/3,1/3,1/3]||^2`.

For outer target `s_i`, weights are selected using only the other nine subjects.
The exact parent comparator is the stable cluster policy with raw all-66 cosine and
mean naming. A learned weight is eligible only when:

- other-nine Exercise-A trial BA improves by at least `0.01`; and
- other-nine B5/B6 trial BA does not decrease.

Among eligible weights, minimize ranking loss, then maximize Exercise-A trial BA,
then use lexical weight order. If none is eligible, retain raw all-66 naming.

## Firewall And Boundary

Phase 1 cannot open primary query caches, truth seals, scored outcomes, or DB1 paths.
It writes a hash-sealed weight table. The predictor writes opaque query predictions
before the evaluator may open truth. Step50 was designed after earlier DB5 results,
so it remains a post hoc DB5-only internal experiment rather than independent
confirmation.
