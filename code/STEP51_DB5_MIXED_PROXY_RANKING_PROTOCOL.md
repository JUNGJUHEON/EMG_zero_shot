# Step51 DB5 Mixed-Proxy Ranking Protocol

Status: locked before primary prediction

Date: 2026-08-12

## Question

Can pose-block weights selected on five-class mixed Exercise-A/Exercise-B proxy
episodes improve semantic naming of unseen Exercise-B clusters?

## Scope

- NinaPro DB5 s1-s10 only; DB1 is forbidden.
- Final fit exposure remains target A1-A12+B5/B6.
- Final evaluated classes remain B1/B2/B3/B4/B7.
- Every target and every legal proxy uses the other nine DB5 subjects as pose donors.
- The Step48 stable cluster-parent policy is frozen.

## Mixed Legal Proxies

Each of 60 proxy episodes contains four held Exercise-A classes and one held
Exercise-B class. When B5 is held, B6 remains in fit; when B6 is held, B5 remains in
fit. Each held class contributes all six repetitions, producing the same five-class,
30-trial, exact-capacity assignment structure as the final task.

No B1/B2/B3/B4/B7 row is used. The mixed proxies contain only A1-A12 and B5/B6.

## Ranking Metric

Step51 uses the three 22-coordinate pose blocks: mean, standard deviation, and
late-minus-early change. It searches 231 nonnegative weight triples on a `0.05`
simplex. The objective enumerates all 120 Hungarian permutations and minimizes the
correct legal-assignment negative log probability at temperature `0.1`, with an
equal-weight shrinkage penalty `0.01`.

For outer target `s_i`, only the other nine subjects select weights. A learned weight
is eligible only when, relative to stable-parent raw all-66 naming:

- held-B recall improves by at least `0.03`; and
- overall five-class mixed-proxy BA decreases by no more than `0.01`.

Among eligible weights, minimize ranking loss, then maximize held-B recall and
overall BA. If none is eligible, retain raw all-66 naming.

## Firewall And Boundary

Phase 1 cannot open primary query caches, truth seals, scored endpoint outcomes, or
DB1 paths. The selected weights are hash-sealed before primary prediction. The
predictor writes opaque endpoint predictions before a separate evaluator opens
truth. The result remains post hoc DB5-only internal evidence.
