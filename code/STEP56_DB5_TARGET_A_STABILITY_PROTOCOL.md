# Step56 DB5 Target-A Stability Protocol

Status: locked before primary prediction

Date: 2026-08-12

## Question

Can target-subject whole-class A cross-validation identify a more stable balanced
cluster-and-name representation for difficult targets while preserving legal B5/B6
calibration performance and the frozen assignment-margin gate?

## Scope

- NinaPro DB5 s1-s10 only; DB1, Exercise C, B8, and B9-B17 are forbidden.
- Target fitting and model selection use A1-A12+B5/B6 only.
- Evaluated B1/B2/B3/B4/B7 labels and query glove are forbidden before Phase 3.
- Other-nine pose semantics and each target's frozen Step52 gate are reused.

## Candidate Family

The fixed family contains 27 candidates:

- cluster EMG view: `MAV16`, `amplitude64`, or `all128`;
- predicted-pose cluster weight: `0`, `0.25`, or `1`;
- within-cluster pose-score aggregation: mean, median, or one-per-tail trimmed mean.

For every candidate, both raw 66D and dynamic 44D naming matrices use the same
aggregation rule. The target's frozen Step52 assignment-margin gate chooses between
the two views.

## Legal Selection Rule

Evaluate each candidate on 12 fixed five-class whole-A holdout episodes and one
B5/B6 episode. For A episodes compute:

`stability = mean trial BA - 0.5 * across-episode SD`.

The parent is the frozen Step48 cluster candidate with mean aggregation and the
frozen Step52 gate. An alternative is eligible only if:

1. mean A trial BA is no more than `0.005` below the parent;
2. B5/B6 trial BA is at least `0.80`;
3. stability exceeds the parent by at least `0.02`.

Rank eligible alternatives by stability, mean A trial BA, mean A window BA, B5/B6
BA, then lexical ID. If none is eligible, retain the parent.

## Firewall

Phase 1 may read only legal target A1-A12+B5/B6 caches, other-nine legal source
poses, Step48 parent selections, and Step52 legal gates. It may not read primary
query caches, primary predictions, truth seals, endpoint metrics, or DB1 paths.

Phase 2 verifies the Phase 1 lock, predicts from target evaluated-query EMG only,
and hash-locks predictions before outcome join. Phase 3 alone reads truth.

## Promotion Rule

Promotion over the Step52 parent requires mean trial BA gain at least `+0.01`, a
subject-bootstrap 95% interval lower bound above zero, no subject loss larger than
`0.10`, and at least as many improved as worsened subjects. s4 is reported
explicitly but cannot determine promotion alone. Results remain post hoc internal
DB5 evidence.
