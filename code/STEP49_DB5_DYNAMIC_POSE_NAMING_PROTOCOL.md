# Step49 DB5 Dynamic Pose Naming Protocol

Status: locked before primary prediction

Date: 2026-08-12

## Question

Can the frozen DB5 balanced-clustering pipeline name unseen Exercise-B clusters
more accurately when semantic similarity emphasizes movement dynamics rather than
the full 66-dimensional pose summary?

## Scope And Exposure

- NinaPro DB5 s1-s10 only; DB1 is forbidden.
- Target fit exposure is A1-A12+B5/B6.
- Evaluated target classes are B1/B2/B3/B4/B7.
- Every target uses the other nine DB5 subjects as pose donors.
- The complete five-class query batch and six-trials-per-class prior remain fixed.

The 66 pose coordinates are ordered as 22 window means, 22 window standard
deviations, and 22 late-minus-early changes. Step49 compares only:

1. `all66`: all three blocks;
2. `dynamic44`: standard-deviation and late-minus-early blocks.

Each pose view uses either trial-score mean or median naming. Clustering is not
trained or changed by the pose-view comparison.

## Frozen Parent Policies

Two cluster-parent policies are legal candidates:

1. `original`: the fully symmetric Step42C other-nine selection;
2. `stable`: the truth-blind Step48 other-nine stability selection.

The baseline is `original + all66 + mean`. For outer target `s_i`, candidate
selection uses only legal proxy outcomes from the other nine subjects. An
alternative replaces the baseline only when:

- Exercise-A trial BA improves by at least `0.01`; and
- B5/B6 trial BA is not below the baseline, up to numerical tolerance.

Eligible candidates are ranked by mean Exercise-A trial BA, lower subject SD,
Exercise-A window BA, and lexical candidate ID.

## Firewall And Status

The legal-proxy screen cannot open primary query caches, truth seals, scored
outcomes, or DB1 paths. It writes a hash-sealed outer-target selection. The predictor
then writes opaque query predictions before a separate evaluator may open truth.

Because the DB5 endpoint has already been inspected, this remains a post hoc
DB5-only internal improvement experiment and cannot be described as independent
confirmation.
