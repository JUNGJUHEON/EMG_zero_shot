# Step48 DB5 Stable Naming Protocol

Status: locked before the Step48 primary prediction run

Date: 2026-08-12

## Question

Can the fully symmetric DB5 balanced-clustering pipeline recover part of its
cluster-to-semantic naming gap without using DB1 or evaluated-class truth for model
selection?

## Dataset Scope

- NinaPro DB5 only, subjects s1-s10.
- Fit exposure: target Exercise A1-A12 and B5/B6.
- Evaluated classes: target B1/B2/B3/B4/B7.
- Every target uses the other nine DB5 subjects as pose-semantic donors.
- DB1 is forbidden for fitting, selection, diagnosis, validation, or confirmation.

## Frozen Parent

- Current amplitude-preserving preprocessing and handcrafted feature cache.
- Weighted RidgeCV EMG-to-pose bridge.
- Five-class complete unlabeled query batch.
- Six trials per class.
- Exact-capacity balanced k-means and Hungarian one-to-one assignment.
- Symmetric other-nine source pose prototypes.

The cluster candidates are the frozen Step42C BCN grid: `mav16`, `amplitude64`, or
`all128`, each with pose-view weight `0`, `0.25`, or `1`. Independent-pose and
balanced-pose are retained as controls rather than selectable Step48 parents.

## Legal Naming Screen

The only naming candidates are:

1. `mean`: arithmetic mean of trial-to-prototype cosine scores in each cluster;
2. `median`: coordinate-wise median of those scores; and
3. `trim1`: remove the smallest and largest of the six scores per candidate and
   average the middle four.

Selection for target `s_i` uses only the other nine subjects' twelve balanced
Exercise-A proxy episodes and B5/B6 guard. The current symmetric Step42C selection
with `mean` naming is the no-regret parent for that target. An alternative can
replace it only when:

- its other-nine mean Exercise-A trial balanced accuracy is no more than `0.005`
  below the parent;
- its other-nine B5/B6 trial balanced accuracy is no more than `0.03` below the
  parent.

Among eligible alternatives, maximize
`mean Exercise-A trial BA - 0.5 * subject-level SD`, then mean Exercise-A trial BA,
then Exercise-A window BA, then lexical candidate ID. If no alternative exceeds the
parent's stability score, retain the parent.

## Firewall

Phase 1 must not open the Step42C primary query cache, truth seal, scored outcomes,
or any DB1 path. It writes a hash-sealed naming selection. Phase 2 prediction may
open only the legal DB5 fit caches, the opaque DB5 query cache, symmetric source
pose prototypes, the protocol, and the frozen selection. The evaluator opens truth
only after predictions are written.

## Evidential Boundary

The DB5 B1/B2/B3/B4/B7 endpoint has been inspected in earlier experiments. Step48 is
therefore a post hoc internal improvement experiment even though its implementation
prevents endpoint truth from entering candidate selection. It is not independent
confirmation.
