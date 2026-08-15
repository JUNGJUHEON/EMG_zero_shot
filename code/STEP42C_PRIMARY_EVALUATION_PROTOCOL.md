# Step42C Primary Evaluation Protocol

Status: locked after the independently verified legal-proxy gate and before any
Step42C primary-query cache, prediction, or truth join.

Date: 2026-07-22

Parent protocol: `STEP42C_BALANCED_TRANSDUCTIVE_PROTOCOL.md` at the hash recorded by
`primary_s1_s10_20260722_v1/config.json`.

## 1. Admission

The legal proxy gate passes every registered check and independent replay verifies:

- five-class selected balanced accuracy `0.5985` versus independent `0.4583`;
- selected-minus-independent `+0.1401`, 95% CI `[+0.0793,+0.2003]`, `p=0.0039`;
- semantic-permutation mean `0.1974` and selected gain `+0.4011`;
- cluster-oracle accuracy `0.9429`;
- held B5/B6 accuracy `0.8889`; and
- all candidate, prediction, control, anchor, exposure, and hash checks.

This permits one two-phase primary evaluation. It does not permit further candidate
selection or hyperparameter tuning.

## 2. Frozen Primary Estimand

- fit: target-subject A1-12+B5/B6 EMG and pose;
- query: target-subject B1/B2/B3/B4/B7 EMG, all six trials per class;
- semantic anchor: `SA_LOSO_prototype_full6` built from s2-s10 minus the target;
- query side information: complete unlabeled query batch, known five-class catalog,
  and exactly six trials per class;
- primary subjects: s2-s10;
- diagnostic only: s1;
- excluded: B8, B9-17, and Exercise C.

Result label: `S0v2_TargetAB56 x SA_LOSO_prototype_full6 x
UQ_balanced_transductive`.

## 3. Physical Query Cache

For each subject, a trusted cache-builder process may read the broad verified Step41
A+B source cache only to create:

1. a model-facing NPZ beside the subject data containing primary-query
   `emg_features` and opaque row/trial IDs;
2. a model-facing manifest containing only `subject`, opaque row/trial IDs, and a
   within-cache row index; and
3. model-facing metadata containing hashes, row/trial counts, and exposure claims.

The model-facing cache must contain no exercise code, gesture class, repetition,
source file, source row index, glove feature, glove window, or other pose value.
Opaque IDs are HMAC-derived with an ephemeral secret that is never written.

The trusted builder writes a separate truth seal under `11. Result`. That seal maps
opaque IDs to the source manifest and is not an input to prediction. It is hashed
before prediction begins.

## 4. Frozen Source Anchors

The trusted builder creates one `(5,66)` source-pose prototype matrix per target in
catalog order `[B1,B2,B3,B4,B7]`:

1. fit each source subject's pose coordinates on equal-group Exercise-A rows only;
2. average source pose windows within repetition;
3. average all six repetitions within source subject;
4. average source subjects; and
5. exclude the target subject, with s1 diagnostic using s2-s10.

Only these frozen prototypes and an anchor audit enter prediction. No target
held-class pose enters either artifact.

## 5. Frozen Subject Policies

Use the exact outer-subject selections from the verified proxy gate:

- s1-s4, s6-s9: `bcn__mav16__posew_0p25`;
- s5: `bcn__amplitude64__posew_0`;
- s10: `bcn__all128__posew_0p25`.

No primary-query statistic may change these policies.

## 6. Phase 1 Prediction

The prediction process may load only:

- physical target legal A1-12+B5/B6 caches;
- physical unlabeled primary-query caches;
- frozen LOSO source prototypes;
- frozen outer-subject selections; and
- the two Step42C model source files.

For every subject it must freeze, before evaluation:

1. selected BCN trial predictions and cluster assignments;
2. independent-pose trial predictions;
3. balanced-pose trial predictions;
4. 32 semantic-permutation selected-BCN predictions;
5. 16 cluster-feature-permutation selected-BCN predictions;
6. predicted-pose matrices, source scores, trial grouping, Ridge state summaries;
7. all input, source, protocol, code, prediction, and artifact hashes.

Phase 1 must be independently replayed without loading the truth seal. Only a passing
Phase 1 verifier permits Phase 2.

## 7. Phase 2 Evaluation

The evaluator may load the frozen prediction seal and truth seal. It may only join,
score, plot, and summarize; it may not fit, normalize, select, cluster, change a
prediction, or write back into Phase 1.

Primary unit-level outputs:

- trial balanced accuracy, macro F1, accuracy, class recall, and confusion matrix;
- window balanced accuracy after propagating each frozen trial decision;
- evaluator-only cluster-oracle accuracy;
- per-subject selected-minus-independent and selected-minus-balanced-pose gains;
- semantic-permutation and cluster-shuffle controls.

Subject s1 is reported separately and excluded from all primary intervals and tests.

## 8. Primary Success Gate

All checks must pass on s2-s10:

1. mean selected trial balanced accuracy is at least `0.50`;
2. mean selected gain over independent pose is at least `+0.05`;
3. its subject-bootstrap 95% lower bound is above zero;
4. exact one-sided sign-flip `p < 0.05` and at least 6/9 subjects improve;
5. mean selected gain over balanced pose is at least `+0.03`;
6. its bootstrap lower bound is above zero, sign-flip `p < 0.05`, and at least 6/9
   subjects improve;
7. mean selected gain over semantic permutations is at least `+0.15`;
8. its bootstrap lower bound is above zero, sign-flip `p < 0.05`, and at least 6/9
   subjects improve;
9. mean evaluator-only cluster-oracle accuracy is at least `0.70`;
10. mean selected gain over cluster-feature permutations is at least `+0.03`;
11. every selected prediction has all five classes and exactly six trials per class;
12. every metric is finite and all cache, anchor, exposure, source, and prediction
    verification checks pass.

The gate is intentionally stronger than merely exceeding chance. A pass supports a
balanced transductive solution under the declared batch/prior assumptions. A failure
does not invalidate the proxy mechanism result but rejects primary transfer.

## 9. Interpretation Boundary

Even a pass cannot be called inductive ZSL, online single-trial recognition, unknown-
prior recognition, or generic domain adaptation. DB5 has been repeatedly inspected,
so the result is internally locked evidence requiring fresh external confirmation.
