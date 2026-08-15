# Step44 Exercise-Conditioned Naming Protocol

Status: preregistered before Step44 prediction generation

Protocol date: 2026-07-23

Canonical label:

`development_only_external_candidate_freeze`

## 1. Evidential Status

Step44 is a development experiment, not a new DB5 primary endpoint. The
B1/B2/B3/B4/B7 primary outcomes were consumed by Step42C and remain permanently
closed to model development. Step44 cannot authorize another join to those labels,
regardless of its result.

A Step44 pass may only freeze one model for untouched external subjects, a new
recording session, or a compatible dataset chosen before outcome inspection. A
Step44 failure closes internal semantic-naming development.

The B5/B6 outcomes used here have also been observed in earlier work. They are
therefore development data, not independent confirmation. Subject-level intervals
and tests describe stability within this development cohort; they do not erase
project-level adaptivity.

## 2. Scientific Question

Step43 transferred residuals learned mostly from Exercise-A landmarks onto
complex-B anchors. It improved held-A naming slightly but damaged the B5/B6 guard,
indicating that A-space interpolation does not estimate the conditional
Exercise-A-to-B offset.

Step44 asks:

> Can a seen complex-B gesture provide a fixed exercise-conditioned pose-space
> correction that improves naming of another held complex-B gesture while
> preserving held-A naming?

This is distinct from Step43. It does not fit a class-similarity transport, choose
an alpha, combine scores, bound cluster mass, or learn a fallback threshold.

## 3. Locked Exposure Boundary

Allowed target-subject model fitting:

- Exercise A1-A12, except A classes explicitly held in a development episode;
- exactly one of B5 or B6 as the seen complex-B calibration class;
- EMG and glove/pose from those fit rows only.

Allowed source semantics:

- glove/pose from the registered LOSO source subjects;
- source poses for the episode's held catalog and its seen calibration classes;
- no source held-class EMG is used by the naming mechanism.

Forbidden everywhere in model-facing Step44:

- target B1/B2/B3/B4/B7;
- target or source B8 and B9-B17;
- Exercise C;
- Step42C/42D Phase-2 truth or outcome files;
- Step43 gate outcomes as model inputs;
- target pose or labels for the current held development query.

Every Step44 builder, predictor, and verifier must load only the existing physical
per-subject `step42b0_s0v2_legal_amplitude_A1_12_B5_B6_2p0s_v1` caches and their
hash-bound metadata. These caches contain only A1-A12+B5/B6. No Step44 process may
load a broader all-class cache, MAT file, Step42C primary query cache, Phase-2 truth
seal, or Phase-2 result. A pack verifier may compare the amplitude cache against
the independently verified Step42A A1-A12+B5/B6 cache, but it may not load any
broader cache.

The trusted pack builder writes mixed-proxy truth to a physically separate sealed
truth pack. The evaluator may load only that truth pack plus frozen Step44
predictions. Model packs and their public seals contain no truth-pack path or
filesystem pointer.

S2-S10 are the primary development subjects. S1 is not built or revealed until the
s2-s10 gate, all independent replays, and the final decision are immutable. A later
S1 diagnostic cannot change code, thresholds, interpretation, external freezing,
or reopen a failed mechanism.

## 4. Frozen Parent Representation

For each target subject, use the already frozen Step42C outer-subject selection row
whose `outer_subject` equals that target. This fixes the EMG feature set and
pose-view weight before Step44.

The selected parent must be identical to the Step42C sealed selection and its
B5/B6-free replay established by Step43. Step44 performs no representation,
feature, clustering, or hyperparameter selection.

For every episode:

1. fit the existing equal-group weighted RidgeCV EMG-to-pose map on the legal fit
   rows;
2. fit its EMG and response scalers on those same fit rows only;
3. fit the frozen parent EMG cluster-view scaler on all legal fit rows and construct
   the parent trial representation;
4. run the Step42E count-unconstrained five-cluster solver with the same starts,
   seeds, maximum iterations, tie rules, and one-nonempty-cluster-per-class
   requirement.

Cluster assignments are frozen before any truth is joined.

## 5. Symmetric Mixed Exercise Proxies

Start from each of the 12 registered five-class held-A catalogs. For each catalog:

1. replace each of its five A positions in turn;
2. insert B6 as held query class while B5 is the seen calibration class;
3. repeat with B5 held and B6 seen.

Replacing one A class produces duplicate four-A sets across the 12 parent catalogs.
Canonicalize each episode by `(held_B, sorted(held_A_classes))`, sort
lexicographically, and keep each unique key once. This yields exactly 30 unique
four-A catalogs for each held-B direction and 60 physical episodes per subject.
No duplicate alias or reweighted copy enters a metric.

Each query catalog contains four held A classes and one held B class. Each class
contributes all six repetitions, yielding 30 opaque query trials. The legal fit
catalog contains the other eight A classes and the one seen B calibration class.

The held B EMG and target pose are physically absent from fitting. The predictor
may know the five-class semantic catalog and which catalog member is Exercise B;
this is declared side information.

The model-facing catalog is always five opaque semantic slots `0..4`, not raw class
integers. This prevents collisions such as A5 and B5. The model pack contains only
the slot exercise roles. The physically separate trusted audit maps each slot to an
exercise-qualified identity such as `A05` or `B05`.

## 6. Locked Landmark And Coordinate Construction

For an episode, let `fit` contain the eight non-held A classes and the one seen B
class. The target coordinate scaler is a `StandardScaler` fit with equal-group
weights on target glove features from `fit & exercise=A`, using all six legal
repetitions. It defines one common 66-dimensional coordinate system for that
episode; it is not refit separately by landmark because its purpose is coordinate
definition.

The full query predictor is the existing equal-group weighted RidgeCV with the
registered 13 alphas. Its EMG scaler, response scaler, Ridge coefficients, and
alpha selection use `fit` only. The selected full-fit alpha is frozen for the six
cross-fit landmark folds. Its raw-pose query predictions are transformed by the
episode target coordinate scaler and then averaged:

1. windows within `(exercise, class, repetition)`;
2. one resulting vector per opaque query trial.

Target seen-class landmarks are repetition-cross-fitted. For held repetition
`r`, the fold trains on `fit & repetition != r` and evaluates only
`fit & repetition == r`. The fold refits the EMG scaler, response scaler, and
fixed-alpha Ridge coefficients from scratch using the full-fit selected alpha. It
does not rerun alpha selection. The raw-pose prediction is transformed by the fixed
episode target coordinate scaler. A class landmark first averages windows within
each repetition and then averages the six repetition prototypes. No in-repetition
row enters that repetition's Ridge or model scalers.

For each LOSO source subject, fit a source coordinate `StandardScaler` with
equal-group weights on that source's eight episode-fit A classes, using all six
repetitions. Transform source glove features into that subject's coordinates.
For each semantic class, average windows within repetition, then average the six
repetitions. Finally average class prototypes equally across LOSO source subjects.
The same ordering constructs source seen landmarks, held anchors, the opposite-B
prototype used by the wrong-correspondence control, and every source-jackknife
variant.

Target and source coordinates are compatible by construction only in the weak
standardized-coordinate sense already used by Step42C/43. Step44's residual is the
registered mechanism being tested; no target-query pose is used to improve that
compatibility.

The pack config freezes SHA-256 hashes of this protocol, the cache loaders, the
parent-selection source, the Ridge/scaler source, the unconstrained clustering
source, the Step44 algorithm, and the builder. Any hash change requires a new run
name and a fresh pre-prediction audit.

## 7. Fixed Naming Rule

All pose vectors use the episode's existing standardized 66-dimensional pose
coordinates.

Let:

- `t_b` be the target subject's repetition-cross-fitted predicted-pose prototype
  for the one seen B calibration class;
- `s_b` be the matching LOSO source-pose prototype for that class;
- `a_h` be a LOSO source-pose anchor for held class `h`;
- `q_i` be the target predicted-pose vector for opaque query trial `i`.

The only Step44 candidate is:

```text
delta_B = t_b - s_b

anchor'_h = a_h             if h is Exercise A
anchor'_h = a_h + delta_B   if h is Exercise B

score(i, h) = cosine(q_i, anchor'_h)
```

Cosine scoring performs row normalization after the coordinate-space residual has
been applied. The implementation may express that normalization algebraically
differently only if unit tests prove numerical equivalence.

Within each frozen cluster, average the trial-to-anchor scores. Exhaustively choose
the one-to-one cluster-to-catalog permutation with maximum summed score, using the
existing lexicographic tie rule.

There is no candidate grid. No alpha, fusion weight, learned confidence threshold,
subject-specific switch, or outcome-dependent fallback exists.

For a catalog with no seen B calibration landmark, the implementation must return
`direct_raw` exactly. This fallback is structural, not learned.

For future external S0-v2 inference, where both B5 and B6 are seen, `delta_B` is the
unweighted mean of their two matched target-minus-source residuals. This external
rule is frozen now and cannot be revised after Step44.

## 8. Locked Comparators

Score every frozen episode with:

- `direct_raw`: unchanged source anchors;
- `a_offset_null`: apply the mean matched seen-A residual to the held B anchor;
- `exercise_b_offset`: the fixed Step44 rule;
- `offset_sign_reverse`: apply `-delta_B` to the held B anchor.

Only `exercise_b_offset` is eligible for external freezing. The other rows are
mechanism controls, not selectable candidates.

Step42C exact BCN, Step42D count-unconstrained BCN, and Step43 `transport_a1`
remain historical locked comparators and are not recomputed on incompatible mixed
episodes.

## 9. Physical Two-Phase Firewall

### Phase 1: prediction

The model-facing pack contains only:

- opaque trial aliases and panel membership;
- the frozen parent representation;
- target fit-side cross-fitted landmarks;
- LOSO source semantic landmarks and held anchors;
- the declared exercise role of each semantic anchor;
- content hashes and protocol metadata.

It contains no query truth, repetition, target query pose, cache path, truth-pack
path, or pointer to a trusted audit file. Phase 1 freezes cluster assignments,
raw/control/Step44 scores, one-to-one mappings, source-jackknife mappings, and
predictions.

### Phase 2: evaluation

Only after Phase 1 and its independent replay are sealed may an evaluator join the
separately sealed mixed-proxy development truth pack. The evaluator cannot call
clustering, naming, Ridge, or score-generation code.

No gate outcome grants access to the consumed B1/B2/B3/B4/B7 primary cache.

## 10. Deterministic Controls

Use fixed SHA-256-derived seeds and freeze all control lists before prediction:

1. all 23 nonidentity permutations of the four A semantic columns while the one B
   column and its exercise role remain fixed;
2. the wrong-B-correspondence control, pairing the target seen-B landmark with the
   source prototype of the opposite B5/B6 class;
3. the `a_offset_null` comparator;
4. the `offset_sign_reverse` comparator;
5. eight nonzero cyclic query-feature row shuffles for every one of the 60 unique
   full-query episodes; offsets are
   `1 + stable_seed("step44_feature_shift", episode_id, shuffle_index) %
   (30 - 1)`, and the matched unshuffled baseline is those same 60 episodes;
6. source-subject jackknife naming, omitting each LOSO source subject in turn;
7. evaluator-only balanced cluster oracle;
8. exact no-seen-B fallback replay against `direct_raw`.

Controls cannot select or alter the Step44 rule.

For each feature shuffle, cyclically permute the 30 rows of the frozen parent
cluster representation, rerun the registered count-unconstrained solver with the
unshuffled episode's seed, and use the resulting assignment vector directly
against the original alias-ordered Step44 naming scores. Do not unpermute the
assignment vector. This breaks the EMG-cluster-to-semantic-row relation while
leaving the naming computation and truth order fixed.

For the role-preserving semantic control, permute source-anchor identities and
the final score columns among A classes after constructing the B-offset anchor,
while keeping catalog positions and truth fixed. Do not permute the exercise-role
vector or recompute the offset. The B anchor, B identity, and B role never move. A
separate full five-column
role-destruction control may be reported descriptively but cannot satisfy a gate.

For the no-seen-B fallback, remove every B row from the target/source landmark
arrays of the same frozen mixed episode, without refitting or changing its query,
anchors, cluster view, seed, or assignments. The resulting Step44 anchors, floating
scores, mappings, and predictions must reproduce `direct_raw` exactly.

## 11. Metrics And Inference

The primary development estimand is the s2-s10 subject-weighted mean difference in
trial balanced accuracy between `exercise_b_offset` and `direct_raw` on the 60
mixed full-query episodes, after averaging all episodes within subject.

Also report:

- macro F1 and ordinary accuracy;
- held-B recall and held-A balanced recall;
- B5-held and B6-held directional results;
- per-subject and per-catalog results;
- cluster oracle and oracle-gap fraction closed;
- naming margin and mapping-change frequency;
- wrong-correspondence, A-offset, sign-reversal, semantic, and shuffle controls;
- source-jackknife mapping stability;
- catalog and prediction coverage.

Every episode receives equal weight within subject. The 30 B5-held and 30 B6-held
episodes receive equal directional weight. Held-B recall is the recall of the one B
class. Held-A balanced recall is the unweighted mean recall of the four A classes.
For a multi-draw control, average draws within episode, episodes within subject,
and then subjects; every inferential contrast is paired at subject level before
cohort aggregation.

Control gains in checks 7-11 are `mean_subject(Step44 metric) -
mean_subject(control metric)`. For semantic permutations, average the 23
role-preserving permutations within episode before the subject mean. For feature
shuffles, average the eight shifts within episode before the subject mean. Wrong-B,
A-offset, and sign-reversal controls each contribute one value per episode.
Directional gains are computed separately by averaging the 30 episodes of that
held-B direction within subject and then averaging the nine subjects.

The aggregate oracle-gap fraction is:

```text
(mean_subject(step44_BA) - mean_subject(raw_BA)) /
(mean_subject(balanced_cluster_oracle_BA) - mean_subject(raw_BA))
```

A nonfinite or nonpositive denominator fails the gate.

The inferential unit is the subject, not the episode or panel. For each registered
paired contrast, use 20,000 paired subject-bootstrap draws: sample nine subject
indices with replacement, compute the mean paired difference, and use NumPy's
default linear 2.5th and 97.5th percentiles for a 95% interval. Seeds are:

- `44100`: Step44 minus raw;
- `44101`: held-A Step44 minus raw;
- `44102`: Step44 minus role-preserving semantic permutations;
- `44103`: Step44 minus A-offset null;
- `44104`: Step44 minus wrong-B correspondence;
- `44105`: Step44 minus sign reversal.

Exact one-sided sign-flip tests enumerate all `2^9` sign vectors for the Step44
minus raw subject differences. The p-value is the fraction whose mean is at least
the observed mean minus `1e-15`; zero differences remain and ties count in the
tail. Positive-subject counts use strict differences greater than zero.

Source-jackknife stability is the number of exact full-source cluster-to-class
mapping matches divided by all `(primary subject, episode, omitted LOSO source
subject)` states. Every primary target has eight LOSO sources, so the denominator
is `9 x 60 x 8 = 4,320`.

## 12. External-Freeze Gate

Step44 freezes `exercise_b_offset` for untouched external confirmation only if all
checks pass:

1. all exposure, physical-cache, hash, nesting, catalog, frozen-parent, and
   independent replay checks pass;
2. mean mixed-proxy balanced accuracy is at least `0.60`;
3. gain over `direct_raw` is at least `+0.05`, its subject-bootstrap lower bound is
   above zero, exact one-sided sign-flip `p < 0.05`, and at least 7/9 subjects
   improve;
4. aggregate raw-to-balanced-cluster-oracle gap closure is at least `0.20`;
5. mean held-B recall gain is at least `+0.10`, and the gain is nonnegative in both
   B5-held and B6-held directions;
6. the held-A recall difference has a subject-bootstrap lower bound above `-0.025`;
7. gain over `a_offset_null` is at least `+0.03`, with at least 6/9 subjects
   positive;
8. gain over wrong B correspondence is at least `+0.10`;
9. gain over offset-sign reversal is at least `+0.10`;
10. gain over semantic permutations is at least `+0.15`, with a positive
    bootstrap lower bound and at least 7/9 subjects positive;
11. gain of the matched 60-episode unshuffled baseline over feature-row shuffles
    is at least
    `+0.05`;
12. at least 75% of source-subject jackknife episode mappings equal the full-source
    mapping;
13. no-seen-B fallback predictions and floating scores reproduce `direct_raw`
    exactly;
14. all values are finite and every episode preserves one-to-one catalog coverage.

Thresholds cannot be weakened. Failure of any check rejects Step44 and closes
internal semantic-naming development. No follow-up alpha, fusion, threshold, or
fallback experiment is allowed.

## 13. Strongest Permitted Statement

If the external-freeze gate passes:

> On adaptively reused DB5 development proxies, a fixed seen-complex-gesture
> residual produces a sufficiently stable candidate to freeze for untouched
> external confirmation.

If it fails:

> Neither bounded mass, deterministic seen-landmark transport, nor a fixed
> exercise-conditioned residual produced a stable legal development candidate.
> Step42C/42D remain the supported internal transductive results.

Neither statement is external confirmation, inductive ZSL, online recognition,
open-set recognition, or a new DB5 primary result.

## 14. Methodological Basis

The mechanism targets conditional visual-semantic prototype mismatch rather than a
global coordinate transform. It is consistent with the prototype-shift problem
formalized in work on evolving semantic prototypes and with transductive
visual-structure alignment, but the fixed residual and mixed Exercise-A/B proxy are
specific to this experiment.

The protocol is intentionally stricter than ordinary development practice because
the DB5 subjects and legal proxy family have already informed multiple prior model
decisions.
