# Pose-Assisted Unseen-Gesture Recognition on NinaPro DB5

**Can a limited library of hand movements support recognition of gestures that
were never labeled in a target user's calibration data?**

An independent undergraduate research project by **Ju Heon Jung**, investigating
generalization between movement sets using paired surface electromyography
(sEMG) and glove measurements. Calibration uses 12 isolated finger movements and
two coordinated movements, finger spreading and fist closure. Evaluation uses
five other gestures.

The central design separates two questions: **which EMG trials belong together,
and what movement does each group represent?** EMG features supply grouping
information; an EMG-to-pose regression and other-participant pose prototypes
supply movement identities. This makes the distinction between signal
separability and meaningful cross-modal correspondence explicit.

**Observed endpoint:** 262/300 trials correct; mean participant balanced accuracy
**87.33%**, across ten DB5 participants. This is a **retrospective internal,
full-batch transductive result**, with known class counts and pose side information,
not an untouched-subject or online recognition benchmark.

[Research logic](#research-logic) · [Protocol](#study-at-a-glance) ·
[Method](#pipeline) · [Results](#internal-development-results) ·
[Verification](#verification) · [Open questions](#what-remains-to-be-tested)

## Research logic

A classifier trained only to output the calibration labels cannot directly name
an excluded class. A pose descriptor offers a different target: the model can
predict continuous hand-state features even when the gesture label was absent
from calibration. But the existence of a pose prediction does not guarantee that
it transfers correctly. The pipeline therefore keeps representation, grouping,
and naming as inspectable operations.

| Research question | Design choice | What the evidence can establish |
|---|---|---|
| Can the task require transfer rather than recall of registered gesture labels? | Exclude all target B1/B2/B3/B4/B7 examples from fitting; retain A1-A12+B5/B6. | A movement-disjoint target calibration task, conditional on the permitted pose references. |
| How can an excluded gesture acquire a name? | Regress EMG onto continuous glove summaries; compare predictions with other-nine class prototypes. | A specified cross-modal inference route whose predictions and assignments can be inspected separately. |
| Can grouping be distinguished from assigning meaning? | Construct anonymous EMG/pose clusters before one-to-one semantic assignment. | Two distinct, inspectable failure locations; classification alone does not prove correct pose recovery. |
| How can settings be chosen without using the final query labels during execution? | Whole-class A holdouts, a B5/B6 guard, and a fixed candidate-selection rule. | Label separation in the final selection/prediction run, not freedom from earlier development exposure. |
| Can the reported result be checked? | Save selections and predictions before scoring; use hashes and a separate replay verifier. | Integrity and numerical reproducibility of the stored run, not external validation. |

The implementation uses handcrafted features, weighted RidgeCV, balanced
clustering, and Hungarian assignment. It is **not the EMBridge MAE/Q-Former
architecture**. A transparent regression baseline remains useful here because
each stage has a defined input, objective, and failure mode.

## Study at a glance

| Item | Setting |
|---|---|
| Dataset | NinaPro DB5 only |
| Participants | `s1`-`s10`, each used once as the target |
| Target calibration | `A1`-`A12` and `B5`/`B6`, EMG and glove permitted |
| Unseen target query | `B1`, `B2`, `B3`, `B4`, `B7`; target EMG only at inference |
| Semantic reference | Pose prototypes from the other nine DB5 participants |
| Query batch | 30 trials: five known catalog classes, six trials per class |
| Evaluation mode | Closed-catalog, full-batch, exact-capacity, transductive |
| Primary metric | Mean subject-level trial balanced accuracy |
| Explicit exclusions | DB1, Exercise C, B8, and B9-B17 |

The calibration library can be called **Basic14**: 12 isolated movements plus
two coordinated movements. B5/B6 remain Exercise-B movements in the dataset;
they are not relabeled as isolated Exercise-A movements. **Complex5** denotes
the five evaluation gestures, not a measured physiological complexity scale.

![Symmetric target calibration, selection, and evaluation boundary](assets/study_design.png)

For each target fold, the current target is excluded from the semantic donor bank.
The other nine participants provide pose descriptors for the evaluated classes.
The target provides paired EMG/glove during calibration, but **query inference
does not use that target's query glove or gesture labels**. Query labels are
joined only after predictions have been saved and hash-locked; query glove is
not required to score gesture recognition.

For example, when s4 is the target, its A1-A12+B5/B6 pairs fit the bridge, and
s1-s3 plus s5-s10 supply the pose reference bank. The same role assignment is
applied to every participant. This is leave-one-participant-out **for the donor
bank**, not a model trained without any target-person calibration.

## Gesture protocol

The movement names follow the NinaPro movement taxonomy.

<details>
<summary>Movement-by-movement calibration and evaluation inventory</summary>

### Calibration movements

| ID | Movement | ID | Movement |
|---|---|---|---|
| A1 | Index-finger flexion | A2 | Index-finger extension |
| A3 | Middle-finger flexion | A4 | Middle-finger extension |
| A5 | Ring-finger flexion | A6 | Ring-finger extension |
| A7 | Little-finger flexion | A8 | Little-finger extension |
| A9 | Thumb adduction | A10 | Thumb abduction |
| A11 | Thumb flexion | A12 | Thumb extension |
| B5 | Abduction of all fingers | B6 | Fingers flexed together in a fist |

### Unseen evaluation movements

| ID | Movement |
|---|---|
| B1 | Thumb up |
| B2 | Index and middle fingers extended; other fingers flexed |
| B3 | Ring and little fingers flexed; other fingers extended |
| B4 | Thumb opposition to the base of the little finger |
| B7 | Pointing with the index finger |

This A-to-B split is an operational semantic design. It should not be interpreted
as a universal physiological ordering in which every Exercise-B movement is more
complex than every Exercise-A movement.

</details>

## Pipeline

![Target-A stability-selected transductive pipeline](assets/pipeline.png)

1. **Windowing and features.** Each trial is represented with non-overlapping
   400-sample windows. The input contains 16 EMG channels and 22 glove channels.
   The pipeline extracts 128 handcrafted EMG features and a 66-dimensional pose
   summary.
2. **EMG-to-pose bridge.** Group-balanced scaling and RidgeCV map calibration EMG
   features to pose features.
3. **Candidate family.** The fixed 27-candidate grid combines three EMG views
   (`MAV16`, `amplitude64`, `all128`), three predicted-pose weights (`0`,
   `0.25`, `1`), and three within-cluster naming aggregations (mean, median,
   trimmed mean).
4. **Legal target-specific selection.** Twelve whole-class Exercise-A holdout
   episodes score each candidate using `mean BA - 0.5 x SD`. Alternatives must be
   non-inferior on mean A performance, pass the B5/B6 guard, and improve the
   stability score. No B1/B2/B3/B4/B7 target label is used in this step.
5. **Exact-capacity clustering.** The complete 30-trial query batch is partitioned
   into five anonymous clusters with six trials per cluster using deterministic
   balanced clustering.
6. **Semantic naming.** Cluster-level predicted-pose similarities are compared
   with other-nine pose prototypes. Hungarian assignment maps the five anonymous
   clusters one-to-one to B1/B2/B3/B4/B7.
7. **Sealed evaluation.** Candidate choices, model state, and predictions are
   hashed before target truth is loaded. Evaluation is then independently replayed.

### What each representation contains

The 128D EMG vector contains eight summaries per channel: mean absolute value,
root mean square, standard deviation, peak-to-peak range, waveform length,
zero-crossing rate, slope-sign-change rate, and normalized waveform length.
The 66D glove vector contains the mean, standard deviation, and end-minus-start
change of each of 22 sensors. These are sensor-space pose summaries, **not
calibrated anatomical angles or full trajectories**.

All final Ridge models take 128 EMG features, even when clustering uses the
16D MAV view. Window weights equalize the total contribution of each calibration
trial. With standardized calibration features and pose targets, the objective is

$$
\min_{B,b}\;\sum_i w_i\|\widetilde y_i-B^T\widetilde x_i-b\|_2^2
+\alpha\|B\|_F^2.
$$

RidgeCV selects the regularization strength. Predicted pose is returned to sensor
units, then standardized against target A calibration statistics for semantic
comparison. Donor poses are standardized against each donor's own A statistics
before averaging within trials, across repetitions, and across the other nine
participants.

### Why grouping and naming are different

For each trial, average its windows and then normalize the EMG view and predicted
pose to unit length, giving $e_t$ and $v_t$. The clustering representation is

$$
h_t=\frac{[e_t;\sqrt{\lambda}\,v_t]}{\sqrt{1+\lambda}},
\qquad\lambda\in\{0,0.25,1\}.
$$

The algorithm minimizes within-cluster squared distances while assigning exactly
six of the 30 trials to each of five clusters. Multiple deterministic starts are
used; this is not a guarantee of the global clustering optimum.

For naming, compute cosine similarities between each trial's predicted pose and
each class prototype, then aggregate scores within each cluster. If $S_{jc}$ is
the resulting score for cluster $j$ and class $c$, naming solves

$$
\pi^*=\arg\max_{\pi\in\mathcal S_5}\sum_{j=1}^{5}S_{j,\pi(j)}.
$$

The Hungarian algorithm finds this one-to-one assignment. A cluster numbered
`0` has no inherent connection to `B1`. **Even when the pose clustering weight is
zero, naming still uses predicted pose and donor pose prototypes.**

### Selection and its boundaries

Candidate selection uses 12 A-class holdout episodes, not random windows from
the same gesture. An alternative must retain mean A BA within 0.005 of its stored
default, achieve B5/B6 proxy BA of at least 0.80, and improve
`mean A BA - 0.5 x SD` by at least 0.02. Otherwise the default remains.

A previously fixed assignment-margin gate selects between full 66D and dynamic
44D pose naming. The dynamic view omits the 22 mean features. Stored defaults and
participant-specific gates are inherited from earlier development; the final
selector is not a complete de novo selection procedure for an untouched person.
See the [selection protocol](code/STEP56_DB5_TARGET_A_STABILITY_PROTOCOL.md) and
[gate implementation](code/step52_db5_margin_gate.py).

## Internal development results

The frozen run contained 300 target trials in total. Mean trial-level balanced
accuracy and macro F1 were both **0.8733**. Mean window-level balanced accuracy was
**0.8730**. Six of ten target subjects had all 30 trials classified correctly.

| Target | Trial BA | Target | Trial BA |
|---|---:|---|---:|
| s1 | 0.6000 | s6 | 0.6000 |
| s2 | 1.0000 | s7 | 1.0000 |
| s3 | 1.0000 | s8 | 1.0000 |
| s4 | 0.8667 | s9 | 0.6667 |
| s5 | 1.0000 | s10 | 1.0000 |
| **Mean** | **0.8733** | **Range** | **0.6000-1.0000** |

![Balanced accuracy for each target subject](assets/subject_performance.png)

### Class-level outcomes

| True class | Correct / 60 | Recall | Main residual confusions |
|---|---:|---:|---|
| B1 | 60 | 1.0000 | None |
| B2 | 47 | 0.7833 | B3: 7, B4: 5, B7: 1 |
| B3 | 47 | 0.7833 | B2: 7, B7: 6 |
| B4 | 55 | 0.9167 | B2: 5 |
| B7 | 53 | 0.8833 | B3: 6, B2: 1 |

![Pooled confusion matrix and per-target class outcomes](assets/confusion_and_class_outcomes.png)

### Reading the result rather than only the average

- **Performance is heterogeneous.** Six participants have 30/30 correct trials,
  while s1 and s6 have 18/30. The mean is not a guarantee for a new user.
- **Errors are structured.** B1 has 60/60 correct trials; B2/B3 each have 47/60.
  A wrong cluster name can produce several correlated errors at once.
- **BA and macro F1 coincide here by construction.** Both true and predicted
  class counts are six per participant, so per-class precision equals recall.
  Their agreement is not a second independent finding.
- **388 windows are not 388 independent decisions.** Each window inherits its
  trial's prediction. The trial endpoint is primary; window BA is secondary.

The result supports the feasibility of this specified cluster-and-name procedure.
It does not establish that the predicted pose is anatomically accurate or that
the batch priors are unnecessary.

## Research practices visible in the code

| Practice | Inspectable implementation |
|---|---|
| Explicit class and subject access rules | [Target-A selection protocol](code/STEP56_DB5_TARGET_A_STABILITY_PROTOCOL.md), [symmetric donor construction](code/step42c_symmetric_reference.py) |
| Transparent signal-to-pose objective | [Group-balanced weights and Ridge fitting](code/step42b0_graded_primitive.py) |
| Separate grouping and semantic assignment | [Balanced clustering](code/step42c_balanced_transductive.py), [naming aggregation](code/step48_db5_stable_naming.py) |
| A selector that can retain its default | [Selection rule](code/step56_db5_target_a_stability.py), [guard and stability tests](tests/test_step56_db5_target_a_stability.py) |
| Prediction before label access | [Prediction runner](code/run_step56_db5_target_a_stability_phase2_predict.py), [separate evaluator](code/evaluate_step56_db5_target_a_stability.py) |
| A result that can be audited separately from training | [Replay verifier](code/verify_step56_db5_target_a_stability.py), [capacity and determinism tests](tests/test_step42c_balanced_transductive.py) |

These checks make a computational claim inspectable. They do not turn repeated
development on one dataset into a prospective study.

## What remains to be tested

| Next question | Discriminating test | Why it matters |
|---|---|---|
| Does the pose correspondence carry gesture information beyond grouping? | Freeze all choices, then compare intact correspondence with pairing- and semantic-permutation controls on an untouched evaluation. | Separates meaningful alignment from a convenient partition plus assignment prior. |
| How much does exact class balance contribute? | Evaluate the same frozen method against count-unconstrained and unequal-frequency conditions. | Separates signal evidence from batch side information. |
| Does the method transfer to a new acquisition? | Fix defaults, gates, and selection rules before acquiring another session or cohort. | Tests generalization beyond the development participants and recordings. |
| What information does the shared pose space preserve? | Measure pose recovery, cross-condition decoding, and nuisance sensitivity separately from gesture BA. | A representation useful for naming need not preserve all aspects of the underlying signal. |

These are tests needed to extend the claim, **not completed validations of this
released endpoint**. Later trajectory/composition experiments in the research
workspace are separate branches and are not pooled with this repository's 87.33%
result.

## Repository layout

```text
.
|-- assets/                 # README figures derived from the locked evaluation
|-- code/                   # Selection, prediction, evaluation, and verification
|-- tests/                  # Dataset-free unit tests for core pipeline behavior
|-- environment.yml         # Recorded Conda dependencies
`-- README.md
```

The numbered protocol and module names are retained as provenance identifiers from
the original research workspace. They record research chronology and do not imply
that every earlier exploratory experiment is part of this repository.

## Installation

```bash
conda env create -f environment.yml
conda activate ninapro-db5-unseen
export PYTHONPATH="$PWD/code"
```

The project was developed with Python 3.12.8. Exact package versions are recorded
in `environment.yml`.

## Dataset setup

NinaPro DB5 is not redistributed here. Obtain it from the
[official NinaPro DB5 data record](https://doi.org/10.5281/zenodo.1000116) and
comply with its license and terms. Raw `.mat` files, derived `.npz` caches,
evaluation labels, and full experiment artifacts must remain outside Git.

The code resolves the research workspace through `NINAPRO_CAPSTONE_ROOT`:

```bash
export NINAPRO_CAPSTONE_ROOT=/absolute/path/to/Capstone
```

If the variable is not set, the repository uses `local_workspace/`, which is
ignored by Git. A complete replay requires the prepared legal DB5 caches and locked
parent artifacts referenced by the protocol modules; the raw dataset alone is not
sufficient.

## Verification

The included unit tests do not require NinaPro data:

```bash
PYTHONPATH=code python -m unittest discover -s tests -v
```

With the legal caches and parent artifacts available, run the frozen sequence:

```bash
PYTHONPATH=code python code/run_step56_db5_target_a_stability_phase1.py
PYTHONPATH=code python code/run_step56_db5_target_a_stability_phase2_predict.py
PYTHONPATH=code python code/evaluate_step56_db5_target_a_stability.py
PYTHONPATH=code python code/verify_step56_db5_target_a_stability.py
```

Each command refuses to overwrite an existing run directory. Use `--run-name` for
a separate replay. If Phase 1 uses a non-default run name, pass its output path to
Phase 2 with `--phase1`.

## Important limitations

- The ten DB5 participants were all part of the broader method-development process;
  the reported value is internal development performance.
- The method assumes a closed five-class catalog, the complete query batch, and
  exactly six trials per class. This is stronger side information than online,
  inductive, or open-set recognition.
- Query inference uses target EMG only, but calibration requires target A1-A12 and
  B5/B6 glove data, and semantic naming requires evaluated-class glove prototypes
  from the other nine participants.
- The primary endpoint concerns one dataset, one acquisition setup, and intact
  participants. It does not establish performance for new subjects, sessions,
  devices, electrode shifts, amputee users, unequal class frequencies, real-time
  latency, or open-set rejection.
- Subject folds share overlapping nine-person donor banks, so the ten outcomes are
  not independent population samples.

## Data and release policy

Do not commit NinaPro subject data, raw or derived signals, query labels, result
manifests, model artifacts, or local paths. This repository intentionally contains
only source code, focused tests, protocol documents, and aggregate README figures.

No software license has been added to this repository. Unless a license is added,
the source remains available for inspection but is not granted an open-source reuse
license. NinaPro data remain governed by their own distribution terms.

## References

- Pizzolato S, et al. *Comparison of six electromyography acquisition setups on
  hand movement classification tasks*. PLOS ONE, 2017.
  [doi:10.1371/journal.pone.0186132](https://doi.org/10.1371/journal.pone.0186132)
- Atzori M, et al. *Electromyography data for non-invasive naturally-controlled
  robotic hand prostheses*. Scientific Data, 2014.
  [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53)
- Malinen MI, Franti P. *Balanced K-means for clustering*, 2014.
  [doi:10.1007/978-3-662-44415-3_4](https://doi.org/10.1007/978-3-662-44415-3_4)
