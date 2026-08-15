# NinaPro DB5 Unseen-Gesture Draft

Private research snapshot accompanying the Korean and English manuscript drafts in
`paper/`. The repository contains the code path used for the final internal
NinaPro DB5 experiment and focused unit tests. Raw NinaPro data, derived caches,
intermediate results, and evaluation labels are not redistributed.

## Experiment

- Dataset: NinaPro DB5, subjects `s1`-`s10`.
- Target-subject calibration: Exercise A gestures `A1`-`A12` and Exercise B
  gestures `B5` and `B6`.
- Evaluated unseen gestures: `B1`, `B2`, `B3`, `B4`, and `B7`.
- Cross-subject reference: the other nine DB5 subjects for each held-out target.
- Target-query input: EMG only. Target-query pose and target-query class labels are
  excluded before prediction.
- Transductive side information: the five-class inventory, six trials per class,
  and the complete unlabeled target-query batch are known.
- Final internal result reported in the drafts: mean trial-level balanced accuracy
  and macro F1 of `0.8733`; window-level balanced accuracy of `0.8730`.

These are internal development results, not an independent external validation.
The protocol Markdown files preserve the selection, prediction-lock, and
truth-opening sequence used during development.

## Contents

- `code/`: final DB5 selection, prediction, evaluation, verification, and their
  direct dependencies.
- `tests/`: focused tests for balanced assignment and target-A stability selection.
- `paper/manuscript_ko.pdf`: Korean manuscript draft.
- `paper/manuscript_en.pdf`: English manuscript draft.

The numbered script names are retained as provenance identifiers from the research
workspace. They are not manuscript section numbers.

## Environment

The recorded environment is defined in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate ninapro-db5-draft
export PYTHONPATH="$PWD/code"
```

For the original local research tree, point the code at the Capstone root:

```bash
export NINAPRO_CAPSTONE_ROOT=/path/to/Capstone
```

Without this variable, the code uses `local_workspace/` under the repository root.
That directory is ignored by Git.

## Verification

The included unit tests do not require the NinaPro dataset:

```bash
PYTHONPATH=code python -m unittest discover -s tests -v
```

Replaying the complete experiment requires the prepared DB5 legal caches and the
locked parent artifacts at the paths encoded by the protocol modules. With those
artifacts present under `NINAPRO_CAPSTONE_ROOT`, the final sequence is:

```bash
PYTHONPATH=code python code/run_step56_db5_target_a_stability_phase1.py
PYTHONPATH=code python code/run_step56_db5_target_a_stability_phase2_predict.py
PYTHONPATH=code python code/evaluate_step56_db5_target_a_stability.py
PYTHONPATH=code python code/verify_step56_db5_target_a_stability.py
```

Each command refuses to overwrite an existing run directory. Use `--run-name` to
create a distinct replay, and pass the resulting Phase 1 path to Phase 2 with
`--phase1` when a non-default run name is used.

## Data

Obtain NinaPro DB5 from its official distributor and comply with its terms. Do not
commit `.mat`, `.npz`, manifests, result tables, model artifacts, or subject data to
this repository.
