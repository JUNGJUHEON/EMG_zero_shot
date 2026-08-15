from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import step42b0_legal_cache as legal
import step42b3_primitive_attribute_bridge as b3
import step42c_balanced_transductive as core
import step42c_primary_cache as primary_cache
import step42c_symmetric_reference as symmetric
import step52_db5_margin_gate as step52
import step56_db5_target_a_stability as step56


STEP52_PHASE1 = (
    step52.RESULT_ROOT / "phase1_legal_proxy" / "db5_only_margin_gate_20260812_v1"
)
STEP52_PHASE2 = (
    step52.RESULT_ROOT
    / "phase2_primary_prediction"
    / "db5_only_margin_gate_20260812_v1"
)
PROTOTYPES = step52.SYMMETRIC_RUN / "symmetric_source_prototypes.npz"


def load_subjects() -> dict[str, b3.PreparedSubject]:
    result: dict[str, b3.PreparedSubject] = {}
    for subject in step52.ALL_SUBJECTS:
        cache = legal.load_legal_amplitude_cache(subject)
        if "DB1" in json.dumps(cache.metadata):
            raise AssertionError("Step56 DB1 path detected")
        result[subject] = b3.prepare_subject(cache)
    return result


def frame(
    episode: core.UnlabeledEpisode,
    candidate_id: str,
    prediction: core.CandidatePrediction,
    control_type: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": episode.subject,
            "candidate_id": candidate_id,
            "control_type": control_type,
            "control_index": -1,
            "opaque_trial_id": episode.trial_ids.astype(str),
            "prediction": prediction.predictions.astype(int),
            "cluster": prediction.cluster_assignments.astype(int),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase1",
        default=str(
            step56.RESULT_ROOT
            / "phase1_legal_proxy"
            / "db5_target_a_stability_20260812_v1"
        ),
    )
    parser.add_argument("--run-name", default="db5_target_a_stability_20260812_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase1 = Path(args.phase1)
    lock_path = phase1 / "phase1_lock.json"
    selections_path = phase1 / "frozen_subject_selections.csv"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_primary_prediction" or lock["truth_opened"]:
        raise AssertionError("Step56 Phase 1 lock failed")
    if step56.sha256_file(step56.PROTOCOL) != lock["protocol_sha256"]:
        raise AssertionError("Step56 protocol changed")
    if (
        step56.sha256_file(selections_path)
        != lock["artifacts"]["frozen_subject_selections.csv"]
    ):
        raise AssertionError("Step56 selections changed")

    parent_lock_path = STEP52_PHASE2 / "prediction_lock.json"
    parent_lock = json.loads(parent_lock_path.read_text(encoding="utf-8"))
    parent_predictions_path = STEP52_PHASE2 / "trial_predictions.csv"
    parent_windows_path = STEP52_PHASE2 / "window_manifest.csv"
    if (
        parent_lock["status"] != "predictions_locked_before_truth_join"
        or parent_lock["truth_opened"]
        or step56.sha256_file(parent_predictions_path)
        != parent_lock["trial_predictions_sha256"]
        or step56.sha256_file(parent_windows_path)
        != parent_lock["window_manifest_sha256"]
    ):
        raise AssertionError("Step56 parent prediction lock failed")

    output = step56.RESULT_ROOT / "phase2_primary_prediction" / args.run_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selections = pd.read_csv(selections_path).set_index("subject")
    gates = pd.read_csv(STEP52_PHASE1 / "frozen_outer_gates.csv").set_index(
        "outer_subject"
    )
    subjects = load_subjects()
    with np.load(PROTOTYPES, allow_pickle=False) as payload:
        prototypes = {
            subject: np.asarray(payload[subject], dtype=np.float64)
            for subject in step52.ALL_SUBJECTS
        }
    parent_predictions = pd.read_csv(parent_predictions_path)
    parent_selected = parent_predictions[
        parent_predictions["control_type"].eq("selected")
    ].copy()
    parent_selected["control_type"] = "step52_parent"
    parent_selected["control_index"] = -1

    frames: list[pd.DataFrame] = [parent_selected]
    details: list[pd.DataFrame] = []
    states: list[dict[str, object]] = []
    for subject in step52.ALL_SUBJECTS:
        selected_id = str(selections.loc[subject, "selected_candidate_id"])
        cluster_id, naming_method = step56.parse_candidate(selected_id)
        gate_id = str(gates.loc[subject, "selected_gate_id"])
        query = primary_cache.load_unlabeled_primary_cache(subject)
        episode, state = symmetric.prepare_primary_episode(
            subjects[subject],
            query,
            prototypes[subject],
            core.get_candidate(cluster_id),
            symmetric.source_subjects_for(subject),
        )
        prediction, gate_state = step56.query_prediction(
            episode, cluster_id, naming_method, gate_id
        )
        frames.append(frame(episode, selected_id, prediction, "selected"))
        details.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "candidate_id": selected_id,
                    "cluster_candidate_id": cluster_id,
                    "naming_method": naming_method,
                    "selected_gate_id": gate_id,
                    "raw_margin": gate_state["raw_margin"],
                    "dynamic_margin": gate_state["dynamic_margin"],
                    "use_dynamic": gate_state["use_dynamic"],
                    "mappings_differ": gate_state["mappings_differ"],
                    "opaque_trial_id": episode.trial_ids.astype(str),
                    "prediction": prediction.predictions.astype(int),
                    "cluster": prediction.cluster_assignments.astype(int),
                }
            )
        )
        states.append(
            {
                "subject": subject,
                "candidate_id": selected_id,
                "cluster_candidate_id": cluster_id,
                "naming_method": naming_method,
                "selected_gate_id": gate_id,
                **gate_state,
                "ridge_alpha": state.ridge.alpha,
                "cluster_sse": prediction.cluster_sse,
            }
        )
        print(f"Step56 primary prediction complete: {subject}", flush=True)

    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["subject", "control_type", "opaque_trial_id"], kind="mergesort"
    )
    if len(predictions) != 600:
        raise AssertionError(f"Step56 expected 600 trial predictions, got {len(predictions)}")
    predictions_path = output / "trial_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    window_frame = pd.read_csv(parent_windows_path)
    windows_path = output / "window_manifest.csv"
    window_frame.to_csv(windows_path, index=False)
    pd.concat(details, ignore_index=True).to_csv(
        output / "selected_trial_details.csv", index=False
    )
    pd.DataFrame(states).to_csv(output / "model_state_summary.csv", index=False)
    step56.write_json(
        output / "prediction_lock.json",
        {
            "created": datetime.now().astimezone().isoformat(),
            "status": "predictions_locked_before_truth_join",
            "dataset": "NinaPro DB5 only",
            "phase1_lock_sha256": step56.sha256_file(lock_path),
            "selections_sha256": step56.sha256_file(selections_path),
            "protocol_sha256": step56.sha256_file(step56.PROTOCOL),
            "source_sha256": step56.sha256_file(Path(__file__).resolve()),
            "parent_prediction_lock_sha256": step56.sha256_file(parent_lock_path),
            "trial_predictions_sha256": step56.sha256_file(predictions_path),
            "window_manifest_sha256": step56.sha256_file(windows_path),
            "truth_opened": False,
        },
    )
    print(f"Step56 Phase 2 prediction output: {output}", flush=True)


if __name__ == "__main__":
    main()
