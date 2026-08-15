from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import step52_db5_margin_gate as step52
import step56_db5_target_a_stability as step56


STEP48_SELECTIONS = step52.STEP48_SELECTIONS
STEP52_PHASE2 = (
    step52.RESULT_ROOT
    / "phase2_primary_prediction"
    / "db5_only_margin_gate_20260812_v1"
)


def manual_ba(truth: np.ndarray, prediction: np.ndarray) -> float:
    labels = sorted(set(truth.tolist()))
    return float(
        np.mean([np.mean(prediction[truth == label] == label) for label in labels])
    )


def independent_selection_replay(
    metrics: pd.DataFrame,
    parent_cluster: str,
) -> str:
    rows = []
    for candidate_id, group in metrics.groupby("candidate_id"):
        a = group[group["proxy"].str.startswith("A_")]
        b = group[group["proxy"].eq("B56")]
        if len(a) != 12 or len(b) != 1:
            raise AssertionError("Step56 verifier proxy inventory changed")
        rows.append(
            {
                "candidate_id": candidate_id,
                "mean_a": float(a["trial_balanced_accuracy"].mean()),
                "sd_a": float(a["trial_balanced_accuracy"].std(ddof=1)),
                "window_a": float(a["window_balanced_accuracy"].mean()),
                "b56": float(b["trial_balanced_accuracy"].iloc[0]),
            }
        )
    summary = pd.DataFrame(rows)
    summary["stability"] = summary["mean_a"] - 0.5 * summary["sd_a"]
    parent_id = f"{parent_cluster}__naming_mean"
    parent = summary[summary["candidate_id"].eq(parent_id)].iloc[0]
    eligible = summary[
        ~summary["candidate_id"].eq(parent_id)
        & (summary["mean_a"] >= float(parent["mean_a"]) - 0.005 - 1e-15)
        & (summary["b56"] >= 0.80 - 1e-15)
        & (summary["stability"] >= float(parent["stability"]) + 0.02 - 1e-15)
    ].sort_values(
        ["stability", "mean_a", "window_a", "b56", "candidate_id"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    return parent_id if eligible.empty else str(eligible.iloc[0]["candidate_id"])


def cluster_oracle(assignments: np.ndarray, truth: np.ndarray) -> float:
    clusters = sorted(set(assignments.tolist()))
    labels = sorted(set(truth.tolist()))
    matrix = np.asarray(
        [
            [np.sum((assignments == cluster) & (truth == label)) for label in labels]
            for cluster in clusters
        ],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].sum() / len(truth))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="db5_target_a_stability_20260812_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase1 = step56.RESULT_ROOT / "phase1_legal_proxy" / args.run_name
    phase2 = step56.RESULT_ROOT / "phase2_primary_prediction" / args.run_name
    phase3 = step56.RESULT_ROOT / "phase3_evaluation" / args.run_name
    phase1_lock = json.loads((phase1 / "phase1_lock.json").read_text(encoding="utf-8"))
    phase2_lock = json.loads((phase2 / "prediction_lock.json").read_text(encoding="utf-8"))
    if phase1_lock["status"] != "locked_before_primary_prediction":
        raise AssertionError("Step56 verifier Phase 1 status failed")
    if phase2_lock["status"] != "predictions_locked_before_truth_join":
        raise AssertionError("Step56 verifier Phase 2 status failed")
    if phase1_lock["truth_opened"] or phase2_lock["truth_opened"]:
        raise AssertionError("Step56 verifier found early truth access")
    checks: dict[str, object] = {
        "dataset_db5_only": phase1_lock["dataset"] == "NinaPro DB5 only"
        and phase2_lock["dataset"] == "NinaPro DB5 only",
        "protocol_hash_matches": step56.sha256_file(step56.PROTOCOL)
        == phase1_lock["protocol_sha256"]
        == phase2_lock["protocol_sha256"],
        "phase1_selection_hash_matches": step56.sha256_file(
            phase1 / "frozen_subject_selections.csv"
        )
        == phase1_lock["artifacts"]["frozen_subject_selections.csv"]
        == phase2_lock["selections_sha256"],
        "phase2_prediction_hash_matches": step56.sha256_file(
            phase2 / "trial_predictions.csv"
        )
        == phase2_lock["trial_predictions_sha256"],
        "phase2_window_hash_matches": step56.sha256_file(
            phase2 / "window_manifest.csv"
        )
        == phase2_lock["window_manifest_sha256"],
    }
    source = (
        Path(__file__).parent / "run_step56_db5_target_a_stability_phase2_predict.py"
    ).read_text(encoding="utf-8")
    checks["phase2_source_has_no_truth_seal_or_evaluator"] = (
        "truth_seal" not in source
        and "score_frozen_predictions" not in source
        and "QUERY_SEAL" not in source
    )

    metrics = pd.read_csv(phase1 / "legal_proxy_candidate_metrics.csv")
    selections = pd.read_csv(phase1 / "frozen_subject_selections.csv").set_index(
        "subject"
    )
    parents = pd.read_csv(STEP48_SELECTIONS).set_index("outer_subject")
    replayed = {}
    for subject in step52.ALL_SUBJECTS:
        replayed[subject] = independent_selection_replay(
            metrics[metrics["subject"].eq(subject)],
            str(parents.loc[subject, "selected_cluster_candidate_id"]),
        )
    checks["all_selections_independently_replayed"] = all(
        replayed[subject] == selections.loc[subject, "selected_candidate_id"]
        for subject in step52.ALL_SUBJECTS
    )

    new_predictions = pd.read_csv(phase2 / "trial_predictions.csv")
    parent_predictions = pd.read_csv(STEP52_PHASE2 / "trial_predictions.csv")
    expected_parent = parent_predictions[
        parent_predictions["control_type"].eq("selected")
    ].copy()
    expected_parent["control_type"] = "step52_parent"
    expected_parent["control_index"] = -1
    keys = ["subject", "control_type", "control_index", "opaque_trial_id"]
    observed_parent = new_predictions[
        new_predictions["control_type"].eq("step52_parent")
    ].sort_values(keys).reset_index(drop=True)
    expected_parent = expected_parent.sort_values(keys).reset_index(drop=True)
    checks["step52_parent_predictions_exactly_reused"] = observed_parent.equals(
        expected_parent
    )

    truth = pd.read_csv(step52.QUERY_SEAL / "truth_seal.csv")
    trial_truth = truth.groupby(["subject", "opaque_trial_id"], as_index=False).agg(
        truth=("class", "first")
    )
    joined = new_predictions.merge(
        trial_truth, on=["subject", "opaque_trial_id"], validate="many_to_one"
    )
    manual_rows = []
    for (subject, control), group in joined.groupby(["subject", "control_type"]):
        manual_rows.append(
            {
                "subject": subject,
                "control_type": control,
                "manual_ba": manual_ba(
                    group["truth"].to_numpy(int), group["prediction"].to_numpy(int)
                ),
            }
        )
    manual = pd.DataFrame(manual_rows)
    reported = pd.read_csv(phase3 / "evaluation_metrics.csv")[
        ["subject", "control_type", "trial_balanced_accuracy"]
    ]
    comparison = manual.merge(
        reported, on=["subject", "control_type"], validate="one_to_one"
    )
    checks["all_trial_ba_independently_recomputed"] = bool(
        np.allclose(
            comparison["manual_ba"],
            comparison["trial_balanced_accuracy"],
            atol=1e-15,
        )
    )
    selected_mean = float(
        manual[manual["control_type"].eq("selected")]["manual_ba"].mean()
    )
    parent_mean = float(
        manual[manual["control_type"].eq("step52_parent")]["manual_ba"].mean()
    )
    checks["selected_mean_is_0p873333"] = abs(selected_mean - 0.8733333333333334) < 1e-12
    checks["parent_mean_is_0p81"] = abs(parent_mean - 0.81) < 1e-12
    s4 = joined[
        joined["subject"].eq("s4") & joined["control_type"].eq("selected")
    ]
    s4_ba = manual_ba(s4["truth"].to_numpy(int), s4["prediction"].to_numpy(int))
    s4_oracle = cluster_oracle(
        s4["cluster"].to_numpy(int), s4["truth"].to_numpy(int)
    )
    checks["s4_selected_ba_is_0p866667"] = abs(s4_ba - 0.8666666666666667) < 1e-12
    checks["all_checks_pass"] = bool(all(bool(value) for value in checks.values()))
    if not checks["all_checks_pass"]:
        failed = [key for key, value in checks.items() if not bool(value)]
        raise AssertionError(f"Step56 verification failed: {failed}")
    payload = {
        "status": "verified",
        "checks": checks,
        "independent_selected_mean_trial_ba": selected_mean,
        "independent_parent_mean_trial_ba": parent_mean,
        "independent_s4_trial_ba": s4_ba,
        "independent_s4_cluster_oracle": s4_oracle,
        "replayed_selections": replayed,
        "protocol_sha256": step56.sha256_file(step56.PROTOCOL),
        "phase1_lock_sha256": step56.sha256_file(phase1 / "phase1_lock.json"),
        "phase2_lock_sha256": step56.sha256_file(phase2 / "prediction_lock.json"),
    }
    step56.write_json(phase3 / "verification.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
