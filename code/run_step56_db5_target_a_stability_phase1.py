from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import step42b0_legal_cache as legal
import step42b3_primitive_attribute_bridge as b3
import step42c_balanced_transductive as core
import step42c_symmetric_reference as symmetric
import step52_db5_margin_gate as step52
import step56_db5_target_a_stability as step56


STEP52_PHASE1 = (
    step52.RESULT_ROOT / "phase1_legal_proxy" / "db5_only_margin_gate_20260812_v1"
)


def load_subjects() -> dict[str, b3.PreparedSubject]:
    result: dict[str, b3.PreparedSubject] = {}
    for subject in step52.ALL_SUBJECTS:
        cache = legal.load_legal_amplitude_cache(subject)
        if "DB1" in json.dumps(cache.metadata):
            raise AssertionError("Step56 DB1 path detected")
        result[subject] = b3.prepare_subject(cache)
    return result


def run_grid() -> pd.DataFrame:
    gates_path = STEP52_PHASE1 / "frozen_outer_gates.csv"
    gate_lock = json.loads(
        (STEP52_PHASE1 / "phase1_lock.json").read_text(encoding="utf-8")
    )
    if gate_lock["status"] != "locked_before_primary_prediction":
        raise AssertionError("Step56 parent gate is not locked")
    if step56.sha256_file(gates_path) != gate_lock["selection_sha256"]:
        raise AssertionError("Step56 parent gates changed")
    gates = pd.read_csv(gates_path).set_index("outer_subject")
    subjects = load_subjects()
    rows: list[dict[str, object]] = []
    for subject in step52.ALL_SUBJECTS:
        gate_id = str(gates.loc[subject, "selected_gate_id"])
        for proxy in core.proxy_specs():
            prepared = symmetric.prepare_episode(subjects[subject], subjects, proxy)
            for cluster_id in step56.CLUSTER_IDS:
                for method in step56.NAMING_METHODS:
                    prediction, state = step56.query_prediction(
                        prepared.unlabeled, cluster_id, method, gate_id
                    )
                    spec = core.get_candidate(cluster_id)
                    metrics, _ = core.evaluate_prediction(
                        prepared, spec, prediction
                    )
                    rows.append(
                        {
                            **metrics,
                            "candidate_id": step56.candidate_id(cluster_id, method),
                            "cluster_candidate_id": cluster_id,
                            "naming_method": method,
                            "selected_gate_id": gate_id,
                            **state,
                        }
                    )
        print(f"Step56 legal A stability grid complete: {subject}", flush=True)
    frame = pd.DataFrame(rows).sort_values(
        ["subject", "proxy", "candidate_id"], kind="mergesort"
    )
    expected = len(step52.ALL_SUBJECTS) * len(core.proxy_specs()) * 27
    if len(frame) != expected:
        raise AssertionError(f"Step56 grid has {len(frame)} rows, expected {expected}")
    return frame


def freeze_selections(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parents = pd.read_csv(step52.STEP48_SELECTIONS).set_index("outer_subject")
    selections: list[dict[str, object]] = []
    rankings: list[pd.DataFrame] = []
    for subject in step52.ALL_SUBJECTS:
        parent_cluster = str(
            parents.loc[subject, "selected_cluster_candidate_id"]
        )
        selected, ranking = step56.select_candidate(
            grid[grid["subject"].eq(subject)], parent_cluster
        )
        parent_id = step56.candidate_id(parent_cluster, "mean")
        winner = ranking[ranking["candidate_id"].eq(selected)].iloc[0]
        parent = ranking[ranking["candidate_id"].eq(parent_id)].iloc[0]
        selected_cluster, selected_method = step56.parse_candidate(selected)
        selections.append(
            {
                "subject": subject,
                "parent_candidate_id": parent_id,
                "selected_candidate_id": selected,
                "selected_cluster_candidate_id": selected_cluster,
                "selected_naming_method": selected_method,
                "changed": bool(selected != parent_id),
                "selected_stability_score": winner["stability_score"],
                "parent_stability_score": parent["stability_score"],
                "stability_gain": (
                    winner["stability_score"] - parent["stability_score"]
                ),
                "selected_mean_a_trial": winner["mean_a_trial"],
                "parent_mean_a_trial": parent["mean_a_trial"],
                "selected_b56_trial": winner["b56_trial"],
                "parent_b56_trial": parent["b56_trial"],
            }
        )
        ranking.insert(0, "subject", subject)
        ranking.insert(1, "parent_candidate_id", parent_id)
        rankings.append(ranking)
    return pd.DataFrame(selections), pd.concat(rankings, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="db5_target_a_stability_20260812_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = step56.RESULT_ROOT / "phase1_legal_proxy" / args.run_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    grid = run_grid()
    grid_path = output / "legal_proxy_candidate_metrics.csv"
    grid.to_csv(grid_path, index=False)
    selections, rankings = freeze_selections(grid)
    selection_path = output / "frozen_subject_selections.csv"
    selections.to_csv(selection_path, index=False)
    rankings.to_csv(output / "subject_candidate_rankings.csv", index=False)
    summary = {
        "created": datetime.now().astimezone().isoformat(),
        "dataset": "NinaPro DB5 only",
        "n_candidates": 27,
        "n_proxy_episodes": len(step52.ALL_SUBJECTS) * len(core.proxy_specs()),
        "n_changed_subjects": int(selections["changed"].sum()),
        "s4_selection": selections[selections["subject"].eq("s4")].iloc[0].to_dict(),
        "mean_stability_gain": float(selections["stability_gain"].mean()),
    }
    step56.write_json(output / "phase1_summary.json", summary)
    artifacts = {
        path.name: step56.sha256_file(path)
        for path in (
            grid_path,
            selection_path,
            output / "subject_candidate_rankings.csv",
            output / "phase1_summary.json",
        )
    }
    step56.write_json(
        output / "phase1_lock.json",
        {
            "created": datetime.now().astimezone().isoformat(),
            "status": "locked_before_primary_prediction",
            "dataset": "NinaPro DB5 only",
            "evaluated_classes_forbidden": "B1/B2/B3/B4/B7",
            "protocol_sha256": step56.sha256_file(step56.PROTOCOL),
            "source_sha256": step56.sha256_file(Path(__file__).resolve()),
            "step48_parent_sha256": step56.sha256_file(step52.STEP48_SELECTIONS),
            "step52_gates_sha256": step56.sha256_file(
                STEP52_PHASE1 / "frozen_outer_gates.csv"
            ),
            "artifacts": artifacts,
            "truth_opened": False,
        },
    )
    print(selections.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Step56 Phase 1 output: {output}", flush=True)


if __name__ == "__main__":
    main()
