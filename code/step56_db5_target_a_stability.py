from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import step42c_balanced_transductive as core
import step48_db5_stable_naming as step48
import step50_db5_block_ranking as blockrank
import step52_db5_margin_gate as step52
from project_paths import CAPSTONE_ROOT as CAPSTONE, CODE_DIR


RESULT_ROOT = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step56_db5_target_a_stability"
)
PROTOCOL = CODE_DIR / "STEP56_DB5_TARGET_A_STABILITY_PROTOCOL.md"
NAMING_METHODS = tuple(step48.NAMING_METHODS)
CLUSTER_IDS = tuple(step48.CLUSTER_CANDIDATE_IDS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    def default(value: object) -> object:
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def candidate_id(cluster_id: str, naming_method: str) -> str:
    if cluster_id not in CLUSTER_IDS or naming_method not in NAMING_METHODS:
        raise ValueError("Unknown Step56 candidate component")
    return f"{cluster_id}__naming_{naming_method}"


def parse_candidate(value: str) -> tuple[str, str]:
    marker = "__naming_"
    if marker not in value:
        raise ValueError(value)
    cluster_id, method = value.rsplit(marker, maxsplit=1)
    if cluster_id not in CLUSTER_IDS or method not in NAMING_METHODS:
        raise ValueError(value)
    return cluster_id, method


def query_prediction(
    episode: core.UnlabeledEpisode,
    cluster_id: str,
    naming_method: str,
    gate_id: str,
) -> tuple[core.CandidatePrediction, dict[str, object]]:
    parent = core.get_candidate(cluster_id)
    clustered = core.predict_candidate(episode, parent)
    assignments = clustered.cluster_assignments
    if assignments is None:
        raise AssertionError("Step56 lost cluster assignments")
    prototypes = np.asarray(episode.source_prototypes)
    raw_scores = core.cosine_scores(episode.predicted_pose_trial, prototypes)
    dynamic_indices = np.arange(22, 66)
    dynamic_scores = core.cosine_scores(
        episode.predicted_pose_trial[:, dynamic_indices],
        prototypes[:, dynamic_indices],
    )
    raw_matrix = step48.naming_scores(raw_scores, assignments, naming_method)
    dynamic_matrix = step48.naming_scores(
        dynamic_scores, assignments, naming_method
    )
    raw_mapping = blockrank.mapping_from_matrix(raw_matrix)
    dynamic_mapping = blockrank.mapping_from_matrix(dynamic_matrix)
    raw_prediction = episode.catalog[raw_mapping[assignments]]
    dynamic_prediction = episode.catalog[dynamic_mapping[assignments]]
    record = step52.GateRecord(
        subject=episode.subject,
        proxy_id=episode.proxy_id,
        truth=np.empty(0, dtype=int),
        assignments=assignments,
        raw_matrix=raw_matrix,
        dynamic_matrix=dynamic_matrix,
        raw_prediction=raw_prediction,
        dynamic_prediction=dynamic_prediction,
        raw_margin=step52.assignment_margin(raw_matrix),
        dynamic_margin=step52.assignment_margin(dynamic_matrix),
    )
    use_dynamic = step52.choose_dynamic(record, gate_id)
    predictions = dynamic_prediction if use_dynamic else raw_prediction
    return (
        core.CandidatePrediction(
            predictions=np.asarray(predictions, dtype=int),
            pose_scores=raw_scores,
            cluster_assignments=assignments,
            cluster_sse=clustered.cluster_sse,
        ),
        {
            "raw_margin": record.raw_margin,
            "dynamic_margin": record.dynamic_margin,
            "use_dynamic": bool(use_dynamic),
            "mappings_differ": bool(
                not np.array_equal(raw_mapping, dynamic_mapping)
            ),
        },
    )


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_id", "cluster_candidate_id", "naming_method"]
    a = rows[rows["proxy"].str.startswith("A_")].groupby(
        keys, as_index=False
    ).agg(
        mean_a_trial=("trial_balanced_accuracy", "mean"),
        sd_a_trial=("trial_balanced_accuracy", "std"),
        mean_a_window=("window_balanced_accuracy", "mean"),
    )
    b = rows[rows["proxy"].eq("B56")][
        keys + ["trial_balanced_accuracy"]
    ].rename(columns={"trial_balanced_accuracy": "b56_trial"})
    result = a.merge(b, on=keys, validate="one_to_one")
    result["stability_score"] = (
        result["mean_a_trial"] - 0.5 * result["sd_a_trial"]
    )
    return result


def select_candidate(
    rows: pd.DataFrame, parent_cluster_id: str
) -> tuple[str, pd.DataFrame]:
    summary = summarize(rows)
    parent_id = candidate_id(parent_cluster_id, "mean")
    parent_rows = summary[summary["candidate_id"].eq(parent_id)]
    if len(parent_rows) != 1:
        raise AssertionError(f"Step56 parent missing: {parent_id}")
    parent = parent_rows.iloc[0]
    summary["gain_mean_a_trial"] = (
        summary["mean_a_trial"] - float(parent["mean_a_trial"])
    )
    summary["gain_stability"] = (
        summary["stability_score"] - float(parent["stability_score"])
    )
    summary["eligible"] = (
        ~summary["candidate_id"].eq(parent_id)
        & (summary["gain_mean_a_trial"] >= -0.005 - 1e-15)
        & (summary["b56_trial"] >= 0.80 - 1e-15)
        & (summary["gain_stability"] >= 0.02 - 1e-15)
    )
    eligible = summary[summary["eligible"]].sort_values(
        [
            "stability_score",
            "mean_a_trial",
            "mean_a_window",
            "b56_trial",
            "candidate_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    winner = parent if eligible.empty else eligible.iloc[0]
    selected = str(winner["candidate_id"])
    summary["selected"] = summary["candidate_id"].eq(selected)
    return selected, summary
