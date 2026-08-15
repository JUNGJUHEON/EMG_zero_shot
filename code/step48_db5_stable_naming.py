from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import step42c_balanced_transductive as core
from project_paths import CAPSTONE_ROOT as CAPSTONE, CODE_DIR


RESULT_ROOT = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step48_db5_stable_naming"
)
PROTOCOL = CODE_DIR / "STEP48_DB5_STABLE_NAMING_PROTOCOL.md"
SYMMETRIC_RUN = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "symmetric_all10_reference"
    / "all10_loso_20260812_v1"
)
QUERY_SEAL = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step42c_balanced_transductive"
    / "query_cache_seal"
    / "primary_s1_s10_20260722_v1"
)
ALL_SUBJECTS = tuple(f"s{index}" for index in range(1, 11))
PRIMARY_CLASSES = np.asarray([1, 2, 3, 4, 7], dtype=int)
NAMING_METHODS = ("mean", "median", "trim1")
CLUSTER_CANDIDATE_IDS = tuple(
    item.candidate_id for item in core.candidate_specs() if item.algorithm == "bcn"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
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
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def candidate_spec(cluster_candidate_id: str, method: str) -> core.CandidateSpec:
    if method not in NAMING_METHODS:
        raise ValueError(f"Unknown Step48 naming method: {method}")
    parent = core.get_candidate(cluster_candidate_id)
    if parent.algorithm != "bcn":
        raise ValueError("Step48 cluster parent must be BCN")
    return core.CandidateSpec(
        candidate_id=f"{cluster_candidate_id}__naming_{method}",
        algorithm="bcn",
        feature_set=parent.feature_set,
        pose_view_weight=parent.pose_view_weight,
    )


def naming_scores(
    pose_scores: np.ndarray,
    assignments: np.ndarray,
    method: str,
) -> np.ndarray:
    pose_scores = np.asarray(pose_scores, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=int)
    clusters = sorted(set(assignments.tolist()))
    rows: list[np.ndarray] = []
    for cluster in clusters:
        values = pose_scores[assignments == cluster]
        if values.shape[0] != 6:
            # Every legal and primary DB5 cluster has six trials.
            raise AssertionError("Step48 cluster capacity changed")
        if method == "mean":
            score = values.mean(axis=0)
        elif method == "median":
            score = np.median(values, axis=0)
        elif method == "trim1":
            score = np.sort(values, axis=0)[1:-1].mean(axis=0)
        else:
            raise ValueError(f"Unknown Step48 naming method: {method}")
        rows.append(score)
    result = np.vstack(rows)
    if result.shape != (pose_scores.shape[1], pose_scores.shape[1]):
        raise AssertionError("Step48 naming score matrix is not square")
    return result


def predict(
    episode: core.UnlabeledEpisode,
    cluster_candidate_id: str,
    method: str,
    *,
    semantic_permutation: np.ndarray | None = None,
    cluster_feature_permutation: np.ndarray | None = None,
    fixed_cluster_assignments: np.ndarray | None = None,
    fixed_cluster_sse: float | None = None,
) -> core.CandidatePrediction:
    parent = core.get_candidate(cluster_candidate_id)
    if parent.algorithm != "bcn":
        raise ValueError("Step48 cluster parent must be BCN")
    base = core.predict_candidate(
        episode,
        parent,
        semantic_permutation=semantic_permutation,
        cluster_feature_permutation=cluster_feature_permutation,
        fixed_cluster_assignments=fixed_cluster_assignments,
        fixed_cluster_sse=fixed_cluster_sse,
    )
    if base.cluster_assignments is None:
        raise AssertionError("Step48 requires BCN cluster assignments")
    scores = naming_scores(base.pose_scores, base.cluster_assignments, method)
    cluster_rows, class_columns = linear_sum_assignment(-scores)
    cluster_to_class = np.empty(len(episode.catalog), dtype=int)
    cluster_to_class[cluster_rows] = class_columns
    class_indices = cluster_to_class[base.cluster_assignments]
    predictions = episode.catalog[class_indices]
    return core.CandidatePrediction(
        predictions=np.asarray(predictions, dtype=int),
        pose_scores=base.pose_scores,
        cluster_assignments=base.cluster_assignments,
        cluster_sse=base.cluster_sse,
    )
