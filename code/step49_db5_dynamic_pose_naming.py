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
    / "step49_db5_dynamic_pose_naming"
)
PROTOCOL = CODE_DIR / "STEP49_DB5_DYNAMIC_POSE_NAMING_PROTOCOL.md"
SYMMETRIC_RUN = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "symmetric_all10_reference"
    / "all10_loso_20260812_v1"
)
STEP48_PHASE1 = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step48_db5_stable_naming"
    / "phase1_legal_proxy"
    / "db5_only_stable_naming_20260812_v1"
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
POSE_VIEWS = {"all66": np.arange(66), "dynamic44": np.arange(22, 66)}
AGGREGATORS = ("mean", "median")
PARENT_POLICIES = ("original", "stable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def candidate_id(parent_policy: str, pose_view: str, aggregator: str) -> str:
    if parent_policy not in PARENT_POLICIES:
        raise ValueError(parent_policy)
    if pose_view not in POSE_VIEWS or aggregator not in AGGREGATORS:
        raise ValueError("Unknown Step49 naming candidate")
    return f"{parent_policy}__{pose_view}__{aggregator}"


def predict(
    episode: core.UnlabeledEpisode,
    cluster_candidate_id: str,
    pose_view: str,
    aggregator: str,
    *,
    semantic_permutation: np.ndarray | None = None,
    cluster_feature_permutation: np.ndarray | None = None,
    fixed_cluster_assignments: np.ndarray | None = None,
    fixed_cluster_sse: float | None = None,
) -> core.CandidatePrediction:
    parent = core.get_candidate(cluster_candidate_id)
    if parent.algorithm != "bcn":
        raise ValueError("Step49 parent must be BCN")
    clustered = core.predict_candidate(
        episode,
        parent,
        cluster_feature_permutation=cluster_feature_permutation,
        fixed_cluster_assignments=fixed_cluster_assignments,
        fixed_cluster_sse=fixed_cluster_sse,
    )
    assignments = clustered.cluster_assignments
    if assignments is None:
        raise AssertionError("Step49 lost cluster assignments")
    indices = POSE_VIEWS[pose_view]
    prototypes = np.asarray(episode.source_prototypes, dtype=np.float64)
    if semantic_permutation is not None:
        permutation = np.asarray(semantic_permutation, dtype=int)
        if set(permutation.tolist()) != set(range(len(prototypes))):
            raise ValueError("Invalid Step49 semantic permutation")
        prototypes = prototypes[permutation]
    pose_scores = core.cosine_scores(
        episode.predicted_pose_trial[:, indices], prototypes[:, indices]
    )
    naming_rows: list[np.ndarray] = []
    for cluster in range(len(episode.catalog)):
        values = pose_scores[assignments == cluster]
        if len(values) != 6:
            raise AssertionError("Step49 cluster capacity changed")
        if aggregator == "mean":
            naming_rows.append(values.mean(axis=0))
        elif aggregator == "median":
            naming_rows.append(np.median(values, axis=0))
        else:
            raise ValueError(aggregator)
    naming = np.vstack(naming_rows)
    cluster_rows, class_columns = linear_sum_assignment(-naming)
    cluster_to_class = np.empty(len(episode.catalog), dtype=int)
    cluster_to_class[cluster_rows] = class_columns
    predictions = episode.catalog[cluster_to_class[assignments]]
    return core.CandidatePrediction(
        predictions=np.asarray(predictions, dtype=int),
        pose_scores=pose_scores,
        cluster_assignments=assignments,
        cluster_sse=clustered.cluster_sse,
    )
