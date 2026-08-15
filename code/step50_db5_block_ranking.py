from __future__ import annotations

import hashlib
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

import step42c_balanced_transductive as core
from project_paths import CAPSTONE_ROOT as CAPSTONE, CODE_DIR


RESULT_ROOT = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step50_db5_block_ranking"
)
PROTOCOL = CODE_DIR / "STEP50_DB5_BLOCK_RANKING_PROTOCOL.md"
SYMMETRIC_RUN = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "symmetric_all10_reference"
    / "all10_loso_20260812_v1"
)
STEP48_SELECTIONS = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step48_db5_stable_naming"
    / "phase1_legal_proxy"
    / "db5_only_stable_naming_20260812_v1"
    / "frozen_outer_selections.csv"
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
BLOCKS = (np.arange(0, 22), np.arange(22, 44), np.arange(44, 66))
TEMPERATURE = 0.1
REGULARIZER = 0.01


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


def simplex_grid() -> tuple[tuple[float, float, float], ...]:
    result = []
    for first in range(21):
        for second in range(21 - first):
            third = 20 - first - second
            result.append((first / 20.0, second / 20.0, third / 20.0))
    if len(result) != 231:
        raise AssertionError("Step50 simplex grid changed")
    return tuple(result)


def block_matrices(
    episode: core.UnlabeledEpisode,
    assignments: np.ndarray,
    semantic_permutation: np.ndarray | None = None,
) -> np.ndarray:
    prototypes = np.asarray(episode.source_prototypes, dtype=np.float64)
    if semantic_permutation is not None:
        permutation = np.asarray(semantic_permutation, dtype=int)
        if set(permutation.tolist()) != set(range(len(prototypes))):
            raise ValueError("Invalid Step50 semantic permutation")
        prototypes = prototypes[permutation]
    matrices = []
    for indices in BLOCKS:
        trial_scores = core.cosine_scores(
            episode.predicted_pose_trial[:, indices], prototypes[:, indices]
        )
        rows = []
        for cluster in range(len(episode.catalog)):
            values = trial_scores[np.asarray(assignments) == cluster]
            if len(values) != 6:
                raise AssertionError("Step50 cluster capacity changed")
            rows.append(values.mean(axis=0))
        matrices.append(np.vstack(rows))
    return np.stack(matrices)


def combined_matrix(matrices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[0] != 3:
        raise ValueError("Invalid Step50 block matrices")
    if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid Step50 block weights")
    return np.tensordot(weights, matrices, axes=(0, 0))


def mapping_from_matrix(matrix: np.ndarray) -> np.ndarray:
    rows, columns = linear_sum_assignment(-np.asarray(matrix, dtype=np.float64))
    mapping = np.empty(len(matrix), dtype=int)
    mapping[rows] = columns
    return mapping


def predict(
    episode: core.UnlabeledEpisode,
    cluster_candidate_id: str,
    weights: np.ndarray,
    *,
    semantic_permutation: np.ndarray | None = None,
    cluster_feature_permutation: np.ndarray | None = None,
    fixed_cluster_assignments: np.ndarray | None = None,
    fixed_cluster_sse: float | None = None,
) -> core.CandidatePrediction:
    parent = core.get_candidate(cluster_candidate_id)
    clustered = core.predict_candidate(
        episode,
        parent,
        cluster_feature_permutation=cluster_feature_permutation,
        fixed_cluster_assignments=fixed_cluster_assignments,
        fixed_cluster_sse=fixed_cluster_sse,
    )
    if clustered.cluster_assignments is None:
        raise AssertionError("Step50 requires BCN assignments")
    matrices = block_matrices(
        episode, clustered.cluster_assignments, semantic_permutation
    )
    matrix = combined_matrix(matrices, np.asarray(weights, dtype=np.float64))
    mapping = mapping_from_matrix(matrix)
    predictions = episode.catalog[mapping[clustered.cluster_assignments]]
    return core.CandidatePrediction(
        predictions=np.asarray(predictions, dtype=int),
        pose_scores=core.cosine_scores(
            episode.predicted_pose_trial, episode.source_prototypes
        ),
        cluster_assignments=clustered.cluster_assignments,
        cluster_sse=clustered.cluster_sse,
    )


def oracle_mapping(assignments: np.ndarray, truth: np.ndarray, catalog: np.ndarray) -> np.ndarray:
    counts = np.asarray(
        [
            [np.sum((assignments == cluster) & (truth == label)) for label in catalog]
            for cluster in range(len(catalog))
        ],
        dtype=float,
    )
    return mapping_from_matrix(counts)


def permutation_nll(matrix: np.ndarray, true_mapping: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    options = np.asarray(list(permutations(range(len(matrix)))), dtype=int)
    cluster_indices = np.arange(len(matrix))[None, :]
    scores = matrix[cluster_indices, options].sum(axis=1) / TEMPERATURE
    true_score = matrix[np.arange(len(matrix)), true_mapping].sum() / TEMPERATURE
    return float(logsumexp(scores) - true_score)
