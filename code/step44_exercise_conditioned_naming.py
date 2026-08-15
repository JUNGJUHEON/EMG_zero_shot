from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12
MECHANISM_IDS = (
    "direct_raw",
    "a_offset_null",
    "exercise_b_offset",
    "offset_sign_reverse",
    "wrong_b_correspondence",
)


@dataclass(frozen=True)
class NamingPrediction:
    predictions: np.ndarray
    trial_scores: np.ndarray
    naming_scores: np.ndarray
    cluster_to_class: np.ndarray
    adjusted_anchors: np.ndarray


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _finite_matrix(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"Step44 {name} must be a finite matrix")
    return result


def _exercise_roles(name: str, values: np.ndarray, size: int) -> np.ndarray:
    result = np.char.upper(np.asarray(values, dtype="U1"))
    if result.shape != (size,) or not set(result.tolist()).issubset({"A", "B"}):
        raise ValueError(f"Step44 {name} exercise roles are invalid")
    return result


def l2_normalize(values: np.ndarray, *, require_nonzero: bool = False) -> np.ndarray:
    values = _finite_matrix("normalization input", values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if require_nonzero and np.any(norms <= EPSILON):
        raise ValueError("Step44 encountered a zero-norm vector")
    return values / np.maximum(norms, EPSILON)


def cosine_scores(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    query = _finite_matrix("cosine query", query)
    prototypes = _finite_matrix("cosine prototypes", prototypes)
    if query.shape[1] != prototypes.shape[1]:
        raise ValueError("Step44 cosine dimensions do not match")
    result = l2_normalize(query) @ l2_normalize(prototypes, require_nonzero=True).T
    if not np.isfinite(result).all():
        raise AssertionError("Step44 cosine scores are not finite")
    return result


def exercise_conditioned_anchors(
    target_landmarks: np.ndarray,
    source_landmarks: np.ndarray,
    landmark_exercises: np.ndarray,
    source_anchors: np.ndarray,
    anchor_exercises: np.ndarray,
    mechanism_id: str,
    *,
    wrong_source_b: np.ndarray | None = None,
) -> np.ndarray:
    target_landmarks = _finite_matrix("target landmarks", target_landmarks)
    source_landmarks = _finite_matrix("source landmarks", source_landmarks)
    source_anchors = _finite_matrix("source anchors", source_anchors)
    if (
        target_landmarks.shape != source_landmarks.shape
        or target_landmarks.shape[1] != source_anchors.shape[1]
        or mechanism_id not in MECHANISM_IDS
    ):
        raise ValueError("Step44 anchor geometry is incompatible")
    landmark_exercises = _exercise_roles(
        "landmark", landmark_exercises, len(target_landmarks)
    )
    anchor_exercises = _exercise_roles("anchor", anchor_exercises, len(source_anchors))
    adjusted = source_anchors.copy()
    held_b = anchor_exercises == "B"
    seen_b = landmark_exercises == "B"
    seen_a = landmark_exercises == "A"

    if mechanism_id == "direct_raw" or not held_b.any():
        return adjusted
    if (
        mechanism_id
        in {
            "exercise_b_offset",
            "offset_sign_reverse",
            "wrong_b_correspondence",
        }
        and not seen_b.any()
    ):
        return adjusted

    if mechanism_id == "a_offset_null":
        if not seen_a.any():
            return adjusted
        residual = (target_landmarks[seen_a] - source_landmarks[seen_a]).mean(axis=0)
    elif mechanism_id == "wrong_b_correspondence":
        wrong_source_b = np.asarray(wrong_source_b, dtype=np.float64)
        if (
            wrong_source_b.shape != (target_landmarks.shape[1],)
            or not np.isfinite(wrong_source_b).all()
        ):
            raise ValueError("Step44 wrong-B source prototype is invalid")
        residual = target_landmarks[seen_b].mean(axis=0) - wrong_source_b
    else:
        residual = (target_landmarks[seen_b] - source_landmarks[seen_b]).mean(axis=0)
        if mechanism_id == "offset_sign_reverse":
            residual = -residual

    adjusted[held_b] += residual
    if not np.isfinite(adjusted).all():
        raise AssertionError("Step44 adjusted anchors are not finite")
    return adjusted


def canonical_cluster_mapping(naming_scores: np.ndarray) -> np.ndarray:
    naming_scores = _finite_matrix("naming scores", naming_scores)
    if naming_scores.shape[0] != naming_scores.shape[1] or len(naming_scores) > 8:
        raise ValueError("Step44 naming scores must be a small square matrix")
    size = len(naming_scores)
    best_objective = -np.inf
    best_mapping: tuple[int, ...] | None = None
    for permutation in itertools.permutations(range(size)):
        objective = float(
            sum(naming_scores[row, permutation[row]] for row in range(size))
        )
        if objective > best_objective or (
            objective == best_objective
            and (best_mapping is None or permutation < best_mapping)
        ):
            best_objective = objective
            best_mapping = permutation
    if best_mapping is None:
        raise AssertionError("Step44 naming found no cluster mapping")
    return np.asarray(best_mapping, dtype=int)


def name_frozen_clusters(
    assignments: np.ndarray,
    trial_scores: np.ndarray,
    catalog: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignments = np.asarray(assignments, dtype=int)
    trial_scores = _finite_matrix("trial scores", trial_scores)
    catalog = np.asarray(catalog, dtype=int)
    size = len(catalog)
    if (
        assignments.shape != (len(trial_scores),)
        or trial_scores.shape[1] != size
        or catalog.shape != (size,)
        or len(set(catalog.tolist())) != size
        or set(assignments.tolist()) != set(range(size))
    ):
        raise ValueError("Step44 frozen-cluster naming inputs are invalid")
    naming_scores = np.vstack(
        [trial_scores[assignments == cluster].mean(axis=0) for cluster in range(size)]
    )
    cluster_to_class = canonical_cluster_mapping(naming_scores)
    predictions = catalog[cluster_to_class[assignments]]
    if set(predictions.tolist()) != set(catalog.tolist()):
        raise AssertionError("Step44 naming lost one-to-one catalog coverage")
    return predictions, naming_scores, cluster_to_class


def predict_mechanism(
    assignments: np.ndarray,
    query_pose: np.ndarray,
    target_landmarks: np.ndarray,
    source_landmarks: np.ndarray,
    landmark_exercises: np.ndarray,
    source_anchors: np.ndarray,
    anchor_exercises: np.ndarray,
    catalog: np.ndarray,
    mechanism_id: str,
    *,
    wrong_source_b: np.ndarray | None = None,
) -> NamingPrediction:
    adjusted_anchors = exercise_conditioned_anchors(
        target_landmarks,
        source_landmarks,
        landmark_exercises,
        source_anchors,
        anchor_exercises,
        mechanism_id,
        wrong_source_b=wrong_source_b,
    )
    trial_scores = cosine_scores(query_pose, adjusted_anchors)
    predictions, naming_scores, cluster_to_class = name_frozen_clusters(
        assignments, trial_scores, catalog
    )
    return NamingPrediction(
        predictions=predictions,
        trial_scores=trial_scores,
        naming_scores=naming_scores,
        cluster_to_class=cluster_to_class,
        adjusted_anchors=adjusted_anchors,
    )
