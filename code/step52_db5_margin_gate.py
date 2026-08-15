from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import step42b0_graded_primitive as b0
import step42b3_primitive_attribute_bridge as b3
import step42c_balanced_transductive as core
import step44_exercise_conditioned_naming as step44
import step44_legal_common as mixed
import step49_db5_dynamic_pose_naming as dynamic
import step50_db5_block_ranking as blockrank
import step51_db5_mixed_proxy_ranking as step51
from project_paths import CAPSTONE_ROOT as CAPSTONE, CODE_DIR


RESULT_ROOT = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step52_db5_margin_gate"
)
PROTOCOL = CODE_DIR / "STEP52_DB5_MARGIN_GATE_PROTOCOL.md"
SYMMETRIC_RUN = step51.SYMMETRIC_RUN
STEP48_SELECTIONS = step51.STEP48_SELECTIONS
QUERY_SEAL = step51.QUERY_SEAL
ALL_SUBJECTS = step51.ALL_SUBJECTS
THRESHOLDS = (0.005, 0.01, 0.02, 0.04, 0.08)


@dataclass(frozen=True)
class GateRecord:
    subject: str
    proxy_id: str
    truth: np.ndarray
    assignments: np.ndarray
    raw_matrix: np.ndarray
    dynamic_matrix: np.ndarray
    raw_prediction: np.ndarray
    dynamic_prediction: np.ndarray
    raw_margin: float
    dynamic_margin: float


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
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def assignment_margin(matrix: np.ndarray) -> float:
    options = np.asarray(list(permutations(range(len(matrix)))), dtype=int)
    scores = matrix[np.arange(len(matrix))[None, :], options].sum(axis=1)
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2])


def build_record(
    target: b3.PreparedSubject,
    subjects: dict[str, b3.PreparedSubject],
    proxy: mixed.MixedProxy,
    cluster_candidate_id: str,
) -> GateRecord:
    fit, query = mixed.proxy_masks(target.manifest, proxy)
    fit_manifest = target.manifest.loc[fit].reset_index(drop=True)
    query_manifest = target.manifest.loc[query].reset_index(drop=True)
    fit_weights = b0.equal_group_weights(fit_manifest)
    ridge = b0.fit_weighted_ridge(
        target.emg_features[fit],
        target.glove_features[fit],
        target.emg_features[query],
        fit_weights,
    )
    pose_scaler = core._pose_scaler(target, fit)
    query_pose_window = b0.transform(ridge.prediction, pose_scaler)
    _, window_trial_index, truth, _ = mixed._opaque_query(
        query_manifest, target.subject, proxy
    )
    query_pose = mixed._group_mean(query_pose_window, window_trial_index, 30)
    candidate = core.get_candidate(cluster_candidate_id)
    columns = b3.FEATURE_SETS[candidate.feature_set]
    emg_scaler = StandardScaler().fit(
        target.emg_features[fit][:, columns], sample_weight=fit_weights
    )
    emg_view = step44.l2_normalize(
        mixed._group_mean(
            emg_scaler.transform(target.emg_features[query][:, columns]),
            window_trial_index,
            30,
        )
    )
    if candidate.pose_view_weight == 0:
        cluster_view = emg_view
    else:
        pose_view = step44.l2_normalize(query_pose)
        weight = float(candidate.pose_view_weight)
        cluster_view = np.concatenate(
            (emg_view, np.sqrt(weight) * pose_view), axis=1
        ) / np.sqrt(1 + weight)
    assignments, _, _ = core.balanced_kmeans(
        cluster_view,
        k=5,
        capacity=6,
        seed=step44.stable_seed(
            "step52_cluster", target.subject, proxy.proxy_id, cluster_candidate_id
        ),
    )
    anchors = step51.source_anchors(subjects, target.subject, proxy)
    raw_scores = core.cosine_scores(query_pose, anchors)
    raw_matrix = np.vstack(
        [raw_scores[assignments == cluster].mean(axis=0) for cluster in range(5)]
    )
    indices = np.arange(22, 66)
    dynamic_scores = core.cosine_scores(query_pose[:, indices], anchors[:, indices])
    dynamic_matrix = np.vstack(
        [dynamic_scores[assignments == cluster].mean(axis=0) for cluster in range(5)]
    )
    raw_mapping = blockrank.mapping_from_matrix(raw_matrix)
    dynamic_mapping = blockrank.mapping_from_matrix(dynamic_matrix)
    return GateRecord(
        subject=target.subject,
        proxy_id=proxy.proxy_id,
        truth=truth,
        assignments=assignments,
        raw_matrix=raw_matrix,
        dynamic_matrix=dynamic_matrix,
        raw_prediction=raw_mapping[assignments],
        dynamic_prediction=dynamic_mapping[assignments],
        raw_margin=assignment_margin(raw_matrix),
        dynamic_margin=assignment_margin(dynamic_matrix),
    )


def gate_ids() -> tuple[str, ...]:
    return ("raw", "larger_margin", *(f"threshold_{value:g}" for value in THRESHOLDS))


def choose_dynamic(record: GateRecord, gate_id: str) -> bool:
    mappings_differ = not np.array_equal(record.raw_prediction, record.dynamic_prediction)
    dynamic_larger = record.dynamic_margin > record.raw_margin
    if gate_id == "raw":
        return False
    if gate_id == "larger_margin":
        return mappings_differ and dynamic_larger
    if gate_id.startswith("threshold_"):
        threshold = float(gate_id.removeprefix("threshold_"))
        if threshold not in THRESHOLDS:
            raise ValueError(gate_id)
        return mappings_differ and dynamic_larger and record.raw_margin < threshold
    raise ValueError(gate_id)


def query_prediction(
    episode: core.UnlabeledEpisode,
    cluster_candidate_id: str,
    gate_id: str,
    *,
    semantic_permutation: np.ndarray | None = None,
    cluster_feature_permutation: np.ndarray | None = None,
    fixed_cluster_assignments: np.ndarray | None = None,
    fixed_cluster_sse: float | None = None,
) -> tuple[core.CandidatePrediction, dict[str, object]]:
    parent = core.get_candidate(cluster_candidate_id)
    clustered = core.predict_candidate(
        episode,
        parent,
        cluster_feature_permutation=cluster_feature_permutation,
        fixed_cluster_assignments=fixed_cluster_assignments,
        fixed_cluster_sse=fixed_cluster_sse,
    )
    assignments = clustered.cluster_assignments
    if assignments is None:
        raise AssertionError("Step52 lost assignments")
    prototypes = np.asarray(episode.source_prototypes)
    if semantic_permutation is not None:
        prototypes = prototypes[np.asarray(semantic_permutation, dtype=int)]
    raw_scores = core.cosine_scores(episode.predicted_pose_trial, prototypes)
    raw_matrix = np.vstack(
        [raw_scores[assignments == cluster].mean(axis=0) for cluster in range(5)]
    )
    indices = np.arange(22, 66)
    dyn_scores = core.cosine_scores(
        episode.predicted_pose_trial[:, indices], prototypes[:, indices]
    )
    dyn_matrix = np.vstack(
        [dyn_scores[assignments == cluster].mean(axis=0) for cluster in range(5)]
    )
    raw_mapping = blockrank.mapping_from_matrix(raw_matrix)
    dyn_mapping = blockrank.mapping_from_matrix(dyn_matrix)
    raw_prediction = episode.catalog[raw_mapping[assignments]]
    dyn_prediction = episode.catalog[dyn_mapping[assignments]]
    raw_margin = assignment_margin(raw_matrix)
    dyn_margin = assignment_margin(dyn_matrix)
    proxy_record = GateRecord(
        subject=episode.subject,
        proxy_id=episode.proxy_id,
        truth=np.empty(0, dtype=int),
        assignments=assignments,
        raw_matrix=raw_matrix,
        dynamic_matrix=dyn_matrix,
        raw_prediction=raw_prediction,
        dynamic_prediction=dyn_prediction,
        raw_margin=raw_margin,
        dynamic_margin=dyn_margin,
    )
    use_dynamic = choose_dynamic(proxy_record, gate_id)
    prediction = dyn_prediction if use_dynamic else raw_prediction
    return (
        core.CandidatePrediction(
            predictions=np.asarray(prediction, dtype=int),
            pose_scores=raw_scores,
            cluster_assignments=assignments,
            cluster_sse=clustered.cluster_sse,
        ),
        {
            "raw_margin": raw_margin,
            "dynamic_margin": dyn_margin,
            "use_dynamic": use_dynamic,
            "mappings_differ": not np.array_equal(raw_mapping, dyn_mapping),
        },
    )
