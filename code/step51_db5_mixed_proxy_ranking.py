from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import step42b0_graded_primitive as b0
import step42b3_primitive_attribute_bridge as b3
import step42c_balanced_transductive as core
import step44_exercise_conditioned_naming as step44
import step44_legal_common as mixed
import step50_db5_block_ranking as blockrank
from project_paths import CAPSTONE_ROOT as CAPSTONE, CODE_DIR


RESULT_ROOT = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step51_db5_mixed_proxy_ranking"
)
PROTOCOL = CODE_DIR / "STEP51_DB5_MIXED_PROXY_RANKING_PROTOCOL.md"
SYMMETRIC_RUN = blockrank.SYMMETRIC_RUN
STEP48_SELECTIONS = blockrank.STEP48_SELECTIONS
QUERY_SEAL = blockrank.QUERY_SEAL
ALL_SUBJECTS = blockrank.ALL_SUBJECTS


@dataclass(frozen=True)
class MixedRecord:
    subject: str
    proxy_id: str
    held_b: int
    catalog: np.ndarray
    assignments: np.ndarray
    truth: np.ndarray
    block_matrices: np.ndarray
    raw_prediction: np.ndarray


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


def source_anchors(
    subjects: dict[str, b3.PreparedSubject],
    target_subject: str,
    proxy: mixed.MixedProxy,
) -> np.ndarray:
    rows = []
    for source_subject in ALL_SUBJECTS:
        if source_subject == target_subject:
            continue
        prepared = subjects[source_subject]
        fit, _ = mixed.proxy_masks(prepared.manifest, proxy)
        scaler = core._pose_scaler(prepared, fit)
        pose = b0.transform(prepared.glove_features, scaler)
        anchors = []
        for exercise_code, class_id in mixed.held_catalog(proxy):
            prototype, _ = mixed._equal_repetition_prototype(
                pose, prepared.manifest, exercise_code, class_id
            )
            anchors.append(prototype)
        rows.append(np.vstack(anchors))
    result = np.mean(np.stack(rows), axis=0)
    if result.shape != (5, 66) or len(rows) != 9:
        raise AssertionError("Step51 source anchor scope changed")
    return result


def build_record(
    target: b3.PreparedSubject,
    subjects: dict[str, b3.PreparedSubject],
    proxy: mixed.MixedProxy,
    cluster_candidate_id: str,
) -> MixedRecord:
    fit, query = mixed.proxy_masks(target.manifest, proxy)
    fit_manifest = target.manifest.loc[fit].reset_index(drop=True)
    query_manifest = target.manifest.loc[query].reset_index(drop=True)
    weights = b0.equal_group_weights(fit_manifest)
    ridge = b0.fit_weighted_ridge(
        target.emg_features[fit],
        target.glove_features[fit],
        target.emg_features[query],
        weights,
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
        target.emg_features[fit][:, columns], sample_weight=weights
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
            "step51_cluster", target.subject, proxy.proxy_id, cluster_candidate_id
        ),
    )
    anchors = source_anchors(subjects, target.subject, proxy)
    episode = core.UnlabeledEpisode(
        subject=target.subject,
        proxy_id=proxy.proxy_id,
        catalog=np.arange(5, dtype=int),
        trial_ids=np.asarray([f"q{index:02d}" for index in range(30)]),
        window_trial_index=window_trial_index,
        window_row_keys=np.arange(len(query_manifest)),
        predicted_pose_trial=query_pose,
        predicted_pose_window=query_pose_window,
        source_prototypes=anchors,
        emg_trial_views={candidate.feature_set: emg_view},
        n_fit_windows=int(fit.sum()),
        ridge_alpha=ridge.alpha,
        source_subjects=tuple(s for s in ALL_SUBJECTS if s != target.subject),
    )
    matrices = blockrank.block_matrices(episode, assignments)
    raw_scores = core.cosine_scores(query_pose, anchors)
    raw_naming = np.vstack(
        [raw_scores[assignments == cluster].mean(axis=0) for cluster in range(5)]
    )
    raw_mapping = blockrank.mapping_from_matrix(raw_naming)
    return MixedRecord(
        subject=target.subject,
        proxy_id=proxy.proxy_id,
        held_b=proxy.held_b,
        catalog=np.arange(5, dtype=int),
        assignments=assignments,
        truth=truth,
        block_matrices=matrices,
        raw_prediction=raw_mapping[assignments],
    )


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean([np.mean(prediction[truth == slot] == slot) for slot in range(5)]))


def held_b_recall(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(prediction[truth == 4] == 4))
