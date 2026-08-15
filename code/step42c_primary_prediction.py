from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

import step42b0_graded_primitive as common
import step42b3_primitive_attribute_bridge as b3
import step42c_balanced_transductive as transductive
import step42c_primary_cache as primary_cache


PRIMARY_CLASSES = np.asarray([1, 2, 3, 4, 7], dtype=int)


@dataclass(frozen=True)
class PrimaryEpisodeState:
    ridge: common.RidgePrediction
    pose_scaler: common.ScalerState
    selected_feature_scaler: common.ScalerState


def prepare_primary_episode(
    legal_subject: b3.PreparedSubject,
    query_cache: primary_cache.UnlabeledPrimaryCache,
    source_prototypes: np.ndarray,
    selected_candidate: transductive.CandidateSpec,
) -> tuple[transductive.UnlabeledEpisode, PrimaryEpisodeState]:
    if legal_subject.subject != query_cache.subject:
        raise AssertionError("Step42C legal/query subject mismatch")
    if selected_candidate.algorithm != "bcn" or selected_candidate.feature_set is None:
        raise AssertionError("Step42C primary selection must be BCN with an EMG view")
    fit_manifest = legal_subject.manifest.reset_index(drop=True)
    weights = common.equal_group_weights(fit_manifest)
    ridge = common.fit_weighted_ridge(
        legal_subject.emg_features,
        legal_subject.glove_features,
        query_cache.emg_features,
        weights,
    )
    fit_all = np.ones(len(fit_manifest), dtype=bool)
    pose_scaler = transductive._pose_scaler(legal_subject, fit_all)
    predicted_pose_window = common.transform(ridge.prediction, pose_scaler)

    trial_ids = np.asarray(
        sorted(set(query_cache.opaque_trial_id.astype(str).tolist())), dtype="U32"
    )
    trial_to_index = {trial_id: index for index, trial_id in enumerate(trial_ids)}
    window_trial_index = np.asarray(
        [trial_to_index[item] for item in query_cache.opaque_trial_id.astype(str)],
        dtype=int,
    )
    if len(trial_ids) != 30:
        raise AssertionError("Step42C primary query must contain 30 opaque trials")
    predicted_pose_trial = transductive._group_mean(
        predicted_pose_window, window_trial_index, len(trial_ids)
    )

    emg_trial_views: dict[str, np.ndarray] = {}
    selected_scaler_state: common.ScalerState | None = None
    for feature_set, columns in transductive.FEATURE_SETS.items():
        scaler = StandardScaler().fit(
            legal_subject.emg_features[:, columns], sample_weight=weights
        )
        transformed = scaler.transform(query_cache.emg_features[:, columns])
        emg_trial_views[feature_set] = transductive._l2_normalize(
            transductive._group_mean(transformed, window_trial_index, len(trial_ids))
        )
        if feature_set == selected_candidate.feature_set:
            selected_scaler_state = common.ScalerState(
                mean=np.asarray(scaler.mean_, dtype=np.float64),
                scale=np.maximum(np.asarray(scaler.scale_, dtype=np.float64), 1e-12),
            )
    if selected_scaler_state is None:
        raise AssertionError("Step42C selected feature scaler was not fitted")

    source_prototypes = np.asarray(source_prototypes, dtype=np.float64)
    if source_prototypes.shape != (5, 66) or not np.isfinite(source_prototypes).all():
        raise AssertionError("Step42C primary source prototypes are invalid")
    episode = transductive.UnlabeledEpisode(
        subject=legal_subject.subject,
        proxy_id="primary_B1_B2_B3_B4_B7",
        catalog=PRIMARY_CLASSES.copy(),
        trial_ids=trial_ids,
        window_trial_index=window_trial_index,
        window_row_keys=query_cache.opaque_row_id.astype(str),
        predicted_pose_trial=predicted_pose_trial,
        predicted_pose_window=predicted_pose_window,
        source_prototypes=source_prototypes,
        emg_trial_views=emg_trial_views,
        n_fit_windows=len(fit_manifest),
        ridge_alpha=ridge.alpha,
        source_subjects=transductive.source_subjects_for(legal_subject.subject),
    )
    return episode, PrimaryEpisodeState(
        ridge=ridge,
        pose_scaler=pose_scaler,
        selected_feature_scaler=selected_scaler_state,
    )


def prediction_rows(
    episode: transductive.UnlabeledEpisode,
    candidate: transductive.CandidateSpec,
    prediction: transductive.CandidatePrediction,
    control_type: str,
    control_index: int | None,
) -> np.ndarray:
    rows = np.empty(
        len(episode.trial_ids),
        dtype=[
            ("subject", "U4"),
            ("candidate_id", "U48"),
            ("control_type", "U32"),
            ("control_index", "i4"),
            ("opaque_trial_id", "U32"),
            ("prediction", "i4"),
            ("cluster", "i4"),
        ],
    )
    rows["subject"] = episode.subject
    rows["candidate_id"] = candidate.candidate_id
    rows["control_type"] = control_type
    rows["control_index"] = -1 if control_index is None else int(control_index)
    rows["opaque_trial_id"] = episode.trial_ids
    rows["prediction"] = prediction.predictions
    rows["cluster"] = (
        prediction.cluster_assignments
        if prediction.cluster_assignments is not None
        else np.full(len(episode.trial_ids), -1, dtype=int)
    )
    return rows
