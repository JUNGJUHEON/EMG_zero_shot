from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import step42b0_graded_primitive as common
import step42b3_primitive_attribute_bridge as b3
import step42c_balanced_transductive as core
import step42c_primary_prediction as primary


ALL_SUBJECTS = tuple(core.SUBJECTS)


@dataclass(frozen=True)
class PoseOnlyData:
    manifest: pd.DataFrame
    glove_features: np.ndarray


def _subject_pose_scaler(
    pose_data: PoseOnlyData, subject: str, policy: str
) -> StandardScaler:
    subject_mask = pose_data.manifest["subject"].astype(str).eq(subject).to_numpy()
    scoped_manifest = pose_data.manifest.loc[subject_mask].reset_index(drop=True)
    scoped_pose = pose_data.glove_features[subject_mask]
    if policy == "PM_A_equal_group":
        rows = scoped_manifest["exercise_code"].astype(str).eq("A").to_numpy()
    elif policy == "PM_AB56_equal_group":
        exercise = scoped_manifest["exercise_code"].astype(str).to_numpy()
        movement = scoped_manifest["class"].astype(int).to_numpy()
        rows = (exercise == "A") | ((exercise == "B") & np.isin(movement, (5, 6)))
    else:
        raise ValueError(f"Unknown source pose metric policy: {policy}")
    weights = common.equal_group_weights(
        scoped_manifest.loc[rows].reset_index(drop=True)
    )
    return StandardScaler().fit(scoped_pose[rows], sample_weight=weights)


def source_subjects_for(target_subject: str) -> tuple[str, ...]:
    if target_subject not in ALL_SUBJECTS:
        raise ValueError(f"Unknown symmetric target subject: {target_subject}")
    sources = tuple(subject for subject in ALL_SUBJECTS if subject != target_subject)
    if len(sources) != 9 or len(set(sources)) != 9 or target_subject in sources:
        raise AssertionError("Symmetric all-ten source scope changed")
    return sources


def prepare_episode(
    target: b3.PreparedSubject,
    sources: Mapping[str, b3.PreparedSubject],
    proxy: core.ProxySpec,
) -> core.PreparedEpisode:
    fit, query = core.proxy_masks(target.manifest, proxy)
    fit_manifest = target.manifest.loc[fit].reset_index(drop=True)
    query_manifest = target.manifest.loc[query].reset_index(drop=True)
    weights = common.equal_group_weights(fit_manifest)
    ridge = common.fit_weighted_ridge(
        target.emg_features[fit],
        target.glove_features[fit],
        target.emg_features[query],
        weights,
    )
    target_pose_scaler = core._pose_scaler(target, fit)
    predicted_pose_window = common.transform(ridge.prediction, target_pose_scaler)

    trial_ids, window_trial_index, trial_truth, trial_repetitions = core._opaque_trials(
        query_manifest, target.subject, proxy.proxy_id
    )
    n_trials = len(trial_ids)
    expected_trials = len(proxy.held_classes) * len(core.REPETITIONS)
    if n_trials != expected_trials:
        raise AssertionError(
            f"Symmetric proxy expected {expected_trials} trials, observed {n_trials}"
        )
    counts = pd.Series(trial_truth).value_counts().to_dict()
    if set(counts) != set(proxy.held_classes) or set(counts.values()) != {6}:
        raise AssertionError("Symmetric proxy query is not six trials per class")
    predicted_pose_trial = core._group_mean(
        predicted_pose_window, window_trial_index, n_trials
    )

    emg_trial_views: dict[str, np.ndarray] = {}
    for feature_set, columns in core.FEATURE_SETS.items():
        scaler = StandardScaler().fit(
            target.emg_features[fit][:, columns], sample_weight=weights
        )
        query_scaled = scaler.transform(target.emg_features[query][:, columns])
        emg_trial_views[feature_set] = core._l2_normalize(
            core._group_mean(query_scaled, window_trial_index, n_trials)
        )

    source_subjects = source_subjects_for(target.subject)
    prototypes, audit = core._source_prototypes(sources, source_subjects, proxy)
    audit.insert(0, "target_subject", target.subject)
    query_row_keys = query_manifest["step41_cache_row_index"].to_numpy(dtype=np.int64)
    unlabeled = core.UnlabeledEpisode(
        subject=target.subject,
        proxy_id=proxy.proxy_id,
        catalog=np.asarray(proxy.held_classes, dtype=int),
        trial_ids=trial_ids,
        window_trial_index=window_trial_index,
        window_row_keys=query_row_keys,
        predicted_pose_trial=predicted_pose_trial,
        predicted_pose_window=predicted_pose_window,
        source_prototypes=prototypes,
        emg_trial_views=emg_trial_views,
        n_fit_windows=int(fit.sum()),
        ridge_alpha=ridge.alpha,
        source_subjects=source_subjects,
    )
    truth = core.EpisodeTruth(
        trial_truth=trial_truth,
        window_truth=query_manifest["class"].to_numpy(dtype=int),
        trial_repetitions=trial_repetitions,
    )
    return core.PreparedEpisode(
        unlabeled=unlabeled,
        truth=truth,
        source_anchor_audit=audit,
    )


def outer_subject_select(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outer in ALL_SUBJECTS:
        development = set(source_subjects_for(outer))
        frame = summary[summary["subject"].isin(development)]
        ranking = (
            frame.groupby(
                ["candidate_id", "algorithm", "feature_set", "pose_view_weight"],
                as_index=False,
                dropna=False,
            )
            .agg(
                development_a_trial=("a_trial_balanced_accuracy", "mean"),
                development_a_window=("a_window_balanced_accuracy", "mean"),
                development_b56_trial=("b56_trial_balanced_accuracy", "mean"),
                development_margin=("a_mean_true_pose_margin", "mean"),
                n_development_subjects=("subject", "nunique"),
            )
            .sort_values(
                [
                    "development_a_trial",
                    "development_a_window",
                    "development_b56_trial",
                    "development_margin",
                    "candidate_id",
                ],
                ascending=[False, False, False, False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        if len(ranking) != len(core.candidate_specs()):
            raise AssertionError("Symmetric outer selection lost a candidate")
        winner = ranking.iloc[0]
        if int(winner["n_development_subjects"]) != 9:
            raise AssertionError("Symmetric outer selection lost a development subject")
        rows.append(
            {
                "outer_subject": outer,
                "development_subjects": ",".join(source_subjects_for(outer)),
                "selected_candidate_id": winner["candidate_id"],
                "selected_algorithm": winner["algorithm"],
                "selected_feature_set": winner["feature_set"],
                "selected_pose_view_weight": winner["pose_view_weight"],
                "development_a_trial": winner["development_a_trial"],
                "development_a_window": winner["development_a_window"],
                "development_b56_trial": winner["development_b56_trial"],
                "development_margin": winner["development_margin"],
                "n_development_subjects": int(winner["n_development_subjects"]),
            }
        )
    return pd.DataFrame(rows)


def primary_source_prototypes(
    pose_data: PoseOnlyData,
    target_subject: str,
    classes: Iterable[int],
    pose_metric_policy: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    labels = tuple(classes)
    sources = source_subjects_for(target_subject)
    manifest = pose_data.manifest
    source_values = manifest["subject"].astype(str).to_numpy()
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    movements = manifest["class"].astype(int).to_numpy()
    repetitions = manifest["repetition"].astype(int).to_numpy()
    subject_class_means: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    for source in sources:
        scaler = _subject_pose_scaler(pose_data, source, pose_metric_policy)
        source_pose = scaler.transform(pose_data.glove_features)
        class_means: list[np.ndarray] = []
        for movement in labels:
            repetition_means: list[np.ndarray] = []
            for repetition in core.REPETITIONS:
                rows = (
                    (source_values == source)
                    & (exercise == "B")
                    & (movements == movement)
                    & (repetitions == repetition)
                )
                if not rows.any():
                    raise AssertionError(
                        f"Missing symmetric source anchor {source}/B{movement}/rep{repetition}"
                    )
                repetition_means.append(source_pose[rows].mean(axis=0))
                audit_rows.append(
                    {
                        "target_subject": target_subject,
                        "source_subject": source,
                        "class": movement,
                        "repetition": repetition,
                        "n_windows": int(rows.sum()),
                        "pose_metric_policy": pose_metric_policy,
                        "source_scope": "all_s1_s10_minus_target",
                    }
                )
            class_means.append(np.mean(repetition_means, axis=0))
        subject_class_means.append(np.vstack(class_means))
    prototypes = np.mean(subject_class_means, axis=0).astype(np.float32)
    if prototypes.shape != (len(labels), 66) or not np.isfinite(prototypes).all():
        raise AssertionError("Symmetric primary source prototypes are invalid")
    return prototypes, pd.DataFrame(audit_rows)


def prepare_primary_episode(
    legal_subject: b3.PreparedSubject,
    query_cache: object,
    source_prototypes: np.ndarray,
    selected_candidate: core.CandidateSpec,
    source_subjects: tuple[str, ...],
) -> tuple[core.UnlabeledEpisode, primary.PrimaryEpisodeState]:
    episode, state = primary.prepare_primary_episode(
        legal_subject,
        query_cache,
        source_prototypes,
        selected_candidate,
    )
    if legal_subject.subject in source_subjects:
        raise AssertionError("Primary source list contains its target")
    return replace(episode, source_subjects=tuple(source_subjects)), state
