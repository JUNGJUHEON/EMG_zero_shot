from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

import step42b0_graded_primitive as common
import step42b3_primitive_attribute_bridge as b3


PRIMARY_SUBJECTS = common.PRIMARY_SUBJECTS
DIAGNOSTIC_SUBJECT = common.DIAGNOSTIC_SUBJECT
SUBJECTS = (DIAGNOSTIC_SUBJECT, *PRIMARY_SUBJECTS)
REPETITIONS = tuple(range(1, 7))
PAIR_CLASSES = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
FEATURE_SETS = b3.FEATURE_SETS
POSE_VIEW_WEIGHTS = (0.0, 0.25, 1.0)
KMEANS_STARTS = 64
KMEANS_MAX_ITER = 100


@dataclass(frozen=True)
class ProxySpec:
    proxy_id: str
    held_exercise: str
    held_classes: tuple[int, ...]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    algorithm: str
    feature_set: str | None = None
    pose_view_weight: float | None = None


@dataclass(frozen=True)
class UnlabeledEpisode:
    subject: str
    proxy_id: str
    catalog: np.ndarray
    trial_ids: np.ndarray
    window_trial_index: np.ndarray
    window_row_keys: np.ndarray
    predicted_pose_trial: np.ndarray
    predicted_pose_window: np.ndarray
    source_prototypes: np.ndarray
    emg_trial_views: Mapping[str, np.ndarray]
    n_fit_windows: int
    ridge_alpha: float
    source_subjects: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeTruth:
    trial_truth: np.ndarray
    window_truth: np.ndarray
    trial_repetitions: np.ndarray


@dataclass(frozen=True)
class PreparedEpisode:
    unlabeled: UnlabeledEpisode
    truth: EpisodeTruth
    source_anchor_audit: pd.DataFrame


@dataclass(frozen=True)
class CandidatePrediction:
    predictions: np.ndarray
    pose_scores: np.ndarray
    cluster_assignments: np.ndarray | None
    cluster_sse: float | None


def proxy_specs() -> tuple[ProxySpec, ...]:
    result: list[ProxySpec] = []
    for omitted in range(len(PAIR_CLASSES)):
        for phase in (0, 1):
            held = tuple(
                pair[(pair_index + phase) % 2]
                for pair_index, pair in enumerate(PAIR_CLASSES)
                if pair_index != omitted
            )
            result.append(
                ProxySpec(
                    proxy_id=f"A_omit{omitted + 1}_phase{phase}",
                    held_exercise="A",
                    held_classes=held,
                )
            )
    result.append(
        ProxySpec(proxy_id="B56", held_exercise="B", held_classes=(5, 6))
    )
    a_specs = [item for item in result if item.held_exercise == "A"]
    counts = {
        class_id: sum(class_id in item.held_classes for item in a_specs)
        for class_id in range(1, 13)
    }
    if len(a_specs) != 12 or set(counts.values()) != {5}:
        raise AssertionError("Step42C A-panel balance changed")
    return tuple(result)


def get_proxy(proxy_id: str) -> ProxySpec:
    matches = [item for item in proxy_specs() if item.proxy_id == proxy_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown Step42C proxy: {proxy_id}")
    return matches[0]


def candidate_specs() -> tuple[CandidateSpec, ...]:
    result = [
        CandidateSpec("independent_pose", "independent_pose"),
        CandidateSpec("balanced_pose", "balanced_pose"),
    ]
    for feature_set in FEATURE_SETS:
        for weight in POSE_VIEW_WEIGHTS:
            token = f"{weight:g}".replace(".", "p")
            result.append(
                CandidateSpec(
                    candidate_id=f"bcn__{feature_set}__posew_{token}",
                    algorithm="bcn",
                    feature_set=feature_set,
                    pose_view_weight=weight,
                )
            )
    identifiers = [item.candidate_id for item in result]
    if len(identifiers) != 11 or len(set(identifiers)) != len(identifiers):
        raise AssertionError("Step42C candidate grid changed")
    return tuple(result)


def get_candidate(candidate_id: str) -> CandidateSpec:
    matches = [item for item in candidate_specs() if item.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown Step42C candidate: {candidate_id}")
    return matches[0]


def source_subjects_for(target_subject: str) -> tuple[str, ...]:
    if target_subject == DIAGNOSTIC_SUBJECT:
        return PRIMARY_SUBJECTS
    if target_subject not in PRIMARY_SUBJECTS:
        raise ValueError(f"Unknown Step42C target subject: {target_subject}")
    result = tuple(subject for subject in PRIMARY_SUBJECTS if subject != target_subject)
    if target_subject in result or len(result) != len(PRIMARY_SUBJECTS) - 1:
        raise AssertionError("Step42C LOSO source scope changed")
    return result


def proxy_masks(manifest: pd.DataFrame, proxy: ProxySpec) -> tuple[np.ndarray, np.ndarray]:
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    classes = manifest["class"].astype(int).to_numpy()
    query = (exercise == proxy.held_exercise) & np.isin(classes, proxy.held_classes)
    fit = ~query
    if proxy.held_exercise == "B":
        fit = exercise == "A"
    if np.any(fit & query) or not fit.any() or not query.any():
        raise AssertionError("Invalid Step42C proxy mask")
    fit_a = classes[fit & (exercise == "A")]
    expected_a = set(range(1, 13)).difference(
        proxy.held_classes if proxy.held_exercise == "A" else ()
    )
    if set(fit_a) != expected_a:
        raise AssertionError("Step42C proxy changed its fit-side A classes")
    fit_b = set(classes[fit & (exercise == "B")])
    expected_b = {5, 6} if proxy.held_exercise == "A" else set()
    if fit_b != expected_b:
        raise AssertionError("Step42C proxy changed its fit-side B classes")
    if set(classes[query]) != set(proxy.held_classes):
        raise AssertionError("Step42C query class coverage changed")
    return fit, query


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("L2 normalization expects a matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def cosine_scores(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return _l2_normalize(query) @ _l2_normalize(prototypes).T


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _opaque_trials(
    query_manifest: pd.DataFrame, subject: str, proxy_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_keys = list(
        zip(
            query_manifest["exercise_code"].astype(str),
            query_manifest["class"].astype(int),
            query_manifest["repetition"].astype(int),
        )
    )
    unique_keys = sorted(set(raw_keys), key=lambda item: (item[0], item[1], item[2]))
    rng = np.random.default_rng(_stable_seed("opaque", subject, proxy_id))
    permuted = [unique_keys[index] for index in rng.permutation(len(unique_keys))]
    key_to_id = {key: f"q{index:03d}" for index, key in enumerate(permuted)}
    window_ids = np.asarray([key_to_id[key] for key in raw_keys], dtype="U8")
    ordered_ids = np.asarray(sorted(set(window_ids.tolist())), dtype="U8")
    id_to_index = {trial_id: index for index, trial_id in enumerate(ordered_ids)}
    window_trial_index = np.asarray([id_to_index[item] for item in window_ids], dtype=int)
    key_by_id = {trial_id: key for key, trial_id in key_to_id.items()}
    truth = np.asarray([key_by_id[item][1] for item in ordered_ids], dtype=int)
    repetitions = np.asarray([key_by_id[item][2] for item in ordered_ids], dtype=int)
    return ordered_ids, window_trial_index, truth, repetitions


def _group_mean(values: np.ndarray, group_index: np.ndarray, n_groups: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty((n_groups, values.shape[1]), dtype=np.float64)
    for group in range(n_groups):
        selected = group_index == group
        if not selected.any():
            raise AssertionError("Step42C produced an empty opaque trial")
        result[group] = values[selected].mean(axis=0)
    return result


def _pose_scaler(
    prepared: b3.PreparedSubject, fit_mask: np.ndarray
) -> common.ScalerState:
    manifest = prepared.manifest
    a_fit = fit_mask & manifest["exercise_code"].astype(str).eq("A").to_numpy()
    if not a_fit.any():
        raise AssertionError("Step42C pose coordinates lack target A rows")
    weights = common.equal_group_weights(manifest.loc[a_fit].reset_index(drop=True))
    scaler = StandardScaler().fit(prepared.glove_features[a_fit], sample_weight=weights)
    return common.ScalerState(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.maximum(np.asarray(scaler.scale_, dtype=np.float64), 1e-12),
    )


def _source_prototypes(
    sources: Mapping[str, b3.PreparedSubject],
    source_subjects: tuple[str, ...],
    proxy: ProxySpec,
) -> tuple[np.ndarray, pd.DataFrame]:
    subject_prototypes: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    for subject in source_subjects:
        prepared = sources[subject]
        fit, _ = proxy_masks(prepared.manifest, proxy)
        scaler = _pose_scaler(prepared, fit)
        pose = common.transform(prepared.glove_features, scaler)
        exercise = prepared.manifest["exercise_code"].astype(str).to_numpy()
        classes = prepared.manifest["class"].astype(int).to_numpy()
        repetitions = prepared.manifest["repetition"].astype(int).to_numpy()
        class_prototypes: list[np.ndarray] = []
        for class_id in proxy.held_classes:
            repetition_means: list[np.ndarray] = []
            for repetition in REPETITIONS:
                rows = (
                    (exercise == proxy.held_exercise)
                    & (classes == class_id)
                    & (repetitions == repetition)
                )
                if not rows.any():
                    raise AssertionError(
                        f"Missing Step42C source anchor {subject}/{proxy.proxy_id}/"
                        f"{class_id}/rep{repetition}"
                    )
                repetition_means.append(pose[rows].mean(axis=0))
                audit_rows.append(
                    {
                        "proxy": proxy.proxy_id,
                        "source_subject": subject,
                        "exercise_code": proxy.held_exercise,
                        "class": class_id,
                        "repetition": repetition,
                        "n_windows": int(rows.sum()),
                    }
                )
            class_prototypes.append(np.mean(repetition_means, axis=0))
        subject_prototypes.append(np.vstack(class_prototypes))
    prototypes = np.mean(subject_prototypes, axis=0)
    if prototypes.shape != (len(proxy.held_classes), 66) or not np.isfinite(prototypes).all():
        raise AssertionError("Step42C source prototypes are invalid")
    return prototypes, pd.DataFrame(audit_rows)


def prepare_episode(
    target: b3.PreparedSubject,
    sources: Mapping[str, b3.PreparedSubject],
    proxy: ProxySpec,
) -> PreparedEpisode:
    fit, query = proxy_masks(target.manifest, proxy)
    fit_manifest = target.manifest.loc[fit].reset_index(drop=True)
    query_manifest = target.manifest.loc[query].reset_index(drop=True)
    weights = common.equal_group_weights(fit_manifest)
    ridge = common.fit_weighted_ridge(
        target.emg_features[fit],
        target.glove_features[fit],
        target.emg_features[query],
        weights,
    )
    target_pose_scaler = _pose_scaler(target, fit)
    predicted_pose_window = common.transform(ridge.prediction, target_pose_scaler)

    trial_ids, window_trial_index, trial_truth, trial_repetitions = _opaque_trials(
        query_manifest, target.subject, proxy.proxy_id
    )
    n_trials = len(trial_ids)
    expected_trials = len(proxy.held_classes) * len(REPETITIONS)
    if n_trials != expected_trials:
        raise AssertionError(
            f"Step42C expected {expected_trials} trials, observed {n_trials}"
        )
    counts = pd.Series(trial_truth).value_counts().to_dict()
    if set(counts) != set(proxy.held_classes) or set(counts.values()) != {6}:
        raise AssertionError("Step42C query is not six trials per class")
    predicted_pose_trial = _group_mean(
        predicted_pose_window, window_trial_index, n_trials
    )

    emg_trial_views: dict[str, np.ndarray] = {}
    for feature_set, columns in FEATURE_SETS.items():
        scaler = StandardScaler().fit(
            target.emg_features[fit][:, columns], sample_weight=weights
        )
        query_scaled = scaler.transform(target.emg_features[query][:, columns])
        emg_trial_views[feature_set] = _l2_normalize(
            _group_mean(query_scaled, window_trial_index, n_trials)
        )

    source_subjects = source_subjects_for(target.subject)
    prototypes, audit = _source_prototypes(sources, source_subjects, proxy)
    audit.insert(0, "target_subject", target.subject)
    query_row_keys = query_manifest["step41_cache_row_index"].to_numpy(dtype=np.int64)
    unlabeled = UnlabeledEpisode(
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
    truth = EpisodeTruth(
        trial_truth=trial_truth,
        window_truth=query_manifest["class"].to_numpy(dtype=int),
        trial_repetitions=trial_repetitions,
    )
    return PreparedEpisode(unlabeled=unlabeled, truth=truth, source_anchor_audit=audit)


def exact_capacity_assignment(scores: np.ndarray, capacity: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or capacity <= 0:
        raise ValueError("Invalid exact-capacity assignment inputs")
    n_rows, n_classes = scores.shape
    if n_rows != n_classes * capacity or not np.isfinite(scores).all():
        raise ValueError("Rows must equal classes times capacity")
    slot_classes = np.repeat(np.arange(n_classes, dtype=int), capacity)
    costs = -scores[:, slot_classes]
    row, column = linear_sum_assignment(costs)
    assignments = np.empty(n_rows, dtype=int)
    assignments[row] = slot_classes[column]
    if set(np.bincount(assignments, minlength=n_classes).tolist()) != {capacity}:
        raise AssertionError("Exact-capacity assignment violated class counts")
    return assignments


def _kmeans_plus_plus(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    first = int(rng.integers(n))
    selected = [first]
    closest = np.sum(np.square(values - values[first]), axis=1)
    while len(selected) < k:
        probabilities = closest.copy()
        probabilities[selected] = 0.0
        total = float(probabilities.sum())
        if total <= 1e-15:
            index = next(item for item in range(n) if item not in selected)
        else:
            index = int(rng.choice(n, p=probabilities / total))
            if index in selected:
                index = next(item for item in range(n) if item not in selected)
        selected.append(index)
        distance = np.sum(np.square(values - values[index]), axis=1)
        closest = np.minimum(closest, distance)
    return values[np.asarray(selected, dtype=int)].copy()


def balanced_kmeans(
    values: np.ndarray,
    k: int,
    capacity: int,
    seed: int,
    starts: int = KMEANS_STARTS,
    max_iter: int = KMEANS_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) != k * capacity or starts <= 0 or max_iter <= 0:
        raise ValueError("Invalid balanced-kmeans inputs")
    best: tuple[float, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    for start in range(starts):
        rng = np.random.default_rng(seed + 104_729 * start)
        centers = _kmeans_plus_plus(values, k, rng)
        previous: np.ndarray | None = None
        for _ in range(max_iter):
            distances = np.sum(
                np.square(values[:, None, :] - centers[None, :, :]), axis=2
            )
            assignments = exact_capacity_assignment(-distances, capacity)
            centers = np.vstack(
                [values[assignments == cluster].mean(axis=0) for cluster in range(k)]
            )
            if previous is not None and np.array_equal(assignments, previous):
                break
            previous = assignments.copy()
        final_distance = np.sum(
            np.square(values - centers[assignments]), axis=1
        )
        objective = float(final_distance.sum())
        key = (objective, tuple(assignments.tolist()), assignments.copy(), centers.copy())
        if best is None or key[:2] < best[:2]:
            best = key
    if best is None:
        raise AssertionError("Balanced k-means produced no solution")
    objective, _, assignments, centers = best
    if set(np.bincount(assignments, minlength=k).tolist()) != {capacity}:
        raise AssertionError("Balanced k-means violated capacity")
    return assignments, centers, objective


def _semantic_permutation(
    prototypes: np.ndarray, permutation: np.ndarray | None
) -> np.ndarray:
    if permutation is None:
        return prototypes
    permutation = np.asarray(permutation, dtype=int)
    if set(permutation.tolist()) != set(range(len(prototypes))):
        raise ValueError("Invalid Step42C semantic permutation")
    return prototypes[permutation]


def predict_candidate(
    episode: UnlabeledEpisode,
    candidate: CandidateSpec,
    semantic_permutation: np.ndarray | None = None,
    cluster_feature_permutation: np.ndarray | None = None,
    fixed_cluster_assignments: np.ndarray | None = None,
    fixed_cluster_sse: float | None = None,
) -> CandidatePrediction:
    prototypes = _semantic_permutation(episode.source_prototypes, semantic_permutation)
    pose_scores = cosine_scores(episode.predicted_pose_trial, prototypes)
    n_trials, k = pose_scores.shape
    if n_trials % k:
        raise AssertionError("Step42C trial count is not divisible by class count")
    capacity = n_trials // k
    cluster_assignments: np.ndarray | None = None
    cluster_sse: float | None = None

    if candidate.algorithm == "independent_pose":
        class_indices = np.argmax(pose_scores, axis=1)
    elif candidate.algorithm == "balanced_pose":
        class_indices = exact_capacity_assignment(pose_scores, capacity)
    elif candidate.algorithm == "bcn":
        if candidate.feature_set not in episode.emg_trial_views:
            raise AssertionError("BCN candidate lacks an EMG view")
        if candidate.pose_view_weight is None or candidate.pose_view_weight < 0:
            raise AssertionError("BCN candidate lacks a valid pose-view weight")
        emg_view = episode.emg_trial_views[candidate.feature_set]
        pose_view = _l2_normalize(episode.predicted_pose_trial)
        if candidate.pose_view_weight == 0:
            cluster_view = emg_view
        else:
            cluster_view = np.concatenate(
                (emg_view, np.sqrt(candidate.pose_view_weight) * pose_view), axis=1
            ) / np.sqrt(1.0 + candidate.pose_view_weight)
        if cluster_feature_permutation is not None:
            cluster_feature_permutation = np.asarray(cluster_feature_permutation, dtype=int)
            if set(cluster_feature_permutation.tolist()) != set(range(n_trials)):
                raise ValueError("Invalid Step42C cluster-feature permutation")
            cluster_view = cluster_view[cluster_feature_permutation]
        if fixed_cluster_assignments is None:
            cluster_assignments, _, cluster_sse = balanced_kmeans(
                cluster_view,
                k=k,
                capacity=capacity,
                seed=_stable_seed(
                    "bcn", episode.subject, episode.proxy_id, candidate.candidate_id
                ),
            )
        else:
            if cluster_feature_permutation is not None:
                raise ValueError(
                    "Fixed clusters cannot be combined with a cluster-feature permutation"
                )
            cluster_assignments = np.asarray(fixed_cluster_assignments, dtype=int)
            if cluster_assignments.shape != (n_trials,) or set(
                np.bincount(cluster_assignments, minlength=k).tolist()
            ) != {capacity}:
                raise ValueError("Invalid fixed Step42C cluster assignments")
            cluster_sse = fixed_cluster_sse
        naming_scores = np.vstack(
            [
                pose_scores[cluster_assignments == cluster].mean(axis=0)
                for cluster in range(k)
            ]
        )
        cluster_rows, class_columns = linear_sum_assignment(-naming_scores)
        cluster_to_class = np.empty(k, dtype=int)
        cluster_to_class[cluster_rows] = class_columns
        class_indices = cluster_to_class[cluster_assignments]
    else:
        raise ValueError(f"Unknown Step42C algorithm: {candidate.algorithm}")

    predictions = episode.catalog[class_indices]
    if predictions.shape != (n_trials,) or not np.isin(predictions, episode.catalog).all():
        raise AssertionError("Step42C produced invalid trial predictions")
    return CandidatePrediction(
        predictions=predictions,
        pose_scores=pose_scores,
        cluster_assignments=cluster_assignments,
        cluster_sse=cluster_sse,
    )


def _balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    recalls = [
        float(np.mean(prediction[truth == label] == label))
        for label in sorted(set(truth.tolist()))
    ]
    return float(np.mean(recalls))


def cluster_oracle_accuracy(assignments: np.ndarray, truth: np.ndarray) -> float:
    assignments = np.asarray(assignments, dtype=int)
    truth = np.asarray(truth, dtype=int)
    clusters = sorted(set(assignments.tolist()))
    labels = sorted(set(truth.tolist()))
    if len(clusters) != len(labels):
        raise ValueError("Cluster oracle requires equal cluster and class counts")
    counts = np.asarray(
        [
            [np.sum((assignments == cluster) & (truth == label)) for label in labels]
            for cluster in clusters
        ],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(-counts)
    return float(counts[rows, columns].sum() / len(truth))


def evaluate_prediction(
    prepared: PreparedEpisode,
    candidate: CandidateSpec,
    prediction: CandidatePrediction,
    control_type: str = "none",
    control_index: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    episode = prepared.unlabeled
    truth = prepared.truth
    trial_prediction = prediction.predictions
    window_prediction = trial_prediction[episode.window_trial_index]
    class_to_index = {value: index for index, value in enumerate(episode.catalog.tolist())}
    true_indices = np.asarray([class_to_index[value] for value in truth.trial_truth], dtype=int)
    true_scores = prediction.pose_scores[np.arange(len(true_indices)), true_indices]
    other_scores = prediction.pose_scores.copy()
    other_scores[np.arange(len(true_indices)), true_indices] = -np.inf
    oracle = (
        cluster_oracle_accuracy(prediction.cluster_assignments, truth.trial_truth)
        if prediction.cluster_assignments is not None
        else np.nan
    )
    metrics: dict[str, object] = {
        "subject": episode.subject,
        "proxy": episode.proxy_id,
        "candidate_id": candidate.candidate_id,
        "algorithm": candidate.algorithm,
        "feature_set": candidate.feature_set,
        "pose_view_weight": candidate.pose_view_weight,
        "control_type": control_type,
        "control_index": control_index,
        "n_fit_windows": episode.n_fit_windows,
        "n_query_windows": len(truth.window_truth),
        "n_query_trials": len(truth.trial_truth),
        "n_classes": len(episode.catalog),
        "trials_per_class": int(len(truth.trial_truth) // len(episode.catalog)),
        "ridge_alpha": episode.ridge_alpha,
        "source_subjects": ",".join(episode.source_subjects),
        "trial_balanced_accuracy": _balanced_accuracy(
            truth.trial_truth, trial_prediction
        ),
        "window_balanced_accuracy": _balanced_accuracy(
            truth.window_truth, window_prediction
        ),
        "mean_true_pose_margin": float(np.mean(true_scores - np.max(other_scores, axis=1))),
        "prediction_coverage": int(len(set(trial_prediction.tolist()))),
        "cluster_oracle_accuracy": oracle,
        "cluster_sse": prediction.cluster_sse,
    }
    rows = pd.DataFrame(
        {
            "subject": episode.subject,
            "proxy": episode.proxy_id,
            "candidate_id": candidate.candidate_id,
            "control_type": control_type,
            "control_index": control_index,
            "opaque_trial_id": episode.trial_ids,
            "repetition": truth.trial_repetitions,
            "truth": truth.trial_truth,
            "prediction": trial_prediction,
            "cluster": (
                prediction.cluster_assignments
                if prediction.cluster_assignments is not None
                else np.full(len(trial_prediction), -1, dtype=int)
            ),
        }
    )
    return metrics, rows


def subject_candidate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    a_rows = metrics[metrics["proxy"].str.startswith("A_")].copy()
    b_rows = metrics[metrics["proxy"].eq("B56")].copy()
    keys = ["subject", "candidate_id", "algorithm", "feature_set", "pose_view_weight"]
    a_summary = a_rows.groupby(keys, as_index=False, dropna=False).agg(
        a_trial_balanced_accuracy=("trial_balanced_accuracy", "mean"),
        a_window_balanced_accuracy=("window_balanced_accuracy", "mean"),
        a_mean_true_pose_margin=("mean_true_pose_margin", "mean"),
        a_cluster_oracle_accuracy=("cluster_oracle_accuracy", "mean"),
    )
    b_summary = b_rows.groupby(keys, as_index=False, dropna=False).agg(
        b56_trial_balanced_accuracy=("trial_balanced_accuracy", "mean"),
        b56_window_balanced_accuracy=("window_balanced_accuracy", "mean"),
        b56_cluster_oracle_accuracy=("cluster_oracle_accuracy", "mean"),
    )
    return a_summary.merge(b_summary, on=keys, how="inner", validate="one_to_one")


def outer_subject_select(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outer in SUBJECTS:
        development = (
            set(PRIMARY_SUBJECTS)
            if outer == DIAGNOSTIC_SUBJECT
            else set(PRIMARY_SUBJECTS).difference({outer})
        )
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
        if len(ranking) != len(candidate_specs()):
            raise AssertionError("Step42C outer selection lost a candidate")
        winner = ranking.iloc[0]
        rows.append(
            {
                "outer_subject": outer,
                "development_subjects": ",".join(
                    sorted(development, key=lambda value: int(value[1:]))
                ),
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


def deterministic_nonidentity_permutation(
    size: int, subject: str, proxy: str, index: int, kind: str
) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(kind, subject, proxy, index))
    identity = np.arange(size, dtype=int)
    for _ in range(100):
        candidate = rng.permutation(size).astype(int)
        if not np.array_equal(candidate, identity):
            return candidate
    raise RuntimeError("Could not construct a non-identity Step42C permutation")


def bootstrap_interval(
    values: np.ndarray, seed: int, draws: int = 20_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def promotion_gate(
    selected_metrics: pd.DataFrame,
    reference_metrics: pd.DataFrame,
    semantic_controls: pd.DataFrame,
    cluster_shuffle_controls: pd.DataFrame,
) -> dict[str, object]:
    selected = selected_metrics[selected_metrics["subject"].isin(PRIMARY_SUBJECTS)].copy()
    reference = reference_metrics[reference_metrics["subject"].isin(PRIMARY_SUBJECTS)].copy()
    semantic = semantic_controls[semantic_controls["subject"].isin(PRIMARY_SUBJECTS)].copy()
    selected_a = selected[selected["proxy"].str.startswith("A_")]
    reference_a = reference[reference["proxy"].str.startswith("A_")]
    semantic_a = semantic[semantic["proxy"].str.startswith("A_")]

    selected_subject = selected_a.groupby("subject")["trial_balanced_accuracy"].mean()
    reference_subject = reference_a.groupby("subject")["trial_balanced_accuracy"].mean()
    semantic_subject = semantic_a.groupby("subject")["trial_balanced_accuracy"].mean()
    selected_subject, reference_subject = selected_subject.align(reference_subject, join="inner")
    selected_subject, semantic_subject = selected_subject.align(semantic_subject, join="inner")
    if len(selected_subject) != len(PRIMARY_SUBJECTS):
        raise AssertionError("Step42C gate lost a primary subject")
    reference_gain = (selected_subject - reference_subject).to_numpy(dtype=float)
    semantic_gain = (selected_subject - semantic_subject).to_numpy(dtype=float)
    reference_ci = bootstrap_interval(reference_gain, seed=42_200)
    semantic_ci = bootstrap_interval(semantic_gain, seed=42_201)

    b56_mean = float(
        selected.loc[selected["proxy"].eq("B56"), "trial_balanced_accuracy"].mean()
    )
    selected_algorithms = selected.groupby("subject")["algorithm"].first()
    bcn_selected = selected_algorithms.eq("bcn")
    if bcn_selected.any():
        bcn_subjects = set(bcn_selected[bcn_selected].index)
        bcn_rows = selected_a[selected_a["subject"].isin(bcn_subjects)]
        cluster_oracle = float(bcn_rows["cluster_oracle_accuracy"].mean())
        shuffle_rows = cluster_shuffle_controls[
            cluster_shuffle_controls["subject"].isin(bcn_subjects)
            & cluster_shuffle_controls["proxy"].str.startswith("A_")
        ]
        shuffle_subject = shuffle_rows.groupby("subject")["trial_balanced_accuracy"].mean()
        bcn_subject_score = bcn_rows.groupby("subject")["trial_balanced_accuracy"].mean()
        bcn_subject_score, shuffle_subject = bcn_subject_score.align(
            shuffle_subject, join="inner"
        )
        shuffle_gain = float((bcn_subject_score - shuffle_subject).mean())
    else:
        cluster_oracle = np.nan
        shuffle_gain = np.nan

    finite_columns = [
        "trial_balanced_accuracy",
        "window_balanced_accuracy",
        "mean_true_pose_margin",
    ]
    checks = {
        "all_primary_selections_transductive": bool(
            selected_algorithms.isin(["balanced_pose", "bcn"]).all()
        ),
        "mean_A_trial_at_least_0p50": bool(selected_subject.mean() >= 0.50),
        "mean_B56_trial_at_least_0p70": bool(b56_mean >= 0.70),
        "mean_gain_over_independent_at_least_0p05": bool(reference_gain.mean() >= 0.05),
        "independent_gain_bootstrap_low_above_zero": bool(reference_ci[0] > 0.0),
        "independent_gain_sign_flip_p_below_0p05": bool(
            common.exact_one_sided_sign_flip_p(reference_gain) < 0.05
        ),
        "at_least_6_of_9_beat_independent": bool(np.sum(reference_gain > 0) >= 6),
        "mean_gain_over_semantic_permutation_at_least_0p15": bool(
            semantic_gain.mean() >= 0.15
        ),
        "semantic_gain_bootstrap_low_above_zero": bool(semantic_ci[0] > 0.0),
        "semantic_gain_sign_flip_p_below_0p05": bool(
            common.exact_one_sided_sign_flip_p(semantic_gain) < 0.05
        ),
        "at_least_6_of_9_beat_semantic_permutation": bool(
            np.sum(semantic_gain > 0) >= 6
        ),
        "selected_BCN_cluster_oracle_at_least_0p65_if_applicable": bool(
            not bcn_selected.any() or cluster_oracle >= 0.65
        ),
        "selected_BCN_gain_over_shuffle_at_least_0p03_if_applicable": bool(
            not bcn_selected.any() or shuffle_gain >= 0.03
        ),
        "all_metrics_finite": bool(
            np.isfinite(selected[finite_columns].to_numpy(dtype=float)).all()
            and np.isfinite(reference[finite_columns].to_numpy(dtype=float)).all()
            and np.isfinite(semantic[finite_columns].to_numpy(dtype=float)).all()
        ),
        "all_query_panels_have_six_trials_per_class": bool(
            selected["trials_per_class"].eq(6).all()
        ),
    }
    return {
        "promotion_pass": bool(all(checks.values())),
        "checks": checks,
        "mean_selected_A_trial_balanced_accuracy": float(selected_subject.mean()),
        "mean_selected_A_window_balanced_accuracy": float(
            selected_a["window_balanced_accuracy"].mean()
        ),
        "mean_selected_B56_trial_balanced_accuracy": b56_mean,
        "mean_independent_A_trial_balanced_accuracy": float(reference_subject.mean()),
        "mean_gain_over_independent": float(reference_gain.mean()),
        "gain_over_independent_bootstrap_95pct": list(reference_ci),
        "gain_over_independent_sign_flip_p": common.exact_one_sided_sign_flip_p(
            reference_gain
        ),
        "subjects_beating_independent": int(np.sum(reference_gain > 0)),
        "mean_semantic_permutation_A_trial_balanced_accuracy": float(
            semantic_subject.mean()
        ),
        "mean_gain_over_semantic_permutation": float(semantic_gain.mean()),
        "gain_over_semantic_permutation_bootstrap_95pct": list(semantic_ci),
        "gain_over_semantic_permutation_sign_flip_p": common.exact_one_sided_sign_flip_p(
            semantic_gain
        ),
        "subjects_beating_semantic_permutation": int(np.sum(semantic_gain > 0)),
        "n_subjects_selecting_BCN": int(bcn_selected.sum()),
        "selected_BCN_mean_cluster_oracle_accuracy": (
            None if not bcn_selected.any() else cluster_oracle
        ),
        "selected_BCN_mean_gain_over_cluster_shuffle": (
            None if not bcn_selected.any() else shuffle_gain
        ),
    }
