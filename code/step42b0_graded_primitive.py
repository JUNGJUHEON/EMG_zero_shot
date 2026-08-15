from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import step42b0_legal_cache as legal


PRIMARY_SUBJECTS = tuple(f"s{index}" for index in range(2, 11))
DIAGNOSTIC_SUBJECT = "s1"
FOLDS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "r12": ((3, 4, 5, 6), (1, 2)),
    "r34": ((1, 2, 5, 6), (3, 4)),
    "r56": ((1, 2, 3, 4), (5, 6)),
}
RIDGE_ALPHAS = np.logspace(-2, 4, 13)
NNLS_LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
A_CLASSES = tuple(range(1, 13))
B56_CLASSES = (5, 6)
METHODS = ("direct_ridge", "hard_primitive", "graded_primitive")
CONTROL_METHODS = (
    "direct_A_only",
    "hard_A_only",
    "graded_A_only",
    "claude_onehot_A_only",
    "permuted_primitive",
    "zero_input_graded",
)
ORACLES = ("hard_pose_oracle", "graded_pose_oracle")
PREDICTION_METHODS = METHODS + CONTROL_METHODS + ORACLES
FINGER_JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "thumb": (0, 1, 2, 3),
    "index": (4, 5, 6, 7),
    "middle": (7, 8, 9, 10, 11),
    "ring": (11, 12, 13, 14, 15),
    "little": (15, 16, 17, 18),
    "global": (19, 20, 21),
}


@dataclass(frozen=True)
class ScalerState:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class RidgePrediction:
    prediction: np.ndarray
    alpha: float
    coefficients: np.ndarray
    intercept: np.ndarray
    x_scaler: ScalerState
    y_scaler: ScalerState


@dataclass(frozen=True)
class PoseContext:
    pose: np.ndarray
    reference: np.ndarray
    dictionary: np.ndarray
    scaler: ScalerState


@dataclass(frozen=True)
class FoldResult:
    metrics: dict[str, object]
    inner_scores: list[dict[str, object]]
    validation_manifest: pd.DataFrame
    true_pose: np.ndarray
    predictions: dict[str, np.ndarray]
    reference: np.ndarray
    dictionary: np.ndarray


def equal_group_weights(manifest: pd.DataFrame) -> np.ndarray:
    if manifest.empty:
        raise ValueError("Cannot weight an empty manifest")
    columns = ["subject", "exercise_code", "class", "repetition"]
    keys = manifest[columns].astype(str).agg("|".join, axis=1)
    counts = keys.map(keys.value_counts()).to_numpy(dtype=np.float64)
    weights = 1.0 / counts
    weights /= weights.mean()
    group_sums = pd.Series(weights).groupby(keys.reset_index(drop=True)).sum().to_numpy()
    if not np.allclose(group_sums, group_sums[0], rtol=1e-12, atol=1e-12):
        raise AssertionError("Fit groups do not have equal total weight")
    return weights


def masks_for_fold(manifest: pd.DataFrame, fold_id: str) -> tuple[np.ndarray, np.ndarray]:
    if fold_id not in FOLDS:
        raise ValueError(f"Unknown Step42B0 fold: {fold_id}")
    fit_reps, validation_reps = FOLDS[fold_id]
    repetitions = manifest["repetition"].astype(int).to_numpy()
    fit = np.isin(repetitions, fit_reps)
    validation = np.isin(repetitions, validation_reps)
    if np.any(fit & validation) or not fit.any() or not validation.any():
        raise AssertionError("Invalid Step42B0 repetition partition")
    if not np.all(fit | validation):
        raise AssertionError("Step42B0 physical cache contains an unassigned repetition")
    return fit, validation


def _state(scaler: StandardScaler) -> ScalerState:
    return ScalerState(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.maximum(np.asarray(scaler.scale_, dtype=np.float64), 1e-12),
    )


def transform(values: np.ndarray, scaler: ScalerState) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - scaler.mean) / scaler.scale


def fit_weighted_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluate: np.ndarray,
    sample_weights: np.ndarray,
) -> RidgePrediction:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_evaluate = np.asarray(x_evaluate, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if len(x_train) != len(y_train) or len(x_train) != len(sample_weights):
        raise ValueError("Weighted Ridge inputs are not row aligned")
    x_scaler = StandardScaler().fit(x_train, sample_weight=sample_weights)
    y_scaler = StandardScaler().fit(y_train, sample_weight=sample_weights)
    x_scaled = x_scaler.transform(x_train)
    y_scaled = y_scaler.transform(y_train)
    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(x_scaled, y_scaled, sample_weight=sample_weights)
    prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_evaluate)))
    return RidgePrediction(
        prediction=np.asarray(prediction, dtype=np.float64),
        alpha=float(model.alpha_),
        coefficients=np.asarray(model.coef_, dtype=np.float64),
        intercept=np.asarray(model.intercept_, dtype=np.float64),
        x_scaler=_state(x_scaler),
        y_scaler=_state(y_scaler),
    )


def predict_weighted_ridge(model: RidgePrediction, x: np.ndarray) -> np.ndarray:
    x_scaled = transform(np.asarray(x, dtype=np.float64), model.x_scaler)
    prediction_scaled = x_scaled @ model.coefficients.T + model.intercept
    return prediction_scaled * model.y_scaler.scale + model.y_scaler.mean


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(np.asarray(values, dtype=np.float64), axis=0, weights=weights)


def build_pose_context(
    cache: legal.LegalAmplitudeCache, fit_mask: np.ndarray
) -> PoseContext:
    manifest = cache.manifest
    fit_mask = np.asarray(fit_mask, dtype=bool)
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    classes = manifest["class"].astype(int).to_numpy()
    a_fit = fit_mask & (exercise == "A")
    if set(classes[a_fit]) != set(A_CLASSES):
        raise AssertionError("Fold-local dictionary lost an Exercise-A atom")
    a_manifest = manifest.loc[a_fit].reset_index(drop=True)
    a_weights = equal_group_weights(a_manifest)
    scaler_sk = StandardScaler().fit(cache.glove_features[a_fit], sample_weight=a_weights)
    scaler = _state(scaler_sk)
    pose = transform(cache.glove_features, scaler)

    reference = _weighted_mean(pose[a_fit], a_weights)
    dictionary_rows = []
    for class_id in A_CLASSES:
        selected = a_fit & (classes == class_id)
        selected_manifest = manifest.loc[selected].reset_index(drop=True)
        weights = equal_group_weights(selected_manifest)
        dictionary_rows.append(_weighted_mean(pose[selected], weights) - reference)
    dictionary = np.vstack(dictionary_rows)
    if dictionary.shape != (12, 66) or not np.isfinite(dictionary).all():
        raise AssertionError("Invalid Step42B0 pose dictionary")
    return PoseContext(
        pose=pose,
        reference=reference,
        dictionary=dictionary,
        scaler=scaler,
    )


def nonnegative_coefficients(
    dictionary: np.ndarray,
    pose: np.ndarray,
    regularization: float,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    dictionary = np.asarray(dictionary, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    if dictionary.shape != (12, 66) or pose.ndim != 2 or pose.shape[1] != 66:
        raise ValueError("NNLS expects dictionary [12,66] and pose [N,66]")
    if regularization < 0:
        raise ValueError("NNLS regularization must be nonnegative")
    if reference is None:
        reference = np.zeros(66, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (66,):
        raise ValueError("NNLS reference must have shape [66]")
    centered_pose = pose - reference
    design = dictionary.T
    if regularization > 0:
        effective_regularization = regularization * 66.0 / 12.0
        design = np.vstack(
            (design, np.sqrt(effective_regularization) * np.eye(12))
        )
        padding = np.zeros(12, dtype=np.float64)
    coefficients = np.empty((len(pose), 12), dtype=np.float64)
    for index, target in enumerate(centered_pose):
        response = np.concatenate((target, padding)) if regularization > 0 else target
        coefficients[index], _ = nnls(design, response, maxiter=5000)
    if np.any(coefficients < -1e-12) or not np.isfinite(coefficients).all():
        raise AssertionError("NNLS produced invalid coefficients")
    return np.maximum(coefficients, 0.0)


def harden(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    result = np.zeros_like(coefficients)
    result[np.arange(len(result)), np.argmax(coefficients, axis=1)] = 1.0
    return result


def _subset_mask(manifest: pd.DataFrame, panel: str) -> np.ndarray:
    if panel == "all_legal":
        return np.ones(len(manifest), dtype=bool)
    if panel == "B56":
        exercise = manifest["exercise_code"].astype(str).to_numpy()
        classes = manifest["class"].astype(int).to_numpy()
        return (exercise == "B") & np.isin(classes, B56_CLASSES)
    raise ValueError(f"Unknown fit-side panel: {panel}")


def weighted_rmse(
    prediction: np.ndarray, truth: np.ndarray, manifest: pd.DataFrame
) -> float:
    weights = equal_group_weights(manifest)
    row_mse = np.mean(np.square(np.asarray(prediction) - np.asarray(truth)), axis=1)
    return float(np.sqrt(np.average(row_mse, weights=weights)))


def qualified_labels(manifest: pd.DataFrame) -> np.ndarray:
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    classes = manifest["class"].astype(int).to_numpy()
    return np.asarray([f"{ex}{class_id:02d}" for ex, class_id in zip(exercise, classes)])


def prototype_predict(
    query: np.ndarray,
    gallery: np.ndarray,
    gallery_manifest: pd.DataFrame,
) -> np.ndarray:
    labels = qualified_labels(gallery_manifest)
    classes = np.asarray(sorted(set(labels.tolist())))
    gallery_weights = equal_group_weights(gallery_manifest)
    prototypes = []
    for label in classes:
        selected = labels == label
        prototypes.append(_weighted_mean(gallery[selected], gallery_weights[selected]))
    prototypes_array = np.vstack(prototypes)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    proto_norm = np.linalg.norm(prototypes_array, axis=1, keepdims=True).T
    similarity = query @ prototypes_array.T
    similarity /= np.maximum(query_norm * proto_norm, 1e-12)
    return classes[np.argmax(similarity, axis=1)]


def classification_ba(
    prediction: np.ndarray,
    validation_manifest: pd.DataFrame,
    train_pose: np.ndarray,
    train_manifest: pd.DataFrame,
    panel: str,
) -> float:
    validation_panel = _subset_mask(validation_manifest, panel)
    train_panel = _subset_mask(train_manifest, panel)
    truth = qualified_labels(validation_manifest.loc[validation_panel].reset_index(drop=True))
    predicted = prototype_predict(
        prediction[validation_panel],
        train_pose[train_panel],
        train_manifest.loc[train_panel].reset_index(drop=True),
    )
    return float(balanced_accuracy_score(truth, predicted))


def classification_f1(
    prediction: np.ndarray,
    validation_manifest: pd.DataFrame,
    train_pose: np.ndarray,
    train_manifest: pd.DataFrame,
    panel: str,
) -> float:
    validation_panel = _subset_mask(validation_manifest, panel)
    train_panel = _subset_mask(train_manifest, panel)
    truth = qualified_labels(validation_manifest.loc[validation_panel].reset_index(drop=True))
    predicted = prototype_predict(
        prediction[validation_panel],
        train_pose[train_panel],
        train_manifest.loc[train_panel].reset_index(drop=True),
    )
    return float(f1_score(truth, predicted, average="macro"))


def reconstruct(context: PoseContext, coefficients: np.ndarray) -> np.ndarray:
    return context.reference + np.asarray(coefficients) @ context.dictionary


def permute_group_targets(
    targets: np.ndarray, manifest: pd.DataFrame, *, seed: int
) -> np.ndarray:
    columns = ["exercise_code", "class", "repetition"]
    keys = manifest[columns].astype(str).agg("|".join, axis=1).to_numpy()
    groups = np.asarray(sorted(set(keys.tolist())))
    rng = np.random.default_rng(seed)
    source_groups = groups[rng.permutation(len(groups))]
    means = {group: np.mean(targets[keys == group], axis=0) for group in groups}
    mapping = dict(zip(groups.tolist(), source_groups.tolist()))
    return np.vstack([means[mapping[key]] for key in keys])


def effective_support(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.maximum(np.asarray(coefficients, dtype=np.float64), 0.0)
    total = coefficients.sum(axis=1, keepdims=True)
    probabilities = coefficients / np.maximum(total, 1e-12)
    support = 1.0 / np.maximum(np.sum(np.square(probabilities), axis=1), 1e-12)
    return np.where(total[:, 0] > 1e-12, support, 0.0)


def _fit_graded_prediction(
    cache: legal.LegalAmplitudeCache,
    train_mask: np.ndarray,
    evaluate_mask: np.ndarray,
    context: PoseContext,
    regularization: float,
) -> tuple[np.ndarray, RidgePrediction, np.ndarray]:
    train_manifest = cache.manifest.loc[train_mask].reset_index(drop=True)
    weights = equal_group_weights(train_manifest)
    coefficients = nonnegative_coefficients(
        context.dictionary,
        context.pose[train_mask],
        regularization,
        context.reference,
    )
    fitted = fit_weighted_ridge(
        cache.emg_features[train_mask],
        coefficients,
        cache.emg_features[evaluate_mask],
        weights,
    )
    predicted_coefficients = np.maximum(fitted.prediction, 0.0)
    return reconstruct(context, predicted_coefficients), fitted, coefficients


def select_nnls_lambda(
    cache: legal.LegalAmplitudeCache, outer_fit: np.ndarray
) -> tuple[float, list[dict[str, object]]]:
    repetitions = cache.manifest["repetition"].astype(int).to_numpy()
    fit_repetitions = tuple(sorted(set(repetitions[outer_fit].tolist())))
    rows: list[dict[str, object]] = []
    for held_repetition in fit_repetitions:
        inner_validation = outer_fit & (repetitions == held_repetition)
        inner_fit = outer_fit & (repetitions != held_repetition)
        if not inner_fit.any() or not inner_validation.any():
            raise AssertionError("Invalid Step42B0 inner repetition split")
        context = build_pose_context(cache, inner_fit)
        validation_manifest = cache.manifest.loc[inner_validation].reset_index(drop=True)
        b56 = _subset_mask(validation_manifest, "B56")
        if not b56.any():
            raise AssertionError("Inner validation lost B5/B6")
        for regularization in NNLS_LAMBDAS:
            predicted, fitted, coefficients = _fit_graded_prediction(
                cache, inner_fit, inner_validation, context, regularization
            )
            score = weighted_rmse(
                predicted[b56],
                context.pose[inner_validation][b56],
                validation_manifest.loc[b56].reset_index(drop=True),
            )
            rows.append(
                {
                    "held_repetition": held_repetition,
                    "regularization": regularization,
                    "b56_rmse": score,
                    "ridge_alpha": fitted.alpha,
                    "coefficient_nonzero_fraction": float(np.mean(coefficients > 1e-8)),
                }
            )
    table = pd.DataFrame(rows)
    means = table.groupby("regularization", as_index=False)["b56_rmse"].mean()
    means = means.sort_values(
        ["b56_rmse", "regularization"], ascending=[True, False], kind="mergesort"
    )
    selected = float(means.iloc[0]["regularization"])
    for row in rows:
        row["selected"] = bool(float(row["regularization"]) == selected)
    return selected, rows


def _method_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    validation_manifest: pd.DataFrame,
    train_pose: np.ndarray,
    train_manifest: pd.DataFrame,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for panel in ("B56", "all_legal"):
        selected = _subset_mask(validation_manifest, panel)
        panel_manifest = validation_manifest.loc[selected].reset_index(drop=True)
        values[f"{panel}_rmse"] = weighted_rmse(
            prediction[selected], truth[selected], panel_manifest
        )
        values[f"{panel}_ba"] = classification_ba(
            prediction,
            validation_manifest,
            train_pose,
            train_manifest,
            panel,
        )
        values[f"{panel}_macro_f1"] = classification_f1(
            prediction,
            validation_manifest,
            train_pose,
            train_manifest,
            panel,
        )
    weights = equal_group_weights(validation_manifest)
    for name, joints in FINGER_JOINT_GROUPS.items():
        indices = tuple(joint + offset for offset in (0, 22, 44) for joint in joints)
        row_mse = np.mean(
            np.square(prediction[:, indices] - truth[:, indices]), axis=1
        )
        values[f"{name}_rmse"] = float(np.sqrt(np.average(row_mse, weights=weights)))
    return values


def run_fold(
    cache: legal.LegalAmplitudeCache, fold_id: str
) -> FoldResult:
    fit_mask, validation_mask = masks_for_fold(cache.manifest, fold_id)
    selected_lambda, inner_scores = select_nnls_lambda(cache, fit_mask)
    context = build_pose_context(cache, fit_mask)
    train_manifest = cache.manifest.loc[fit_mask].reset_index(drop=True)
    validation_manifest = cache.manifest.loc[validation_mask].reset_index(drop=True)
    train_weights = equal_group_weights(train_manifest)

    direct = fit_weighted_ridge(
        cache.emg_features[fit_mask],
        context.pose[fit_mask],
        cache.emg_features[validation_mask],
        train_weights,
    )
    train_coefficients = nonnegative_coefficients(
        context.dictionary,
        context.pose[fit_mask],
        selected_lambda,
        context.reference,
    )
    hard_targets = harden(train_coefficients)
    hard_model = fit_weighted_ridge(
        cache.emg_features[fit_mask],
        hard_targets,
        cache.emg_features[validation_mask],
        train_weights,
    )
    graded_model = fit_weighted_ridge(
        cache.emg_features[fit_mask],
        train_coefficients,
        cache.emg_features[validation_mask],
        train_weights,
    )
    validation_coefficients = nonnegative_coefficients(
        context.dictionary,
        context.pose[validation_mask],
        selected_lambda,
        context.reference,
    )
    exercise = cache.manifest["exercise_code"].astype(str).to_numpy()
    a_fit_mask = fit_mask & (exercise == "A")
    a_manifest = cache.manifest.loc[a_fit_mask].reset_index(drop=True)
    a_weights = equal_group_weights(a_manifest)
    a_coefficients = nonnegative_coefficients(
        context.dictionary,
        context.pose[a_fit_mask],
        selected_lambda,
        context.reference,
    )
    direct_a_model = fit_weighted_ridge(
        cache.emg_features[a_fit_mask],
        context.pose[a_fit_mask],
        cache.emg_features[validation_mask],
        a_weights,
    )
    hard_a_model = fit_weighted_ridge(
        cache.emg_features[a_fit_mask],
        harden(a_coefficients),
        cache.emg_features[validation_mask],
        a_weights,
    )
    graded_a_model = fit_weighted_ridge(
        cache.emg_features[a_fit_mask],
        a_coefficients,
        cache.emg_features[validation_mask],
        a_weights,
    )
    a_classes = cache.manifest.loc[a_fit_mask, "class"].astype(int).to_numpy()
    class_onehot = np.eye(12, dtype=np.float64)[a_classes - 1]
    claude_model = fit_weighted_ridge(
        cache.emg_features[a_fit_mask],
        class_onehot,
        cache.emg_features[validation_mask],
        a_weights,
    )
    permuted_targets = permute_group_targets(
        train_coefficients, train_manifest, seed=4200 + int(cache.subject[1:]) * 10 + list(FOLDS).index(fold_id)
    )
    permuted_model = fit_weighted_ridge(
        cache.emg_features[fit_mask],
        permuted_targets,
        cache.emg_features[validation_mask],
        train_weights,
    )
    zero_coefficients = np.maximum(
        predict_weighted_ridge(
            graded_model,
            np.zeros((int(validation_mask.sum()), cache.emg_features.shape[1])),
        ),
        0.0,
    )
    predictions = {
        "direct_ridge": direct.prediction,
        "hard_primitive": reconstruct(context, np.maximum(hard_model.prediction, 0.0)),
        "graded_primitive": reconstruct(context, np.maximum(graded_model.prediction, 0.0)),
        "direct_A_only": direct_a_model.prediction,
        "hard_A_only": reconstruct(context, np.maximum(hard_a_model.prediction, 0.0)),
        "graded_A_only": reconstruct(context, np.maximum(graded_a_model.prediction, 0.0)),
        "claude_onehot_A_only": reconstruct(context, np.maximum(claude_model.prediction, 0.0)),
        "permuted_primitive": reconstruct(context, np.maximum(permuted_model.prediction, 0.0)),
        "zero_input_graded": reconstruct(context, zero_coefficients),
        "hard_pose_oracle": reconstruct(context, harden(validation_coefficients)),
        "graded_pose_oracle": reconstruct(context, validation_coefficients),
    }
    truth = context.pose[validation_mask]
    train_pose = context.pose[fit_mask]
    metrics: dict[str, object] = {
        "subject": cache.subject,
        "fold": fold_id,
        "fit_repetitions": ",".join(map(str, FOLDS[fold_id][0])),
        "validation_repetitions": ",".join(map(str, FOLDS[fold_id][1])),
        "n_fit": int(fit_mask.sum()),
        "n_validation": int(validation_mask.sum()),
        "selected_nnls_lambda": selected_lambda,
        "dictionary_condition": float(np.linalg.cond(context.dictionary)),
        "graded_target_nonzero_fraction": float(np.mean(train_coefficients > 1e-8)),
        "B56_graded_target_effective_support": float(
            np.average(
                effective_support(train_coefficients[_subset_mask(train_manifest, "B56")]),
                weights=equal_group_weights(
                    train_manifest.loc[_subset_mask(train_manifest, "B56")].reset_index(drop=True)
                ),
            )
        ),
        "direct_ridge_alpha": direct.alpha,
        "hard_primitive_alpha": hard_model.alpha,
        "graded_primitive_alpha": graded_model.alpha,
        "permuted_primitive_alpha": permuted_model.alpha,
    }
    for method, prediction in predictions.items():
        for metric, value in _method_metrics(
            prediction,
            truth,
            validation_manifest,
            train_pose,
            train_manifest,
        ).items():
            metrics[f"{method}_{metric}"] = value
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise AssertionError("Step42B0 produced a non-finite prediction")
    return FoldResult(
        metrics=metrics,
        inner_scores=inner_scores,
        validation_manifest=validation_manifest,
        true_pose=truth,
        predictions=predictions,
        reference=context.reference,
        dictionary=context.dictionary,
    )


def subject_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in fold_metrics.columns
        if column.endswith("_rmse")
        or column.endswith("_ba")
        or column.endswith("_macro_f1")
        or column == "B56_graded_target_effective_support"
    ]
    return fold_metrics.groupby("subject", as_index=False)[columns].mean()


def bootstrap_lower_bound(
    differences: Iterable[float], *, seed: int = 4200, draws: int = 10000
) -> float:
    values = np.asarray(tuple(differences), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.05))


def exact_one_sided_sign_flip_p(differences: Iterable[float]) -> float:
    values = np.asarray(tuple(differences), dtype=np.float64)
    observed = float(values.mean())
    null = np.asarray(
        [np.mean(values * np.asarray(signs)) for signs in product((-1.0, 1.0), repeat=len(values))]
    )
    return float(np.mean(null >= observed - 1e-15))


def admission_gate(summary: pd.DataFrame) -> dict[str, object]:
    primary = summary[summary["subject"].isin(PRIMARY_SUBJECTS)].copy()
    if set(primary["subject"]) != set(PRIMARY_SUBJECTS):
        raise AssertionError("Step42B0 gate requires all primary subjects")
    direct = primary["direct_ridge_B56_rmse"].to_numpy(dtype=float)
    hard = primary["hard_primitive_B56_rmse"].to_numpy(dtype=float)
    graded = primary["graded_primitive_B56_rmse"].to_numpy(dtype=float)
    permuted = primary["permuted_primitive_B56_rmse"].to_numpy(dtype=float)
    zero = primary["zero_input_graded_B56_rmse"].to_numpy(dtype=float)
    hard_oracle = primary["hard_pose_oracle_B56_rmse"].to_numpy(dtype=float)
    oracle = primary["graded_pose_oracle_B56_rmse"].to_numpy(dtype=float)
    direct_difference = direct - graded
    hard_difference = hard - graded
    direct_relative = float((direct.mean() - graded.mean()) / direct.mean())
    hard_relative = float((hard.mean() - graded.mean()) / hard.mean())
    direct_lower = bootstrap_lower_bound(direct_difference, seed=4200)
    hard_lower = bootstrap_lower_bound(hard_difference, seed=4201)
    improved_subjects = int(np.sum(direct_difference > 0))
    direct_ba = float(primary["direct_ridge_B56_ba"].mean())
    graded_ba = float(primary["graded_primitive_B56_ba"].mean())
    direct_p = exact_one_sided_sign_flip_p(direct_difference)
    hard_p = exact_one_sided_sign_flip_p(hard_difference)
    hard_improved_subjects = int(np.sum(hard_difference > 0))
    oracle_relative = float((hard_oracle.mean() - oracle.mean()) / hard_oracle.mean())
    permuted_relative = float((permuted.mean() - graded.mean()) / permuted.mean())
    zero_relative = float((zero.mean() - graded.mean()) / zero.mean())
    effective_support_mean = float(
        primary["B56_graded_target_effective_support"].mean()
    )
    checks = {
        "relative_gain_vs_direct_at_least_0p01": direct_relative >= 0.01,
        "relative_gain_vs_hard_at_least_0p01": hard_relative >= 0.01,
        "bootstrap_lower_vs_direct_above_zero": direct_lower > 0.0,
        "bootstrap_lower_vs_hard_above_zero": hard_lower > 0.0,
        "at_least_6_of_9_improve_vs_direct": improved_subjects >= 6,
        "at_least_6_of_9_improve_vs_hard": hard_improved_subjects >= 6,
        "sign_flip_p_vs_direct_below_0p05": direct_p < 0.05,
        "sign_flip_p_vs_hard_below_0p05": hard_p < 0.05,
        "B56_ba_not_more_than_0p02_below_direct": graded_ba >= direct_ba - 0.02,
        "graded_oracle_better_than_direct": float(oracle.mean()) < float(direct.mean()),
        "graded_oracle_at_least_0p05_better_than_hard_oracle": oracle_relative >= 0.05,
        "B56_effective_support_above_1p25": effective_support_mean > 1.25,
        "real_pairing_at_least_0p05_better_than_permuted": permuted_relative >= 0.05,
        "real_pairing_at_least_0p05_better_than_zero_input": zero_relative >= 0.05,
    }
    return {
        "admit_graded_primitive": bool(all(checks.values())),
        "checks": checks,
        "primary_subjects": list(PRIMARY_SUBJECTS),
        "mean_B56_rmse": {
            "direct_ridge": float(direct.mean()),
            "hard_primitive": float(hard.mean()),
            "graded_primitive": float(graded.mean()),
            "permuted_primitive": float(permuted.mean()),
            "zero_input_graded": float(zero.mean()),
            "hard_pose_oracle": float(hard_oracle.mean()),
            "graded_pose_oracle": float(oracle.mean()),
        },
        "relative_rmse_gain": {
            "graded_vs_direct": direct_relative,
            "graded_vs_hard": hard_relative,
        },
        "bootstrap_95pct_lower_absolute_gain": {
            "graded_vs_direct": direct_lower,
            "graded_vs_hard": hard_lower,
        },
        "exact_one_sided_sign_flip_p": {
            "graded_vs_direct": direct_p,
            "graded_vs_hard": hard_p,
        },
        "subjects_improved_vs_direct": improved_subjects,
        "subjects_improved_vs_hard": hard_improved_subjects,
        "graded_oracle_relative_gain_vs_hard_oracle": oracle_relative,
        "graded_relative_gain_vs_controls": {
            "permuted_target": permuted_relative,
            "zero_input": zero_relative,
        },
        "mean_B56_graded_target_effective_support": effective_support_mean,
        "mean_B56_balanced_accuracy": {
            "direct_ridge": direct_ba,
            "graded_primitive": graded_ba,
        },
    }
