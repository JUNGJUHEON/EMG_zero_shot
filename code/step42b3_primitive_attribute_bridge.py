from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.linalg import solve_sylvester
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import step42b0_graded_primitive as common
import step42b0_legal_cache as legal


PRIMARY_SUBJECTS = common.PRIMARY_SUBJECTS
ATTRIBUTE_NAMES = (
    "I_F",
    "I_E",
    "M_F",
    "M_E",
    "R_F",
    "R_E",
    "L_F",
    "L_E",
    "T_AD",
    "T_AB",
    "T_F",
    "T_E",
)
ATTRIBUTE_INDEX = {name: index for index, name in enumerate(ATTRIBUTE_NAMES)}

B_MOVEMENT_NAMES = {
    1: "thumb up",
    2: "index/middle extended; others flexed; thumb across fingers",
    3: "ring/little flexed; others extended",
    4: "thumb opposing base of little finger",
    5: "fingers abducted/open",
    6: "fingers flexed together/fist",
    7: "pointing index; remaining digits flexed",
    8: "extended fingers adducted/joined",
}
B_ACTIVE_ATTRIBUTES = {
    1: ("I_F", "M_F", "R_F", "L_F", "T_E"),
    2: ("I_E", "M_E", "R_F", "L_F", "T_AD", "T_F"),
    3: ("I_E", "M_E", "R_F", "L_F", "T_E"),
    4: ("T_AD", "T_F"),
    5: ("I_E", "M_E", "R_E", "L_E", "T_AB"),
    6: ("I_F", "M_F", "R_F", "L_F", "T_AD", "T_F"),
    7: ("I_E", "M_F", "R_F", "L_F", "T_AD", "T_F"),
    8: ("I_E", "M_E", "R_E", "L_E", "T_AD"),
}

FEATURE_SETS = {
    "all128": np.arange(128, dtype=np.int64),
    "amplitude64": np.arange(64, dtype=np.int64),
    "mav16": np.arange(16, dtype=np.int64),
}
REGULARIZERS = (0.01, 1.0, 100.0)
PRIMARY_PROXIES = ("hold_B5", "hold_B6")
PROXIES = {
    "hold_B5": {"fit_b": (6,), "held_b": (5,), "catalog": (1, 2, 3, 4, 5, 7)},
    "hold_B6": {"fit_b": (5,), "held_b": (6,), "catalog": (1, 2, 3, 4, 6, 7)},
    "hold_B56": {"fit_b": (), "held_b": (5, 6), "catalog": (5, 6)},
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    feature_set: str
    regularizer: float
    second_regularizer: float | None = None


@dataclass(frozen=True)
class PreparedSubject:
    subject: str
    manifest: pd.DataFrame
    emg_features: np.ndarray
    glove_features: np.ndarray


@dataclass(frozen=True)
class ProxyEvaluation:
    metrics: dict[str, object]
    query_manifest: pd.DataFrame
    catalog: np.ndarray
    scores: np.ndarray
    predictions: np.ndarray


def class_key(exercise: str, class_id: int) -> str:
    return f"{str(exercise)}{int(class_id)}"


def signature_for_class(exercise: str, class_id: int) -> np.ndarray:
    exercise = str(exercise)
    class_id = int(class_id)
    result = np.zeros(len(ATTRIBUTE_NAMES), dtype=np.float64)
    if exercise == "A":
        if class_id not in range(1, 13):
            raise ValueError(f"Unknown Exercise-A class: {class_id}")
        result[class_id - 1] = 1.0
        return result
    if exercise == "B":
        if class_id not in B_ACTIVE_ATTRIBUTES:
            raise ValueError(f"Step42B3 defines only B1-B8, got B{class_id}")
        for name in B_ACTIVE_ATTRIBUTES[class_id]:
            result[ATTRIBUTE_INDEX[name]] = 1.0
        return result
    raise ValueError(f"Unknown exercise code: {exercise}")


def attribute_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_id in range(1, 13):
        signature = signature_for_class("A", class_id)
        rows.append(
            {
                "exercise_code": "A",
                "class": class_id,
                "class_key": class_key("A", class_id),
                "movement": ATTRIBUTE_NAMES[class_id - 1],
                "active_attributes": ATTRIBUTE_NAMES[class_id - 1],
                **dict(zip(ATTRIBUTE_NAMES, signature.astype(int))),
            }
        )
    for class_id in range(1, 9):
        signature = signature_for_class("B", class_id)
        rows.append(
            {
                "exercise_code": "B",
                "class": class_id,
                "class_key": class_key("B", class_id),
                "movement": B_MOVEMENT_NAMES[class_id],
                "active_attributes": " ".join(B_ACTIVE_ATTRIBUTES[class_id]),
                **dict(zip(ATTRIBUTE_NAMES, signature.astype(int))),
            }
        )
    return pd.DataFrame(rows)


def _regularizer_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def candidate_specs() -> tuple[CandidateSpec, ...]:
    result: list[CandidateSpec] = []
    for feature_set in FEATURE_SETS:
        for regularizer in REGULARIZERS:
            token = _regularizer_token(regularizer)
            for family in ("dap_ridge", "dap_logistic", "sae", "attribute_to_pose"):
                result.append(
                    CandidateSpec(
                        candidate_id=f"{family}__{feature_set}__reg_{token}",
                        family=family,
                        feature_set=feature_set,
                        regularizer=regularizer,
                    )
                )
        for gamma in REGULARIZERS:
            for lambda_ in REGULARIZERS:
                result.append(
                    CandidateSpec(
                        candidate_id=(
                            f"eszsl__{feature_set}__gamma_{_regularizer_token(gamma)}"
                            f"__lambda_{_regularizer_token(lambda_)}"
                        ),
                        family="eszsl",
                        feature_set=feature_set,
                        regularizer=gamma,
                        second_regularizer=lambda_,
                    )
                )
    ids = [candidate.candidate_id for candidate in result]
    if len(ids) != 63 or len(ids) != len(set(ids)):
        raise AssertionError("Step42B3 candidate grid changed")
    return tuple(result)


def fixed_reference_id() -> str:
    return "dap_ridge__all128__reg_1"


def get_candidate(candidate_id: str) -> CandidateSpec:
    matches = [item for item in candidate_specs() if item.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicated Step42B3 candidate: {candidate_id}")
    return matches[0]


def prepare_subject(cache: legal.LegalAmplitudeCache) -> PreparedSubject:
    manifest = cache.manifest.reset_index(drop=True).copy()
    exercises = set(manifest["exercise_code"].astype(str))
    b_classes = set(
        manifest.loc[manifest["exercise_code"].astype(str).eq("B"), "class"].astype(int)
    )
    a_classes = set(
        manifest.loc[manifest["exercise_code"].astype(str).eq("A"), "class"].astype(int)
    )
    if exercises != {"A", "B"} or a_classes != set(range(1, 13)) or b_classes != {5, 6}:
        raise AssertionError("Step42B3 received a cache outside A1-12+B5/B6")
    if len(manifest) != len(cache.emg_features) or len(manifest) != len(cache.glove_features):
        raise AssertionError("Step42B3 cache arrays and manifest are misaligned")
    return PreparedSubject(
        subject=cache.subject,
        manifest=manifest,
        emg_features=np.asarray(cache.emg_features, dtype=np.float64),
        glove_features=np.asarray(cache.glove_features, dtype=np.float64),
    )


def proxy_masks(manifest: pd.DataFrame, proxy: str) -> tuple[np.ndarray, np.ndarray]:
    if proxy not in PROXIES:
        raise ValueError(f"Unknown Step42B3 proxy: {proxy}")
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    classes = manifest["class"].astype(int).to_numpy()
    config = PROXIES[proxy]
    fit = (exercise == "A") | (
        (exercise == "B") & np.isin(classes, config["fit_b"])
    )
    held = (exercise == "B") & np.isin(classes, config["held_b"])
    if np.any(fit & held) or not fit.any() or not held.any():
        raise AssertionError("Invalid Step42B3 whole-class mask")
    if set(classes[fit & (exercise == "A")]) != set(range(1, 13)):
        raise AssertionError("Step42B3 proxy lost an Exercise-A primitive")
    if set(classes[fit & (exercise == "B")]) != set(config["fit_b"]):
        raise AssertionError("Step42B3 proxy fit B exposure changed")
    if set(classes[held]) != set(config["held_b"]):
        raise AssertionError("Step42B3 proxy held B exposure changed")
    return fit, held


def _qualified_labels(manifest: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [class_key(exercise, class_id) for exercise, class_id in zip(
            manifest["exercise_code"].astype(str), manifest["class"].astype(int)
        )],
        dtype="U4",
    )


def _ordered_fit_classes(manifest: pd.DataFrame) -> tuple[str, ...]:
    labels = set(_qualified_labels(manifest).tolist())
    expected = [f"A{index}" for index in range(1, 13)]
    expected.extend(key for key in ("B5", "B6") if key in labels)
    if set(expected) != labels:
        raise AssertionError("Unexpected Step42B3 fit classes")
    return tuple(expected)


def _true_fit_signatures(class_keys: Iterable[str]) -> np.ndarray:
    result = []
    for key in class_keys:
        result.append(signature_for_class(key[0], int(key[1:])))
    return np.vstack(result)


def semantic_assignment(
    class_keys: tuple[str, ...], permutation: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    signatures = _true_fit_signatures(class_keys)
    if permutation is not None:
        permutation = np.asarray(permutation, dtype=np.int64)
        if permutation.shape != (len(class_keys),) or set(permutation.tolist()) != set(
            range(len(class_keys))
        ):
            raise ValueError("Invalid Step42B3 semantic permutation")
        signatures = signatures[permutation]
    return {key: signature for key, signature in zip(class_keys, signatures)}


def deterministic_permutation(
    class_keys: tuple[str, ...], subject: str, proxy: str, index: int
) -> np.ndarray:
    subject_number = int(subject[1:])
    proxy_number = list(PROXIES).index(proxy) + 1
    rng = np.random.default_rng(423_000 + subject_number * 1_000 + proxy_number * 100 + index)
    identity = np.arange(len(class_keys), dtype=np.int64)
    for _ in range(100):
        candidate = rng.permutation(len(class_keys)).astype(np.int64)
        if not np.array_equal(candidate, identity):
            return candidate
    raise RuntimeError("Could not create a non-identity semantic permutation")


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _cosine_scores(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return _l2_normalize(query) @ _l2_normalize(prototypes).T


def _sample_targets(
    labels: np.ndarray, assignment: Mapping[str, np.ndarray]
) -> np.ndarray:
    return np.vstack([assignment[str(label)] for label in labels])


def _fit_dap_ridge(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    weights: np.ndarray,
    eval_x: np.ndarray,
    catalog_signatures: np.ndarray,
    regularizer: float,
) -> np.ndarray:
    model = Ridge(alpha=regularizer)
    model.fit(train_x, train_targets, sample_weight=weights)
    return _cosine_scores(model.predict(eval_x), catalog_signatures)


def _fit_dap_logistic(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    weights: np.ndarray,
    eval_x: np.ndarray,
    catalog_signatures: np.ndarray,
    regularizer: float,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for attribute in range(train_targets.shape[1]):
        target = train_targets[:, attribute].astype(int)
        if set(target.tolist()) != {0, 1}:
            raise AssertionError("Every Step42B3 attribute must have positive and negative fit rows")
        positive_mass = float(weights[target == 1].sum())
        negative_mass = float(weights[target == 0].sum())
        balanced = weights.copy()
        balanced[target == 1] *= weights.sum() / (2.0 * positive_mass)
        balanced[target == 0] *= weights.sum() / (2.0 * negative_mass)
        model = LogisticRegression(
            C=regularizer,
            solver="liblinear",
            max_iter=2_000,
            random_state=423,
        )
        model.fit(train_x, target, sample_weight=balanced)
        probabilities.append(model.predict_proba(eval_x)[:, 1])
    return _cosine_scores(np.column_stack(probabilities), catalog_signatures)


def _fit_eszsl(
    train_x: np.ndarray,
    labels: np.ndarray,
    class_keys: tuple[str, ...],
    assignment: Mapping[str, np.ndarray],
    weights: np.ndarray,
    eval_x: np.ndarray,
    catalog_signatures: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    x_train = _l2_normalize(train_x)
    x_eval = _l2_normalize(eval_x)
    signatures = np.vstack([assignment[key] for key in class_keys])
    class_to_index = {key: index for index, key in enumerate(class_keys)}
    label_indices = np.asarray([class_to_index[str(label)] for label in labels], dtype=int)
    signed = -np.ones((len(labels), len(class_keys)), dtype=np.float64)
    signed[np.arange(len(labels)), label_indices] = 1.0
    left = x_train.T @ (weights[:, None] * x_train) + gamma * np.eye(x_train.shape[1])
    right = signatures.T @ signatures + lambda_ * np.eye(signatures.shape[1])
    cross = x_train.T @ (weights[:, None] * signed) @ signatures
    compatibility = np.linalg.solve(left, cross)
    compatibility = np.linalg.solve(right.T, compatibility.T).T
    return x_eval @ compatibility @ catalog_signatures.T


def _fit_sae(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    weights: np.ndarray,
    eval_x: np.ndarray,
    catalog_signatures: np.ndarray,
    regularizer: float,
) -> np.ndarray:
    root_weight = np.sqrt(weights / np.mean(weights))[:, None]
    x = (train_x * root_weight).T
    targets = (train_targets * root_weight).T
    left = regularizer * (targets @ targets.T) + 1e-8 * np.eye(targets.shape[0])
    right = x @ x.T + 1e-8 * np.eye(x.shape[0])
    cross = (1.0 + regularizer) * targets @ x.T
    encoder = solve_sylvester(left, right, cross)
    return _cosine_scores(eval_x @ encoder.T, catalog_signatures)


def _fit_attribute_to_pose(
    train_x: np.ndarray,
    train_pose: np.ndarray,
    labels: np.ndarray,
    class_keys: tuple[str, ...],
    assignment: Mapping[str, np.ndarray],
    weights: np.ndarray,
    eval_x: np.ndarray,
    catalog_signatures: np.ndarray,
    regularizer: float,
) -> np.ndarray:
    emg_to_pose = Ridge(alpha=regularizer)
    emg_to_pose.fit(train_x, train_pose, sample_weight=weights)
    predicted_pose = emg_to_pose.predict(eval_x)
    class_pose = []
    for key in class_keys:
        selected = labels == key
        class_pose.append(np.average(train_pose[selected], axis=0, weights=weights[selected]))
    attribute_to_pose = Ridge(alpha=regularizer)
    attribute_to_pose.fit(
        np.vstack([assignment[key] for key in class_keys]), np.vstack(class_pose)
    )
    synthesized_pose = attribute_to_pose.predict(catalog_signatures)
    return _cosine_scores(predicted_pose, synthesized_pose)


def fit_candidate_scores(
    prepared: PreparedSubject,
    proxy: str,
    candidate: CandidateSpec,
    semantic_permutation: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    fit, held = proxy_masks(prepared.manifest, proxy)
    fit_manifest = prepared.manifest.loc[fit].reset_index(drop=True)
    query_manifest = prepared.manifest.loc[held].reset_index(drop=True)
    columns = FEATURE_SETS[candidate.feature_set]
    train_raw = prepared.emg_features[fit][:, columns]
    eval_raw = prepared.emg_features[held][:, columns]
    weights = common.equal_group_weights(fit_manifest)
    scaler = StandardScaler().fit(train_raw, sample_weight=weights)
    train_x = scaler.transform(train_raw)
    eval_x = scaler.transform(eval_raw)
    labels = _qualified_labels(fit_manifest)
    class_keys = _ordered_fit_classes(fit_manifest)
    assignment = semantic_assignment(class_keys, semantic_permutation)
    train_targets = _sample_targets(labels, assignment)
    catalog = np.asarray(PROXIES[proxy]["catalog"], dtype=np.int64)
    catalog_signatures = np.vstack([signature_for_class("B", item) for item in catalog])

    if candidate.family == "dap_ridge":
        scores = _fit_dap_ridge(
            train_x, train_targets, weights, eval_x, catalog_signatures, candidate.regularizer
        )
    elif candidate.family == "dap_logistic":
        scores = _fit_dap_logistic(
            train_x, train_targets, weights, eval_x, catalog_signatures, candidate.regularizer
        )
    elif candidate.family == "eszsl":
        if candidate.second_regularizer is None:
            raise AssertionError("ESZSL candidate lacks lambda")
        scores = _fit_eszsl(
            train_x,
            labels,
            class_keys,
            assignment,
            weights,
            eval_x,
            catalog_signatures,
            candidate.regularizer,
            candidate.second_regularizer,
        )
    elif candidate.family == "sae":
        scores = _fit_sae(
            train_x, train_targets, weights, eval_x, catalog_signatures, candidate.regularizer
        )
    elif candidate.family == "attribute_to_pose":
        a_rows = fit_manifest["exercise_code"].astype(str).eq("A").to_numpy()
        pose_weights = common.equal_group_weights(fit_manifest.loc[a_rows].reset_index(drop=True))
        pose_scaler = StandardScaler().fit(
            prepared.glove_features[fit][a_rows], sample_weight=pose_weights
        )
        train_pose = pose_scaler.transform(prepared.glove_features[fit])
        scores = _fit_attribute_to_pose(
            train_x,
            train_pose,
            labels,
            class_keys,
            assignment,
            weights,
            eval_x,
            catalog_signatures,
            candidate.regularizer,
        )
    else:
        raise ValueError(f"Unknown Step42B3 family: {candidate.family}")
    if scores.shape != (int(held.sum()), len(catalog)) or not np.isfinite(scores).all():
        raise AssertionError("Step42B3 produced invalid candidate scores")
    return scores, query_manifest, catalog


def _balanced_weighted_recall(
    truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    recalls = []
    for label in sorted(set(truth.tolist())):
        selected = truth == label
        recalls.append(float(np.average(prediction[selected] == truth[selected], weights=weights[selected])))
    return float(np.mean(recalls))


def evaluate_scores(
    prepared: PreparedSubject,
    proxy: str,
    candidate: CandidateSpec,
    scores: np.ndarray,
    query_manifest: pd.DataFrame,
    catalog: np.ndarray,
    control_index: int | None = None,
) -> ProxyEvaluation:
    truth = query_manifest["class"].to_numpy(dtype=np.int64)
    predictions = catalog[np.argmax(scores, axis=1)]
    weights = common.equal_group_weights(query_manifest)
    window_recall = _balanced_weighted_recall(truth, predictions, weights)
    true_indices = np.asarray([int(np.where(catalog == value)[0][0]) for value in truth])
    true_scores = scores[np.arange(len(scores)), true_indices]
    other_scores = scores.copy()
    other_scores[np.arange(len(scores)), true_indices] = -np.inf
    margins = true_scores - np.max(other_scores, axis=1)

    repetition_truth: list[int] = []
    repetition_prediction: list[int] = []
    repetition_rows: list[dict[str, object]] = []
    group_columns = ["class", "repetition"]
    for (class_id, repetition), indices in query_manifest.groupby(group_columns).groups.items():
        index = np.asarray(list(indices), dtype=np.int64)
        mean_scores = scores[index].mean(axis=0)
        prediction = int(catalog[int(np.argmax(mean_scores))])
        repetition_truth.append(int(class_id))
        repetition_prediction.append(prediction)
        repetition_rows.append(
            {"class": int(class_id), "repetition": int(repetition), "prediction": prediction}
        )
    repetition_recall = _balanced_weighted_recall(
        np.asarray(repetition_truth, dtype=np.int64),
        np.asarray(repetition_prediction, dtype=np.int64),
        np.ones(len(repetition_truth), dtype=np.float64),
    )
    fit, held = proxy_masks(prepared.manifest, proxy)
    metrics: dict[str, object] = {
        "subject": prepared.subject,
        "proxy": proxy,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "feature_set": candidate.feature_set,
        "regularizer": candidate.regularizer,
        "second_regularizer": candidate.second_regularizer,
        "control_index": control_index,
        "fit_B_classes": ",".join(map(str, PROXIES[proxy]["fit_b"])) or "none",
        "held_B_classes": ",".join(map(str, PROXIES[proxy]["held_b"])),
        "catalog_B_classes": ",".join(map(str, catalog.tolist())),
        "n_fit": int(fit.sum()),
        "n_query": int(held.sum()),
        "window_balanced_recall": window_recall,
        "repetition_balanced_recall": repetition_recall,
        "mean_true_margin": float(np.average(margins, weights=weights)),
        "prediction_coverage": int(len(set(predictions.tolist()))),
        "n_repetition_decisions": len(repetition_rows),
    }
    return ProxyEvaluation(
        metrics=metrics,
        query_manifest=query_manifest,
        catalog=catalog,
        scores=scores,
        predictions=predictions,
    )


def run_proxy_candidate(
    prepared: PreparedSubject,
    proxy: str,
    candidate: CandidateSpec,
    semantic_permutation: np.ndarray | None = None,
    control_index: int | None = None,
) -> ProxyEvaluation:
    scores, query_manifest, catalog = fit_candidate_scores(
        prepared, proxy, candidate, semantic_permutation
    )
    return evaluate_scores(
        prepared,
        proxy,
        candidate,
        scores,
        query_manifest,
        catalog,
        control_index=control_index,
    )


def summarize_candidate_subjects(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics["proxy"].isin(PRIMARY_PROXIES)].copy()
    keys = [
        "subject",
        "candidate_id",
        "family",
        "feature_set",
        "regularizer",
        "second_regularizer",
    ]
    return primary.groupby(keys, as_index=False, dropna=False).agg(
        repetition_balanced_recall=("repetition_balanced_recall", "mean"),
        window_balanced_recall=("window_balanced_recall", "mean"),
        mean_true_margin=("mean_true_margin", "mean"),
        prediction_coverage_min=("prediction_coverage", "min"),
    )


def outer_subject_select(subject_summary: pd.DataFrame) -> pd.DataFrame:
    primary_set = set(PRIMARY_SUBJECTS)
    rows: list[dict[str, object]] = []
    subjects = sorted(set(subject_summary["subject"]), key=lambda value: int(value[1:]))
    for outer in subjects:
        development = primary_set if outer == "s1" else primary_set.difference({outer})
        frame = subject_summary[subject_summary["subject"].isin(development)]
        ranking = (
            frame.groupby(
                [
                    "candidate_id",
                    "family",
                    "feature_set",
                    "regularizer",
                    "second_regularizer",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                development_repetition_recall=("repetition_balanced_recall", "mean"),
                development_window_recall=("window_balanced_recall", "mean"),
                development_margin=("mean_true_margin", "mean"),
                n_development_subjects=("subject", "nunique"),
            )
            .sort_values(
                [
                    "development_repetition_recall",
                    "development_window_recall",
                    "development_margin",
                    "candidate_id",
                ],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        if len(ranking) != len(candidate_specs()):
            raise AssertionError("Step42B3 outer selection did not see every candidate")
        winner = ranking.iloc[0]
        rows.append(
            {
                "outer_subject": outer,
                "development_subjects": ",".join(
                    sorted(development, key=lambda value: int(value[1:]))
                ),
                "selected_candidate_id": winner["candidate_id"],
                "selected_family": winner["family"],
                "selected_feature_set": winner["feature_set"],
                "selected_regularizer": winner["regularizer"],
                "selected_second_regularizer": winner["second_regularizer"],
                "development_repetition_recall": winner[
                    "development_repetition_recall"
                ],
                "development_window_recall": winner["development_window_recall"],
                "development_margin": winner["development_margin"],
                "n_development_subjects": int(winner["n_development_subjects"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def branch_gate(
    selected_metrics: pd.DataFrame,
    permutation_metrics: pd.DataFrame,
) -> dict[str, object]:
    selected_primary = selected_metrics[
        selected_metrics["subject"].isin(PRIMARY_SUBJECTS)
        & selected_metrics["proxy"].isin(PRIMARY_PROXIES)
    ]
    controls_primary = permutation_metrics[
        permutation_metrics["subject"].isin(PRIMARY_SUBJECTS)
        & permutation_metrics["proxy"].isin(PRIMARY_PROXIES)
    ]
    selected_subject = selected_primary.groupby("subject", as_index=True).agg(
        repetition=("repetition_balanced_recall", "mean"),
        window=("window_balanced_recall", "mean"),
    )
    control_subject = controls_primary.groupby("subject", as_index=True).agg(
        repetition=("repetition_balanced_recall", "mean"),
        window=("window_balanced_recall", "mean"),
    )
    selected_subject, control_subject = selected_subject.align(
        control_subject, join="inner", axis=0
    )
    if len(selected_subject) != len(PRIMARY_SUBJECTS):
        raise AssertionError("Step42B3 gate lost primary subjects")
    difference = (
        selected_subject["repetition"].to_numpy()
        - control_subject["repetition"].to_numpy()
    )
    low, high = bootstrap_interval(difference, seed=4233)
    proxy_recalls = {
        proxy: float(
            selected_primary.loc[
                selected_primary["proxy"].eq(proxy), "window_balanced_recall"
            ].mean()
        )
        for proxy in PRIMARY_PROXIES
    }
    checks = {
        "mean_repetition_recall_at_least_0p35": bool(
            selected_subject["repetition"].mean() >= 0.35
        ),
        "mean_window_recall_at_least_0p35": bool(
            selected_subject["window"].mean() >= 0.35
        ),
        "hold_B5_recall_at_least_0p25": bool(proxy_recalls["hold_B5"] >= 0.25),
        "hold_B6_recall_at_least_0p25": bool(proxy_recalls["hold_B6"] >= 0.25),
        "permutation_gain_at_least_0p10": bool(difference.mean() >= 0.10),
        "permutation_gain_bootstrap_low_above_zero": bool(low > 0.0),
        "permutation_gain_sign_flip_p_below_0p05": bool(
            common.exact_one_sided_sign_flip_p(difference) < 0.05
        ),
        "at_least_6_of_9_beat_permutation_mean": bool(np.sum(difference > 0) >= 6),
        "all_metrics_finite": bool(
            np.isfinite(
                selected_primary[
                    [
                        "repetition_balanced_recall",
                        "window_balanced_recall",
                        "mean_true_margin",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            and np.isfinite(
                controls_primary[
                    [
                        "repetition_balanced_recall",
                        "window_balanced_recall",
                        "mean_true_margin",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
    }
    return {
        "promotion_pass": bool(all(checks.values())),
        "checks": checks,
        "chance": 1.0 / 6.0,
        "mean_selected_repetition_recall": float(selected_subject["repetition"].mean()),
        "mean_selected_window_recall": float(selected_subject["window"].mean()),
        "proxy_window_recalls": proxy_recalls,
        "mean_permuted_repetition_recall": float(control_subject["repetition"].mean()),
        "mean_repetition_gain_over_permutation": float(difference.mean()),
        "gain_bootstrap_95pct": [low, high],
        "gain_exact_one_sided_sign_flip_p": common.exact_one_sided_sign_flip_p(
            difference
        ),
        "subjects_beating_permutation": int(np.sum(difference > 0)),
    }
