from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import step42b0_graded_primitive as b0
import step42b0_legal_cache as legal
import step42b3_primitive_attribute_bridge as b3
import step42c_balanced_transductive as step42c
import step44_exercise_conditioned_naming as step44
from project_paths import CAPSTONE_ROOT as CAPSTONE, CODE_DIR


RESULT_BASE = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step44_exercise_conditioned_naming"
)
MODEL_PACK_ROOT = RESULT_BASE / "legal_mixed_model_pack"
TRUSTED_PACK_ROOT = RESULT_BASE / "legal_mixed_trusted_audit"
PACK_VERIFICATION_ROOT = RESULT_BASE / "legal_mixed_pack_verification"
PREDICTION_ROOT = RESULT_BASE / "phase1_frozen_predictions"
PREDICTION_VERIFICATION_ROOT = RESULT_BASE / "phase1_prediction_verification"
EVALUATION_ROOT = RESULT_BASE / "phase2_development_evaluation"
FINAL_VERIFICATION_ROOT = RESULT_BASE / "phase2_evaluation_verification"
PROTOCOL = CODE_DIR / "STEP44_EXERCISE_CONDITIONED_NAMING_PROTOCOL.md"

STEP42C_GATE = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step42c_balanced_transductive"
    / "legal_proxy_gate"
    / "primary_s1_s10_20260722_v1"
)
PARENT_SELECTIONS = STEP42C_GATE / "outer_subject_selections.csv"
PARENT_VERIFICATION = STEP42C_GATE / "verification" / "verified_replay.json"
PARENT_MANIFEST = STEP42C_GATE / "artifact_manifest.json"
STEP43_PACK_VERIFICATION = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step43_semantic_landmark_naming"
    / "legal_landmark_pack_verification"
    / "primary_s1_s10_20260723_v1"
    / "verification.json"
)
STEP43_FINAL_VERIFICATION = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step43_semantic_landmark_naming"
    / "legal_proxy_gate_verification"
    / "primary_s1_s10_20260723_v1"
    / "verification.json"
)

PRIMARY_SUBJECTS = tuple(f"s{index}" for index in range(2, 11))
POSE_DIMENSION = 66
N_EPISODES = 60
N_CLASSES = 5
N_TRIALS = 30
N_LANDMARKS = 9
N_SOURCES = 8


@dataclass(frozen=True)
class MixedProxy:
    proxy_id: str
    held_a: tuple[int, int, int, int]
    seen_b: int
    held_b: int


@dataclass(frozen=True)
class ParentPolicy:
    subject: str
    candidate_id: str
    feature_set: str
    pose_view_weight: float
    development_subjects: tuple[str, ...]


@dataclass(frozen=True)
class TargetGeometry:
    trial_aliases: np.ndarray
    trial_slots: np.ndarray
    trial_repetitions: np.ndarray
    window_trial_index: np.ndarray
    query_row_keys: np.ndarray
    query_predicted_pose: np.ndarray
    cluster_view: np.ndarray
    target_landmarks: np.ndarray
    landmark_roles: np.ndarray
    ridge_alpha: float
    target_audit: pd.DataFrame
    crossfit_audit: pd.DataFrame


@dataclass(frozen=True)
class SourceGeometry:
    source_landmarks: np.ndarray
    source_anchors: np.ndarray
    anchor_roles: np.ndarray
    wrong_source_b: np.ndarray
    per_source_landmarks: np.ndarray
    per_source_anchors: np.ndarray
    source_subjects: tuple[str, ...]
    source_audit: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    def default(value: object) -> object:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def artifact_manifest(
    output: Path, *, excluded: set[str] | None = None
) -> dict[str, object]:
    excluded = excluded or set()
    records = []
    output = Path(output)
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            records.append(
                {
                    "name": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"files": records, "n_files": len(records)}


def ensure_new_output(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(f"Step44 immutable output already exists: {path}")
    Path(path).mkdir(parents=True, exist_ok=False)


def require_literal_bool(
    payload: Mapping[str, object], key: str, expected: bool
) -> None:
    value = payload.get(key)
    if type(value) is not bool or value is not expected:
        raise AssertionError(f"Step44 prerequisite {key} must be literal {expected}")


@lru_cache(maxsize=1)
def canonical_mixed_proxies() -> tuple[MixedProxy, ...]:
    keys: set[tuple[int, tuple[int, int, int, int]]] = set()
    a_proxies = [proxy for proxy in step42c.proxy_specs() if proxy.held_exercise == "A"]
    if len(a_proxies) != 12:
        raise AssertionError("Step44 parent A-proxy inventory changed")
    for parent in a_proxies:
        for removed in parent.held_classes:
            held_a = tuple(
                sorted(
                    class_id for class_id in parent.held_classes if class_id != removed
                )
            )
            if len(held_a) != 4:
                raise AssertionError("Step44 failed to construct a four-A catalog")
            for held_b in (5, 6):
                keys.add((held_b, held_a))
    if len(keys) != N_EPISODES:
        raise AssertionError(f"Step44 expected 60 unique proxies, observed {len(keys)}")
    result = []
    for index, (held_b, held_a) in enumerate(sorted(keys)):
        seen_b = 6 if held_b == 5 else 5
        result.append(
            MixedProxy(
                proxy_id=f"e{index:03d}",
                held_a=held_a,
                seen_b=seen_b,
                held_b=held_b,
            )
        )
    counts = {
        held_b: sum(proxy.held_b == held_b for proxy in result) for held_b in (5, 6)
    }
    if counts != {5: 30, 6: 30}:
        raise AssertionError(f"Step44 held-B directions are imbalanced: {counts}")
    return tuple(result)


def get_proxy(proxy_id: str) -> MixedProxy:
    matches = [
        proxy for proxy in canonical_mixed_proxies() if proxy.proxy_id == proxy_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown Step44 proxy: {proxy_id}")
    return matches[0]


def held_catalog(proxy: MixedProxy) -> tuple[tuple[str, int], ...]:
    result = tuple(("A", class_id) for class_id in proxy.held_a) + (
        ("B", proxy.held_b),
    )
    if len(result) != N_CLASSES or len(set(result)) != N_CLASSES:
        raise AssertionError("Step44 held catalog is invalid")
    return result


def fit_catalog(proxy: MixedProxy) -> tuple[tuple[str, int], ...]:
    result = tuple(
        ("A", class_id) for class_id in range(1, 13) if class_id not in proxy.held_a
    ) + (("B", proxy.seen_b),)
    if len(result) != N_LANDMARKS or len(set(result)) != N_LANDMARKS:
        raise AssertionError("Step44 fit landmark catalog is invalid")
    return result


def proxy_masks(
    manifest: pd.DataFrame, proxy: MixedProxy
) -> tuple[np.ndarray, np.ndarray]:
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    classes = manifest["class"].astype(int).to_numpy()
    fit = ((exercise == "A") & ~np.isin(classes, proxy.held_a)) | (
        (exercise == "B") & (classes == proxy.seen_b)
    )
    query = ((exercise == "A") & np.isin(classes, proxy.held_a)) | (
        (exercise == "B") & (classes == proxy.held_b)
    )
    if (
        np.any(fit & query)
        or not fit.any()
        or not query.any()
        or not np.all(fit | query)
    ):
        raise AssertionError("Step44 fit/query partition is invalid")
    observed_fit = set(zip(exercise[fit], classes[fit]))
    observed_query = set(zip(exercise[query], classes[query]))
    if observed_fit != set(fit_catalog(proxy)):
        raise AssertionError("Step44 fit catalog changed")
    if observed_query != set(held_catalog(proxy)):
        raise AssertionError("Step44 query catalog changed")
    return fit, query


def source_subjects_for(target_subject: str) -> tuple[str, ...]:
    if target_subject not in PRIMARY_SUBJECTS:
        raise ValueError(f"Step44 supports primary subjects only: {target_subject}")
    result = tuple(subject for subject in PRIMARY_SUBJECTS if subject != target_subject)
    if len(result) != N_SOURCES or target_subject in result:
        raise AssertionError("Step44 LOSO source scope changed")
    return result


@lru_cache(maxsize=1)
def require_verified_parent() -> dict[str, object]:
    for path in (
        PARENT_SELECTIONS,
        PARENT_VERIFICATION,
        PARENT_MANIFEST,
        STEP43_PACK_VERIFICATION,
        STEP43_FINAL_VERIFICATION,
        PROTOCOL,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step44 prerequisite: {path}")
    parent = read_json(PARENT_VERIFICATION)
    parent_manifest = read_json(PARENT_MANIFEST)
    step43_pack_verification = read_json(STEP43_PACK_VERIFICATION)
    step43_verification = read_json(STEP43_FINAL_VERIFICATION)
    if (
        not isinstance(parent, dict)
        or not isinstance(parent_manifest, dict)
        or not isinstance(step43_pack_verification, dict)
        or not isinstance(step43_verification, dict)
    ):
        raise AssertionError("Step44 prerequisite schema changed")
    require_literal_bool(parent, "verified", True)
    require_literal_bool(parent, "primary_or_forbidden_rows_loaded", False)
    require_literal_bool(step43_pack_verification, "verified", True)
    require_literal_bool(step43_pack_verification, "forbidden_exposure_absent", True)
    require_literal_bool(step43_verification, "verified", True)
    require_literal_bool(step43_verification, "forbidden_exposure_absent", True)
    if int(step43_pack_verification.get("b56_free_parent_selection_changes", -1)) != 0:
        raise AssertionError("Step44 parent selection is not B5/B6-free")
    manifest_records = {
        str(record["name"]): str(record["sha256"])
        for record in parent_manifest.get("files", [])
        if isinstance(record, dict) and "name" in record and "sha256" in record
    }
    if manifest_records.get(PARENT_SELECTIONS.name) != sha256_file(PARENT_SELECTIONS):
        raise AssertionError("Step44 parent selection changed after sealing")
    return {
        "step42c_parent_verification_sha256": sha256_file(PARENT_VERIFICATION),
        "step42c_parent_manifest_sha256": sha256_file(PARENT_MANIFEST),
        "step42c_selections_sha256": sha256_file(PARENT_SELECTIONS),
        "step43_pack_verification_sha256": sha256_file(STEP43_PACK_VERIFICATION),
        "step43_final_verification_sha256": sha256_file(STEP43_FINAL_VERIFICATION),
        "protocol_sha256": sha256_file(PROTOCOL),
    }


@lru_cache(maxsize=1)
def parent_policies() -> dict[str, ParentPolicy]:
    require_verified_parent()
    selections = pd.read_csv(PARENT_SELECTIONS)
    rows = selections[selections["outer_subject"].isin(PRIMARY_SUBJECTS)].copy()
    if (
        len(rows) != len(PRIMARY_SUBJECTS)
        or rows["outer_subject"].duplicated().any()
        or set(rows["outer_subject"]) != set(PRIMARY_SUBJECTS)
    ):
        raise AssertionError("Step44 parent policy table changed")
    result = {}
    for row in rows.itertuples(index=False):
        subject = str(row.outer_subject)
        feature_set = str(row.selected_feature_set)
        weight = float(row.selected_pose_view_weight)
        candidate_id = str(row.selected_candidate_id)
        expected = f"bcn__{feature_set}__posew_{str(weight).replace('.', 'p')}"
        if weight == 0.0:
            expected = f"bcn__{feature_set}__posew_0"
        if (
            candidate_id != expected
            or feature_set not in b3.FEATURE_SETS
            or weight not in step42c.POSE_VIEW_WEIGHTS
        ):
            raise AssertionError(f"Step44 parent policy is invalid for {subject}")
        development = tuple(str(row.development_subjects).split(","))
        if set(development) != set(source_subjects_for(subject)):
            raise AssertionError("Step44 parent development scope changed")
        result[subject] = ParentPolicy(
            subject=subject,
            candidate_id=candidate_id,
            feature_set=feature_set,
            pose_view_weight=weight,
            development_subjects=development,
        )
    return result


def load_legal_subjects() -> (
    tuple[dict[str, b3.PreparedSubject], list[dict[str, object]]]
):
    subjects: dict[str, b3.PreparedSubject] = {}
    records = []
    for subject in PRIMARY_SUBJECTS:
        cache = legal.load_legal_amplitude_cache(subject)
        subjects[subject] = b3.prepare_subject(cache)
        records.append(
            {
                "subject": subject,
                "npz_sha256": legal.sha256_file(cache.npz_path),
                "manifest_sha256": legal.sha256_file(cache.manifest_path),
                "metadata_sha256": legal.sha256_file(cache.metadata_path),
                "rows": len(cache.manifest),
            }
        )
    return subjects, records


def _scaler_state(scaler: StandardScaler) -> b0.ScalerState:
    return b0.ScalerState(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.maximum(np.asarray(scaler.scale_, dtype=np.float64), 1e-12),
    )


def _pose_scaler(prepared: b3.PreparedSubject, fit: np.ndarray) -> b0.ScalerState:
    manifest = prepared.manifest
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    a_fit = np.asarray(fit, dtype=bool) & (exercise == "A")
    weights = b0.equal_group_weights(manifest.loc[a_fit].reset_index(drop=True))
    scaler = StandardScaler().fit(prepared.glove_features[a_fit], sample_weight=weights)
    return _scaler_state(scaler)


def fixed_alpha_ridge_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluate: np.ndarray,
    sample_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_evaluate = np.asarray(x_evaluate, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if (
        len(x_train) != len(y_train)
        or len(x_train) != len(sample_weights)
        or x_train.ndim != 2
        or y_train.ndim != 2
        or x_evaluate.ndim != 2
        or x_train.shape[1] != x_evaluate.shape[1]
        or not np.isfinite(alpha)
        or alpha <= 0.0
    ):
        raise ValueError("Invalid Step44 fixed-alpha Ridge inputs")
    x_scaler = StandardScaler().fit(x_train, sample_weight=sample_weights)
    y_scaler = StandardScaler().fit(y_train, sample_weight=sample_weights)
    model = Ridge(alpha=float(alpha), solver="cholesky", fit_intercept=True)
    model.fit(
        x_scaler.transform(x_train),
        y_scaler.transform(y_train),
        sample_weight=sample_weights,
    )
    prediction = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(x_evaluate))
    )
    result = np.asarray(prediction, dtype=np.float64)
    if result.shape != (len(x_evaluate), y_train.shape[1]):
        raise AssertionError("Step44 fixed-alpha Ridge output shape changed")
    return result


def _equal_repetition_prototype(
    values: np.ndarray,
    manifest: pd.DataFrame,
    exercise_code: str,
    class_id: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    exercise = manifest["exercise_code"].astype(str).to_numpy()
    classes = manifest["class"].astype(int).to_numpy()
    repetitions = manifest["repetition"].astype(int).to_numpy()
    prototypes = []
    audit = []
    for repetition in step42c.REPETITIONS:
        rows = (
            (exercise == exercise_code)
            & (classes == class_id)
            & (repetitions == repetition)
        )
        if not rows.any() or not np.isfinite(values[rows]).all():
            raise AssertionError(
                f"Step44 missing prototype {exercise_code}{class_id}/r{repetition}"
            )
        prototypes.append(np.asarray(values[rows].mean(axis=0), dtype=np.float64))
        audit.append(
            {
                "exercise_code": exercise_code,
                "class": class_id,
                "repetition": repetition,
                "n_windows": int(rows.sum()),
            }
        )
    result = np.mean(prototypes, axis=0)
    if result.shape != (POSE_DIMENSION,) or not np.isfinite(result).all():
        raise AssertionError("Step44 prototype is invalid")
    return result, audit


def _opaque_query(
    query_manifest: pd.DataFrame,
    subject: str,
    proxy: MixedProxy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    catalog = held_catalog(proxy)
    slot_by_semantic = {semantic: slot for slot, semantic in enumerate(catalog)}
    keys = list(
        zip(
            query_manifest["exercise_code"].astype(str),
            query_manifest["class"].astype(int),
            query_manifest["repetition"].astype(int),
        )
    )
    unique = sorted(set(keys), key=lambda item: (item[0], item[1], item[2]))
    if len(unique) != N_TRIALS:
        raise AssertionError("Step44 query trial inventory changed")
    rng = np.random.default_rng(
        step44.stable_seed("step44_opaque", subject, proxy.proxy_id)
    )
    shuffled = [unique[index] for index in rng.permutation(len(unique))]
    key_to_alias = {key: f"q{index:02d}" for index, key in enumerate(shuffled)}
    aliases = np.asarray(sorted(key_to_alias.values()), dtype="U3")
    key_by_alias = {alias: key for key, alias in key_to_alias.items()}
    alias_to_index = {alias: index for index, alias in enumerate(aliases)}
    window_aliases = [key_to_alias[key] for key in keys]
    window_trial_index = np.asarray(
        [alias_to_index[alias] for alias in window_aliases], dtype=np.int16
    )
    truth = np.asarray(
        [slot_by_semantic[key_by_alias[alias][:2]] for alias in aliases],
        dtype=np.int8,
    )
    repetitions = np.asarray(
        [key_by_alias[alias][2] for alias in aliases], dtype=np.int8
    )
    counts = np.bincount(truth, minlength=N_CLASSES)
    if not np.array_equal(counts, np.full(N_CLASSES, 6)):
        raise AssertionError("Step44 query is not six trials per semantic slot")
    return aliases, window_trial_index, truth, repetitions


def _group_mean(
    values: np.ndarray, group_index: np.ndarray, n_groups: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty((n_groups, values.shape[1]), dtype=np.float64)
    for group in range(n_groups):
        rows = group_index == group
        if not rows.any():
            raise AssertionError("Step44 produced an empty trial group")
        result[group] = values[rows].mean(axis=0)
    return result


def build_target_geometry(
    prepared: b3.PreparedSubject,
    proxy: MixedProxy,
    policy: ParentPolicy,
) -> TargetGeometry:
    if prepared.subject != policy.subject:
        raise ValueError("Step44 target and parent policy subjects differ")
    fit, query = proxy_masks(prepared.manifest, proxy)
    fit_manifest = prepared.manifest.loc[fit].reset_index(drop=True)
    query_manifest = prepared.manifest.loc[query].reset_index(drop=True)
    weights = b0.equal_group_weights(fit_manifest)
    full_ridge = b0.fit_weighted_ridge(
        prepared.emg_features[fit],
        prepared.glove_features[fit],
        prepared.emg_features[query],
        weights,
    )
    pose_scaler = _pose_scaler(prepared, fit)
    query_pose_window = b0.transform(full_ridge.prediction, pose_scaler)
    aliases, window_trial_index, truth, repetitions = _opaque_query(
        query_manifest, prepared.subject, proxy
    )
    query_pose = _group_mean(query_pose_window, window_trial_index, N_TRIALS)

    columns = b3.FEATURE_SETS[policy.feature_set]
    cluster_scaler = StandardScaler().fit(
        prepared.emg_features[fit][:, columns], sample_weight=weights
    )
    emg_view = step44.l2_normalize(
        _group_mean(
            cluster_scaler.transform(prepared.emg_features[query][:, columns]),
            window_trial_index,
            N_TRIALS,
        )
    )
    pose_view = step44.l2_normalize(query_pose)
    if policy.pose_view_weight == 0.0:
        cluster_view = emg_view
    else:
        weight = policy.pose_view_weight
        cluster_view = np.concatenate(
            (emg_view, np.sqrt(weight) * pose_view), axis=1
        ) / np.sqrt(1.0 + weight)

    manifest = prepared.manifest
    manifest_repetitions = manifest["repetition"].astype(int).to_numpy()
    cross_fitted = np.full((len(manifest), POSE_DIMENSION), np.nan, dtype=np.float64)
    crossfit_rows = []
    for repetition in step42c.REPETITIONS:
        train = fit & (manifest_repetitions != repetition)
        evaluate = fit & (manifest_repetitions == repetition)
        train_manifest = manifest.loc[train].reset_index(drop=True)
        train_weights = b0.equal_group_weights(train_manifest)
        prediction = fixed_alpha_ridge_prediction(
            prepared.emg_features[train],
            prepared.glove_features[train],
            prepared.emg_features[evaluate],
            train_weights,
            full_ridge.alpha,
        )
        cross_fitted[evaluate] = b0.transform(prediction, pose_scaler)
        crossfit_rows.append(
            {
                "held_repetition": repetition,
                "n_train_windows": int(train.sum()),
                "n_evaluate_windows": int(evaluate.sum()),
                "fixed_alpha": full_ridge.alpha,
            }
        )
    if np.isnan(cross_fitted[fit]).any() or np.isfinite(cross_fitted[query]).any():
        raise AssertionError("Step44 cross-fit exposure boundary failed")

    landmark_rows = []
    target_audit = []
    for position, (exercise_code, class_id) in enumerate(fit_catalog(proxy)):
        prototype, audit = _equal_repetition_prototype(
            cross_fitted, manifest, exercise_code, class_id
        )
        landmark_rows.append(prototype)
        target_audit.extend(
            {
                **row,
                "landmark_position": position,
                "fixed_alpha": full_ridge.alpha,
            }
            for row in audit
        )
    target_landmarks = np.vstack(landmark_rows)
    landmark_roles = np.asarray(
        [exercise for exercise, _ in fit_catalog(proxy)], dtype="U1"
    )
    query_row_keys = query_manifest["step41_cache_row_index"].to_numpy(dtype=np.int64)
    if (
        query_pose.shape != (N_TRIALS, POSE_DIMENSION)
        or cluster_view.shape[0] != N_TRIALS
        or target_landmarks.shape != (N_LANDMARKS, POSE_DIMENSION)
        or not np.isfinite(query_pose).all()
        or not np.isfinite(cluster_view).all()
    ):
        raise AssertionError("Step44 target geometry is invalid")
    return TargetGeometry(
        trial_aliases=aliases,
        trial_slots=truth,
        trial_repetitions=repetitions,
        window_trial_index=window_trial_index,
        query_row_keys=query_row_keys,
        query_predicted_pose=query_pose,
        cluster_view=cluster_view,
        target_landmarks=target_landmarks,
        landmark_roles=landmark_roles,
        ridge_alpha=full_ridge.alpha,
        target_audit=pd.DataFrame(target_audit),
        crossfit_audit=pd.DataFrame(crossfit_rows),
    )


def build_source_geometry(
    subjects: Mapping[str, b3.PreparedSubject],
    source_subjects: tuple[str, ...],
    proxy: MixedProxy,
) -> SourceGeometry:
    if (
        len(source_subjects) != N_SOURCES
        or len(set(source_subjects)) != N_SOURCES
        or not set(source_subjects).issubset(subjects)
    ):
        raise ValueError("Step44 source scope is invalid")
    landmarks = fit_catalog(proxy)
    anchors = held_catalog(proxy)
    per_source_landmarks = []
    per_source_anchors = []
    audit_rows = []
    for source_subject in source_subjects:
        prepared = subjects[source_subject]
        fit, _ = proxy_masks(prepared.manifest, proxy)
        scaler = _pose_scaler(prepared, fit)
        pose = b0.transform(prepared.glove_features, scaler)
        landmark_rows = []
        anchor_rows = []
        for role, catalog, output in (
            ("seen_landmark", landmarks, landmark_rows),
            ("held_anchor", anchors, anchor_rows),
        ):
            for position, (exercise_code, class_id) in enumerate(catalog):
                prototype, audit = _equal_repetition_prototype(
                    pose, prepared.manifest, exercise_code, class_id
                )
                output.append(prototype)
                audit_rows.extend(
                    {
                        **row,
                        "source_subject": source_subject,
                        "semantic_role": role,
                        "semantic_position": position,
                    }
                    for row in audit
                )
        per_source_landmarks.append(np.vstack(landmark_rows))
        per_source_anchors.append(np.vstack(anchor_rows))
    per_source_landmarks_array = np.stack(per_source_landmarks)
    per_source_anchors_array = np.stack(per_source_anchors)
    source_landmarks = per_source_landmarks_array.mean(axis=0)
    source_anchors = per_source_anchors_array.mean(axis=0)
    anchor_roles = np.asarray([exercise for exercise, _ in anchors], dtype="U1")
    held_b_positions = [
        position
        for position, semantic in enumerate(anchors)
        if semantic == ("B", proxy.held_b)
    ]
    if len(held_b_positions) != 1:
        raise AssertionError("Step44 opposite held-B source anchor is ambiguous")
    wrong_source_b = source_anchors[held_b_positions[0]].copy()
    if (
        source_landmarks.shape != (N_LANDMARKS, POSE_DIMENSION)
        or source_anchors.shape != (N_CLASSES, POSE_DIMENSION)
        or per_source_landmarks_array.shape != (N_SOURCES, N_LANDMARKS, POSE_DIMENSION)
        or per_source_anchors_array.shape != (N_SOURCES, N_CLASSES, POSE_DIMENSION)
        or not np.isfinite(source_landmarks).all()
        or not np.isfinite(source_anchors).all()
    ):
        raise AssertionError("Step44 source geometry is invalid")
    return SourceGeometry(
        source_landmarks=source_landmarks,
        source_anchors=source_anchors,
        anchor_roles=anchor_roles,
        wrong_source_b=wrong_source_b,
        per_source_landmarks=per_source_landmarks_array,
        per_source_anchors=per_source_anchors_array,
        source_subjects=source_subjects,
        source_audit=pd.DataFrame(audit_rows),
    )
