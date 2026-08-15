from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from project_paths import CAPSTONE_ROOT as CAPSTONE


DATA_ROOT = CAPSTONE / "10. Ninapro_DB5"
CACHE_STEM = "step42c_unlabeled_primary_B1_B2_B3_B4_B7_2p0s_v1"
SUBJECTS = tuple(f"s{index}" for index in range(1, 11))
MODEL_MANIFEST_COLUMNS = (
    "unlabeled_cache_row_index",
    "subject",
    "opaque_row_id",
    "opaque_trial_id",
)
MODEL_ARRAY_NAMES = ("emg_features", "opaque_row_id", "opaque_trial_id")
FORBIDDEN_TOKENS = (
    "class",
    "gesture",
    "repetition",
    "exercise_code",
    "source_file",
    "source_row",
    "glove",
    "pose",
)


@dataclass(frozen=True)
class UnlabeledPrimaryCache:
    subject: str
    manifest: pd.DataFrame
    emg_features: np.ndarray
    opaque_row_id: np.ndarray
    opaque_trial_id: np.ndarray
    metadata: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_paths(subject: str) -> tuple[Path, Path, Path]:
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown Step42C subject: {subject}")
    root = DATA_ROOT / subject / "EMBridge_data_cache"
    return (
        root / f"{CACHE_STEM}.npz",
        root / f"{CACHE_STEM}_manifest.csv",
        root / f"{CACHE_STEM}_metadata.json",
    )


def validate_model_schema(manifest: pd.DataFrame, metadata: dict[str, object]) -> None:
    if tuple(manifest.columns) != MODEL_MANIFEST_COLUMNS:
        raise AssertionError("Step42C model-facing manifest schema changed")
    lowered_columns = [column.lower() for column in manifest.columns]
    for token in FORBIDDEN_TOKENS:
        if any(token in column for column in lowered_columns):
            raise AssertionError(f"Step42C manifest leaks forbidden token: {token}")
    serialized_metadata = json.dumps(metadata, sort_keys=True).lower()
    for token in ("glove", "pose", "repetition", "source_row"):
        if token in serialized_metadata:
            raise AssertionError(f"Step42C metadata leaks forbidden token: {token}")


def validate_cache(cache: UnlabeledPrimaryCache) -> None:
    manifest = cache.manifest.reset_index(drop=True)
    validate_model_schema(manifest, cache.metadata)
    if cache.subject not in SUBJECTS or set(manifest["subject"].astype(str)) != {
        cache.subject
    }:
        raise AssertionError("Step42C cache subject identity changed")
    if not np.array_equal(
        manifest["unlabeled_cache_row_index"].to_numpy(dtype=int),
        np.arange(len(manifest)),
    ):
        raise AssertionError("Step42C cache row order changed")
    if cache.emg_features.shape != (len(manifest), 128):
        raise AssertionError("Step42C unlabeled EMG feature shape changed")
    if cache.opaque_row_id.shape != (len(manifest),) or cache.opaque_trial_id.shape != (
        len(manifest),
    ):
        raise AssertionError("Step42C opaque ID arrays are misaligned")
    if not np.isfinite(cache.emg_features).all():
        raise AssertionError("Step42C unlabeled EMG contains non-finite values")
    if not np.array_equal(
        cache.opaque_row_id.astype(str), manifest["opaque_row_id"].astype(str).to_numpy()
    ) or not np.array_equal(
        cache.opaque_trial_id.astype(str),
        manifest["opaque_trial_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("Step42C opaque IDs disagree between NPZ and manifest")
    if manifest["opaque_row_id"].duplicated().any():
        raise AssertionError("Step42C opaque row IDs are not unique")
    if manifest["opaque_trial_id"].nunique() != 30:
        raise AssertionError("Step42C model cache must contain exactly 30 trials")
    trial_sizes = manifest.groupby("opaque_trial_id").size()
    if (trial_sizes <= 0).any():
        raise AssertionError("Step42C model cache contains an empty trial")
    metadata = cache.metadata
    if metadata.get("cache_stem") != CACHE_STEM or metadata.get("subject") != cache.subject:
        raise AssertionError("Step42C cache metadata identity changed")
    if int(metadata.get("rows", -1)) != len(manifest) or int(
        metadata.get("trials", -1)
    ) != 30:
        raise AssertionError("Step42C cache metadata counts changed")
    if metadata.get("contains_labels") is not False or metadata.get("contains_target_kinematics") is not False:
        raise AssertionError("Step42C model cache exposure claims changed")
    if metadata.get("class_inventory_known") is not True or int(
        metadata.get("trials_per_class_prior", -1)
    ) != 6:
        raise AssertionError("Step42C transductive side-information claim changed")


def load_unlabeled_primary_cache(subject: str) -> UnlabeledPrimaryCache:
    npz_path, manifest_path, metadata_path = cache_paths(subject)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata.get("outputs", {})
    for name, path in (("npz", npz_path), ("manifest", manifest_path)):
        record = outputs.get(name, {})
        if record.get("path") != str(path) or record.get("sha256") != sha256_file(path):
            raise AssertionError(f"Step42C unlabeled cache hash changed: {subject}/{name}")
    manifest = pd.read_csv(manifest_path)
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != set(MODEL_ARRAY_NAMES):
            raise AssertionError("Step42C model NPZ arrays changed")
        cache = UnlabeledPrimaryCache(
            subject=subject,
            manifest=manifest,
            emg_features=np.asarray(payload["emg_features"], dtype=np.float32),
            opaque_row_id=np.asarray(payload["opaque_row_id"]),
            opaque_trial_id=np.asarray(payload["opaque_trial_id"]),
            metadata=metadata,
        )
    validate_cache(cache)
    return cache
