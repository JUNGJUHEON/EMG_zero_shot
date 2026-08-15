from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_paths import CAPSTONE_ROOT as CAPSTONE


DATA_ROOT = CAPSTONE / "10. Ninapro_DB5"
CACHE_STEM = "step42a_s0v2_legal_A1_12_B5_B6_2p0s_v1"
SUBJECTS = tuple(f"s{index}" for index in range(1, 11))
LEGAL_A_CLASSES = tuple(range(1, 13))
LEGAL_B_CLASSES = (5, 6)
REPETITIONS = tuple(range(1, 7))
IDENTITY_COLUMNS = (
    "step41_cache_row_index",
    "subject",
    "exercise_code",
    "class",
    "repetition",
    "start_sample",
    "end_sample",
)


@dataclass(frozen=True)
class LegalFitCache:
    subject: str
    manifest: pd.DataFrame
    emg: np.ndarray
    pose: np.ndarray
    metadata: dict[str, Any]
    npz_path: Path
    manifest_path: Path
    metadata_path: Path


def cache_paths(subject: str) -> tuple[Path, Path, Path]:
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown DB5 subject: {subject}")
    root = DATA_ROOT / subject / "EMBridge_data_cache"
    return (
        root / f"{CACHE_STEM}.npz",
        root / f"{CACHE_STEM}_manifest.csv",
        root / f"{CACHE_STEM}_metadata.json",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: pd.DataFrame, subject: str) -> None:
    missing = set(IDENTITY_COLUMNS) - set(manifest.columns)
    if missing:
        raise AssertionError(f"Legal cache manifest lacks identity columns: {sorted(missing)}")
    if manifest.empty:
        raise AssertionError("Legal cache manifest is empty")
    if set(manifest["subject"].astype(str)) != {subject}:
        raise AssertionError("Legal cache contains another subject")
    if not manifest["s0v2_targetab56_fit"].astype(bool).all():
        raise AssertionError("Legal cache contains a row outside the S0-v2 fit mask")

    exercise = manifest["exercise_code"].astype(str)
    classes = manifest["class"].astype(int)
    observed_a = set(classes[exercise.eq("A")])
    observed_b = set(classes[exercise.eq("B")])
    if observed_a != set(LEGAL_A_CLASSES) or observed_b != set(LEGAL_B_CLASSES):
        raise AssertionError(
            f"Legal cache class coverage changed: A={sorted(observed_a)}, B={sorted(observed_b)}"
        )
    if set(exercise) != {"A", "B"}:
        raise AssertionError("Legal cache contains a forbidden exercise")

    for (exercise_code, class_id), rows in manifest.groupby(["exercise_code", "class"]):
        observed_repetitions = set(rows["repetition"].astype(int))
        if observed_repetitions != set(REPETITIONS):
            raise AssertionError(
                f"{exercise_code}{class_id} repetition coverage changed: "
                f"{sorted(observed_repetitions)}"
            )

    outcome_columns = (
        "s0v2_targetab56_test",
        "s0v2_targetab56_b8sensitivity_test",
        "v2_b9_17_excluded",
        "v2_exercise_c_excluded",
    )
    for column in outcome_columns:
        if column in manifest and manifest[column].astype(bool).any():
            raise AssertionError(f"Legal cache contains forbidden role rows: {column}")


def validate_cache(cache: LegalFitCache) -> None:
    validate_manifest(cache.manifest, cache.subject)
    expected_shapes = {
        "emg": (len(cache.manifest), 400, 16),
        "pose": (len(cache.manifest), 400, 22),
    }
    for name, expected in expected_shapes.items():
        values = getattr(cache, name)
        if values.shape != expected:
            raise AssertionError(f"Legal cache {name} shape {values.shape} != {expected}")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise AssertionError(f"Legal cache {name} must be finite float32")

    metadata = cache.metadata
    if metadata.get("cache_stem") != CACHE_STEM:
        raise AssertionError("Legal cache metadata identity changed")
    if metadata.get("subject") != cache.subject:
        raise AssertionError("Legal cache metadata subject changed")
    if metadata.get("fit_exposure") != "A1-12+B5+B6":
        raise AssertionError("Legal cache fit exposure changed")
    if metadata.get("contains_evaluated_b_rows") is not False:
        raise AssertionError("Legal cache does not declare the outcome firewall")
    if int(metadata.get("rows", -1)) != len(cache.manifest):
        raise AssertionError("Legal cache metadata row count changed")


def load_legal_fit_cache(subject: str) -> LegalFitCache:
    npz_path, manifest_path, metadata_path = cache_paths(subject)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        raise AssertionError("Legal cache metadata lacks output records")
    for name, path in (("npz", npz_path), ("manifest", manifest_path)):
        record = outputs.get(name)
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(path):
            raise AssertionError(f"Legal cache {name} hash does not match metadata")

    manifest = pd.read_csv(manifest_path)
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"emg_windows_norm", "glove_windows"}:
            raise AssertionError(f"Unexpected legal cache arrays: {sorted(payload.files)}")
        emg = np.ascontiguousarray(payload["emg_windows_norm"], dtype=np.float32)
        pose = np.ascontiguousarray(payload["glove_windows"], dtype=np.float32)

    cache = LegalFitCache(
        subject=subject,
        manifest=manifest,
        emg=emg,
        pose=pose,
        metadata=metadata,
        npz_path=npz_path,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    validate_cache(cache)
    return cache
