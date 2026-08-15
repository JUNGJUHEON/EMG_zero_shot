from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step42a_legal_cache as legal42a
from project_paths import CAPSTONE_ROOT as CAPSTONE


DATA_ROOT = CAPSTONE / "10. Ninapro_DB5"
CACHE_STEM = "step42b0_s0v2_legal_amplitude_A1_12_B5_B6_2p0s_v1"
SUBJECTS = legal42a.SUBJECTS

ARRAY_SPECS: dict[str, tuple[int, ...]] = {
    "emg_windows_filtered": (400, 16),
    "emg_windows_norm": (400, 16),
    "emg_envelope_5hz": (400, 16),
    "emg_features": (128,),
    "glove_features": (66,),
    "glove_windows": (400, 22),
}


@dataclass(frozen=True)
class LegalAmplitudeCache:
    subject: str
    manifest: pd.DataFrame
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    npz_path: Path
    manifest_path: Path
    metadata_path: Path

    def __getattr__(self, name: str) -> np.ndarray:
        if name in ARRAY_SPECS:
            return self.arrays[name]
        raise AttributeError(name)


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


def validate_cache(cache: LegalAmplitudeCache) -> None:
    legal42a.validate_manifest(cache.manifest, cache.subject)
    if set(cache.arrays) != set(ARRAY_SPECS):
        raise AssertionError(f"Unexpected Step42B0 arrays: {sorted(cache.arrays)}")
    for name, tail in ARRAY_SPECS.items():
        values = cache.arrays[name]
        expected = (len(cache.manifest), *tail)
        if values.shape != expected or values.dtype != np.float32:
            raise AssertionError(
                f"Step42B0 {name} must be float32 {expected}, got {values.dtype} {values.shape}"
            )
        if not np.isfinite(values).all():
            raise AssertionError(f"Step42B0 {name} contains non-finite values")

    metadata = cache.metadata
    if metadata.get("cache_stem") != CACHE_STEM:
        raise AssertionError("Step42B0 cache metadata identity changed")
    if metadata.get("subject") != cache.subject:
        raise AssertionError("Step42B0 cache subject metadata changed")
    if metadata.get("fit_exposure") != "A1-12+B5+B6":
        raise AssertionError("Step42B0 cache fit exposure changed")
    if metadata.get("contains_evaluated_b_rows") is not False:
        raise AssertionError("Step42B0 cache lacks its evaluated-class firewall")
    if int(metadata.get("rows", -1)) != len(cache.manifest):
        raise AssertionError("Step42B0 cache row count changed")


def load_legal_amplitude_cache(subject: str) -> LegalAmplitudeCache:
    npz_path, manifest_path, metadata_path = cache_paths(subject)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        raise AssertionError("Step42B0 cache metadata lacks output records")
    for name, path in (("npz", npz_path), ("manifest", manifest_path)):
        record = outputs.get(name)
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(path):
            raise AssertionError(f"Step42B0 cache {name} hash does not match metadata")

    manifest = pd.read_csv(manifest_path)
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != set(ARRAY_SPECS):
            raise AssertionError(f"Unexpected Step42B0 cache arrays: {sorted(payload.files)}")
        arrays = {
            name: np.ascontiguousarray(payload[name], dtype=np.float32)
            for name in ARRAY_SPECS
        }
    cache = LegalAmplitudeCache(
        subject=subject,
        manifest=manifest,
        arrays=arrays,
        metadata=metadata,
        npz_path=npz_path,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    validate_cache(cache)
    return cache
