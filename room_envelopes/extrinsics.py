"""Extrinsics manifest loader for Room Envelopes.

The manifest is an NPZ produced by the (now-skipped) Hypersim pose-recovery
pipeline. Fields:

    sample_id              (N,)         shard-relative path, e.g. "train-0000/ai_..."
    w2c                    (N, 4, 4)    OpenCV camera-from-world, ALREADY metric.
    valid                  (N,)         bool, rotation sanity / matched-by-builder.
    meters_per_asset_unit  (N,)         per-sample scale used by the builder when
                                        converting raw asset-unit translations to
                                        metres. The stored ``w2c`` already has it
                                        applied; we expose it for record-keeping
                                        only and NEVER multiply depth or pose by
                                        it again.

Each entry is keyed by the **basename** of the sample (last path component of
``sample_id``), e.g. ``ai_001_001-cam_00-0``, so it matches the basename the
dataset loader uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


class MissingExtrinsicsError(KeyError):
    """Raised when extrinsics are required but missing/invalid for a sample."""


@dataclass(frozen=True)
class ExtrinsicsEntry:
    w2c44: np.ndarray  # (4, 4) float32, already metric
    valid: bool
    meters_per_asset_unit: float


class ExtrinsicsManifest:
    """In-memory view of the extrinsics manifest, keyed by sample basename."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"extrinsics manifest not found: {path}")
        self.path = path
        with np.load(path) as f:
            sample_ids = f["sample_id"]
            w2c = f["w2c"].astype(np.float32, copy=False)
            valid = f["valid"]
            mau = f["meters_per_asset_unit"].astype(np.float32, copy=False)
        if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
            raise ValueError(f"manifest w2c shape must be (N, 4, 4), got {w2c.shape}")
        if not (len(sample_ids) == len(w2c) == len(valid) == len(mau)):
            raise ValueError("manifest arrays have mismatched lengths")
        self._by_base: dict[str, ExtrinsicsEntry] = {}
        for i in range(len(sample_ids)):
            base = str(sample_ids[i]).rsplit("/", 1)[-1]
            self._by_base[base] = ExtrinsicsEntry(
                w2c44=w2c[i].copy(),
                valid=bool(valid[i]),
                meters_per_asset_unit=float(mau[i]),
            )

    def __contains__(self, base: str) -> bool:
        return base in self._by_base

    def __len__(self) -> int:
        return len(self._by_base)

    def get(self, base: str) -> ExtrinsicsEntry | None:
        return self._by_base.get(base)

    def w2c34(self, base: str) -> np.ndarray:
        """Return the OpenCV w2c 3×4 for ``base``.

        Raises :class:`MissingExtrinsicsError` if the entry is absent or invalid.
        """
        entry = self._by_base.get(base)
        if entry is None:
            raise MissingExtrinsicsError(
                f"sample {base!r} not in extrinsics manifest at {self.path}"
            )
        if not entry.valid:
            raise MissingExtrinsicsError(
                f"sample {base!r} has valid=False in extrinsics manifest"
            )
        return entry.w2c44[:3, :].copy()

    def valid_bases(self) -> set[str]:
        return {b for b, e in self._by_base.items() if e.valid}
