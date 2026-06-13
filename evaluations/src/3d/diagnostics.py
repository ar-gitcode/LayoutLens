"""Geometry diagnostics: axis-aligned bounding-box size."""
from __future__ import annotations

import numpy as np

_NAN3 = np.array([np.nan, np.nan, np.nan], dtype=np.float64)


def bbox_size(points: np.ndarray) -> dict:
    """Axis-aligned bbox stats. Empty cloud → all-NaN dict (no exception).

    Returns:
        dict with float lists ``bbox_min``, ``bbox_max``, ``bbox_size`` (sx, sy, sz).
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return {
            "bbox_min": _NAN3.tolist(),
            "bbox_max": _NAN3.tolist(),
            "bbox_size": _NAN3.tolist(),
        }
    mn = pts.min(axis=0).astype(np.float64)
    mx = pts.max(axis=0).astype(np.float64)
    return {
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "bbox_size": (mx - mn).tolist(),
    }

