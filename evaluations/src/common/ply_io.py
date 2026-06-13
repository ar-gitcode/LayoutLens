"""PLY writing and metric-dict key prefixing."""
from __future__ import annotations

import numpy as np

from geometry.room_envelope_geometry import save_point_cloud_ply


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    """Return a new dict with every key prefixed by ``prefix``.

    Skips ``None``-valued entries to keep CSV columns clean.
    """
    if not prefix:
        return dict(metrics)
    out = {}
    for k, v in metrics.items():
        out[f"{prefix}{k}"] = v
    return out


def write_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """ASCII PLY writer (delegates to ``room_envelope_geometry.save_point_cloud_ply``)."""
    save_point_cloud_ply(np.asarray(points, dtype=np.float32), str(path),
                         colors=None if colors is None else np.asarray(colors))

