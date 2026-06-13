"""Surface-normal metric for 2D eval.

Wraps the canonical ``compute_normal_metrics``, owns the predicted-normal
layout preprocessing, and provides ``compute_normals_numpy`` (depth→normals
fallback used when a model has no normal head).
"""
from __future__ import annotations

import numpy as np

from eval_metrics import compute_normal_metrics

__all__ = ["compute_normal_metrics", "compute_normals_numpy", "normals_from_pred"]


def normals_from_pred(pred_normal) -> np.ndarray:
    """Convert predicted normals ``(S, 3, H, W)`` → ``(S, H, W, 3)``.

    Pass-through when already channel-last.
    """
    pn = np.asarray(pred_normal)
    if pn.ndim == 4 and pn.shape[1] == 3:
        pn = np.transpose(pn, (0, 2, 3, 1))
    return pn


def compute_normals_numpy(depth: np.ndarray) -> tuple:
    """Compute surface normals from depth map using central finite differences.

    Args:
        depth: (H, W) float32 metres.

    Returns:
        (normals, valid_mask): normals (H, W, 3) unit-length, valid_mask (H, W) bool.
    """
    H, W = depth.shape
    valid = depth > 1e-6

    dx = np.zeros_like(depth)
    dy = np.zeros_like(depth)
    dx[:, 1:-1] = depth[:, 2:] - depth[:, :-2]
    dy[1:-1, :] = depth[2:, :] - depth[:-2, :]

    valid_x = np.zeros((H, W), dtype=bool)
    valid_y = np.zeros((H, W), dtype=bool)
    valid_x[:, 1:-1] = valid[:, 2:] & valid[:, :-2] & valid[:, 1:-1]
    valid_y[1:-1, :] = valid[2:, :] & valid[:-2, :] & valid[1:-1, :]
    valid_n = valid_x & valid_y

    nx = -dx
    ny = -dy
    nz = np.full_like(depth, 2.0)

    norm = np.sqrt(nx**2 + ny**2 + nz**2).clip(min=1e-8)
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    normals = np.stack([nx, ny, nz], axis=-1)
    normals[~valid_n] = 0.0
    return normals, valid_n
