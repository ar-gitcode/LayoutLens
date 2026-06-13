"""Depth-scale diagnostics shared by 2D and 3D eval."""
from __future__ import annotations

import numpy as np


def _stack_depths(arr_or_list):
    """Return a single (S,H,W) array from either a list of 2D arrays or a stack."""
    if arr_or_list is None:
        return None
    if isinstance(arr_or_list, list):
        if not arr_or_list:
            return None
        return np.stack([np.asarray(a) for a in arr_or_list], axis=0)
    a = np.asarray(arr_or_list)
    return a


def _gather_overlap(pred_depths, gt_depths, valid_masks=None, eps: float = 1e-6):
    """Return (pred_vec, gt_vec) of overlapping valid pixels across all views."""
    p = _stack_depths(pred_depths)
    g = _stack_depths(gt_depths)
    if p is None or g is None or p.size == 0 or g.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if p.shape != g.shape:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    valid = (p > eps) & (g > eps)
    if valid_masks is not None:
        v = _stack_depths(valid_masks)
        if v is not None and v.shape == p.shape:
            valid = valid & v.astype(bool)
    return p[valid].astype(np.float64), g[valid].astype(np.float64)


def summarize_depth_scale(pred_depths,
                          gt_depths,
                          valid_masks=None,
                          eps: float = 1e-6) -> dict:
    """Per-pixel depth-scale diagnostics over the valid pred/GT overlap.

    Returns:
        ``median_pred_depth``, ``median_gt_depth``, ``median_pred_gt_ratio``,
        ``median_gt_pred_scale``, ``n_overlap``. NaN with ``n_overlap=0``
        when there is no overlap.
    """
    p, g = _gather_overlap(pred_depths, gt_depths, valid_masks, eps=eps)
    n = int(p.size)
    if n == 0:
        nan = float("nan")
        return {
            "median_pred_depth":     nan,
            "median_gt_depth":       nan,
            "median_pred_gt_ratio":  nan,
            "median_gt_pred_scale":  nan,
            "n_overlap":             0,
        }
    return {
        "median_pred_depth":     float(np.median(p)),
        "median_gt_depth":       float(np.median(g)),
        "median_pred_gt_ratio":  float(np.median(p / np.maximum(g, eps))),
        "median_gt_pred_scale":  float(np.median(g / np.maximum(p, eps))),
        "n_overlap":             n,
    }


def compute_median_depth_scale(pred_depths,
                               gt_depths,
                               valid_masks=None,
                               eps: float = 1e-6) -> float:
    """Scalar scale ``s = median(gt / pred)`` over the valid pred/GT overlap.

    Returns NaN when no overlap is available.
    """
    p, g = _gather_overlap(pred_depths, gt_depths, valid_masks, eps=eps)
    if p.size == 0:
        return float("nan")
    return float(np.median(g / np.maximum(p, eps)))

