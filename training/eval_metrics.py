"""Reusable 2D evaluation metrics for room-envelope experiments.

Used both by training-time validation and by offline evaluation scripts.

Conventions:
- All metrics accept numpy arrays (operate on float32/float64).
- ``valid_mask`` is a boolean array selecting which pixels participate. Where
  ``valid_mask`` is None, the metric falls back to a "depth > eps" / "all-True"
  rule documented per function.
- Functions return Python ``float`` values (or dicts of floats). NaN is returned
  when no valid pixels exist; callers should aggregate with ``np.nanmean``.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------

def compute_depth_metrics(pred_depth: np.ndarray,
                          gt_depth: np.ndarray,
                          valid_mask: np.ndarray | None = None) -> dict:
    """Per-pixel depth metrics over a single mask.

    Args:
        pred_depth: (H, W) or arbitrary numpy array of predicted depths (metres).
        gt_depth:   same shape, GT depths (metres).
        valid_mask: boolean array of same shape; True = use this pixel. If None,
            falls back to ``gt_depth > EPS``.

    Returns:
        dict with keys: absrel, rmse, log_rmse, delta1, delta2, delta3, silog,
        n_valid (int).  All metric values are Python floats; NaN if there are
        no valid pixels. ``silog`` is the scale-invariant log error in the KITTI
        convention: ``100 * sqrt(Var(ln pred - ln gt))`` (non-negative).
    """
    pred = np.asarray(pred_depth, dtype=np.float64)
    gt = np.asarray(gt_depth, dtype=np.float64)

    if valid_mask is None:
        valid = gt > EPS
    else:
        valid = np.asarray(valid_mask, dtype=bool) & (gt > EPS)

    n_valid = int(valid.sum())
    nan = float("nan")
    if n_valid == 0:
        return {
            "absrel": nan, "rmse": nan, "log_rmse": nan,
            "delta1": nan, "delta2": nan, "delta3": nan,
            "silog": nan, "n_valid": 0,
        }

    p = np.clip(pred[valid], EPS, None)
    g = gt[valid]

    abs_rel = float((np.abs(p - g) / g).mean())
    rmse = float(np.sqrt(((p - g) ** 2).mean()))
    log_rmse = float(np.sqrt(((np.log(p) - np.log(g)) ** 2).mean()))

    ratio = np.maximum(p / g, g / p)
    delta1 = float((ratio < 1.25).mean())
    delta2 = float((ratio < 1.25 ** 2).mean())
    delta3 = float((ratio < 1.25 ** 3).mean())

    # Scale-invariant log error (SILog), KITTI depth-prediction benchmark
    # convention:  SILog = 100 * sqrt(Var(d)),  d = ln(pred) - ln(gt)
    #            = 100 * sqrt(mean(d^2) - mean(d)^2).
    # ``np.var`` (ddof=0) IS mean(d^2) - mean(d)^2, so this is exactly the
    # scale-invariant log MSE (Eigen 2014 Eqn.1, lambda=1) under a sqrt and the
    # standard x100 reporting scale. Non-negative by construction; max(...,0)
    # only guards float round-off when the variance is ~0.
    #
    # NOTE: the previous form ``d.var() - 0.85 * d.mean()**2`` double-subtracted
    # the mean term (== mean(d^2) - 1.85*mean(d)^2), matched no standard SILog,
    # could go negative, and could invert model rankings. See the eval audit.
    d = np.log(p) - np.log(g)
    silog = float(100.0 * np.sqrt(max(float(d.var()), 0.0)))

    return {
        "absrel": abs_rel,
        "rmse": rmse,
        "log_rmse": log_rmse,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "silog": silog,
        "n_valid": n_valid,
    }


def compute_depth_metrics_with_splits(pred_depth: np.ndarray,
                                      gt_depth: np.ndarray,
                                      valid_mask: np.ndarray | None,
                                      layout_mask: np.ndarray | None) -> dict:
    """Compute depth metrics on three subsets: all / visible / occluded.

    Args:
        pred_depth: (H, W) predicted depth.
        gt_depth:   (H, W) GT layout depth.
        valid_mask: optional (H, W) bool, valid layout pixels (e.g.
            ``layout_depth_masks`` from the dataset). If None, defaults to
            ``gt_depth > EPS``.
        layout_mask: optional (H, W) float / bool, 1 where structural surface
            is visible (not behind clutter), 0 elsewhere. If None, only the
            "_all" subset is reported.

    Returns:
        dict with keys ``<metric>_all``, ``<metric>_visible``, ``<metric>_occluded``
        for ``absrel``, ``rmse``, ``log_rmse``, ``delta1``, ``delta2``, ``delta3``,
        ``silog``. Plus ``n_valid_<subset>`` integer counts.

        The "_visible" / "_occluded" keys are NaN when ``layout_mask`` is None
        or empty for that subset. The occluded subset is the headline metric:
        it measures depth accuracy *behind* clutter, the room envelope.
    """
    if valid_mask is None:
        base_valid = np.asarray(gt_depth) > EPS
    else:
        base_valid = np.asarray(valid_mask, dtype=bool) & (np.asarray(gt_depth) > EPS)

    out: dict = {}
    metric_keys = ("absrel", "rmse", "log_rmse", "delta1", "delta2", "delta3", "silog")

    def _add(prefix: str, m: dict):
        for k in metric_keys:
            out[f"{k}_{prefix}"] = m[k]
        out[f"n_valid_{prefix}"] = m["n_valid"]

    _add("all", compute_depth_metrics(pred_depth, gt_depth, base_valid))

    if layout_mask is None:
        for subset in ("visible", "occluded"):
            for k in metric_keys:
                out[f"{k}_{subset}"] = float("nan")
            out[f"n_valid_{subset}"] = 0
        return out

    lm = np.asarray(layout_mask)
    visible = base_valid & (lm > 0.5)
    occluded = base_valid & (lm <= 0.5)
    _add("visible", compute_depth_metrics(pred_depth, gt_depth, visible))
    _add("occluded", compute_depth_metrics(pred_depth, gt_depth, occluded))
    return out


# ---------------------------------------------------------------------------
# Mask metrics
# ---------------------------------------------------------------------------

def compute_mask_metrics(pred_logits_or_prob: np.ndarray,
                         gt_mask: np.ndarray,
                         valid_mask: np.ndarray | None = None,
                         threshold: float = 0.5) -> dict:
    """Binary mask metrics. Accepts either logits or probabilities.

    Args:
        pred_logits_or_prob: arbitrary numeric array. If any value falls
            outside ``[0, 1]`` the array is interpreted as logits and a sigmoid
            is applied. Otherwise it is interpreted as probabilities directly.
        gt_mask:   array of same shape, treated as boolean (>0.5 = positive).
        valid_mask: optional bool array of same shape; pixels where
            ``valid_mask == False`` are excluded from the metric.
        threshold: classification threshold on probabilities.

    Returns:
        dict with iou, f1, precision, recall, accuracy (all floats in [0,1]),
        plus n_pixels (int).
    """
    pred = np.asarray(pred_logits_or_prob, dtype=np.float64)
    pmin, pmax = pred.min(), pred.max()
    if pmin < 0.0 or pmax > 1.0:
        # Sigmoid (use stable form)
        pred = 1.0 / (1.0 + np.exp(-pred))

    pred_bin = pred >= threshold
    gt_bin = np.asarray(gt_mask) > 0.5

    if valid_mask is not None:
        v = np.asarray(valid_mask, dtype=bool)
        pred_bin = pred_bin & v
        gt_bin = gt_bin & v
        n_pixels = int(v.sum())
        # Limit accuracy/tn computation to valid pixels
        v_flat = v
    else:
        n_pixels = int(gt_bin.size)
        v_flat = np.ones_like(gt_bin, dtype=bool)

    tp = int((pred_bin & gt_bin).sum())
    fp = int((pred_bin & ~gt_bin & v_flat).sum())
    fn = int((~pred_bin & gt_bin & v_flat).sum())
    tn = int((~pred_bin & ~gt_bin & v_flat).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
        "n_pixels": n_pixels,
    }


# ---------------------------------------------------------------------------
# Normal metrics
# ---------------------------------------------------------------------------

def compute_normal_metrics(pred_normals: np.ndarray,
                           gt_normals: np.ndarray,
                           valid_mask: np.ndarray | None = None) -> dict:
    """Angular-error metrics for surface normals.

    Args:
        pred_normals: (..., 3) array. Will be L2-normalized internally.
        gt_normals:   (..., 3) array. Will be L2-normalized internally.
        valid_mask:   optional bool array of shape ``pred_normals.shape[:-1]``.

    Returns:
        dict with mean_deg, median_deg, pct_under_11_25, pct_under_22_5,
        pct_under_30, n_valid.  NaN for percentages/means when no valid pixels.
    """
    pred = np.asarray(pred_normals, dtype=np.float64)
    gt = np.asarray(gt_normals, dtype=np.float64)

    pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
    gt_norm = np.linalg.norm(gt, axis=-1, keepdims=True)
    p = pred / np.clip(pred_norm, EPS, None)
    g = gt / np.clip(gt_norm, EPS, None)

    has_p = (pred_norm.squeeze(-1) > EPS)
    has_g = (gt_norm.squeeze(-1) > EPS)
    valid = has_p & has_g
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)

    n_valid = int(valid.sum())
    if n_valid == 0:
        nan = float("nan")
        return {
            "mean_deg": nan, "median_deg": nan,
            "pct_under_11_25": nan, "pct_under_22_5": nan, "pct_under_30": nan,
            "n_valid": 0,
        }

    dot = np.clip((p * g).sum(axis=-1), -1.0, 1.0)
    err_deg = np.degrees(np.arccos(dot))[valid]

    return {
        "mean_deg": float(err_deg.mean()),
        "median_deg": float(np.median(err_deg)),
        "pct_under_11_25": float((err_deg < 11.25).mean()),
        "pct_under_22_5": float((err_deg < 22.5).mean()),
        "pct_under_30": float((err_deg < 30.0).mean()),
        "n_valid": n_valid,
    }
