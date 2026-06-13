"""Chamfer distance and F-score between predicted and GT point clouds."""
from __future__ import annotations

import numpy as np


def _nn_distances(query: np.ndarray,
                  ref: np.ndarray | None,
                  *,
                  tree=None,
                  workers: int = -1) -> np.ndarray:
    """Nearest-neighbour Euclidean distances from each query to ref.

    Uses scipy.spatial.cKDTree if available; otherwise falls back to a
    chunked torch.cdist computation. When ``tree`` is supplied it is used
    directly and ``ref`` may be ``None``, this saves the per-call KD-tree
    build cost when the same ``ref`` cloud is queried multiple times within
    one scene.

    ``workers`` is forwarded to ``cKDTree.query``: ``-1`` (default) uses all
    available cores, ``1`` is single-threaded. Set to ``1`` when wrapping the
    caller in a process pool to avoid CPU oversubscription.
    """
    n_q = len(query) if query is not None else 0
    if n_q == 0:
        return np.full((0,), np.inf, dtype=np.float64)
    if tree is not None:
        d, _ = tree.query(query, k=1, workers=workers)
        return np.asarray(d, dtype=np.float64)
    if ref is None or len(ref) == 0:
        return np.full((n_q,), np.inf, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree
        _tree = cKDTree(ref)
        d, _ = _tree.query(query, k=1, workers=workers)
        return np.asarray(d, dtype=np.float64)
    except ImportError:
        pass

    # Fallback: torch cdist in chunks
    import torch
    q = torch.as_tensor(query, dtype=torch.float32)
    r = torch.as_tensor(ref, dtype=torch.float32)
    out = torch.empty(len(q), dtype=torch.float32)
    chunk = 4096
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], r, p=2)
        out[i:i + chunk] = d.min(dim=1).values
    return out.numpy().astype(np.float64)


def chamfer_and_fscore(pred_points: np.ndarray,
                       gt_points: np.ndarray,
                       thresholds: Iterable[float] = (0.05, 0.10, 0.20),
                       *,
                       physical_thresholds: dict | None = None,
                       gt_tree=None,
                       pred_tree=None,
                       workers: int = -1,
                       pred_acc_mask: np.ndarray | None = None) -> dict:
    """Symmetric chamfer + per-threshold F-score / precision / recall.

    Args:
        pred_points: (N, 3) predicted world-frame points.
        gt_points:   (M, 3) GT world-frame points.
        thresholds:  iterable of distance thresholds in the same units as the
            point clouds. For metric-space evaluation these are metres
            (defaults to 5/10/20 cm). For normalized-space (vggt_scene)
            evaluation they are unitless normalized-space values.
        physical_thresholds: optional ``{label: threshold}`` mapping for an
            extra set of thresholds in normalized space that represent
            physical-equivalent distances. Used by the vggt_scene eval pass:
            for a physical threshold ``T_m`` in metres, supply
            ``T_m / vggt_scene_scale`` so the threshold check is exactly
            "physical distance < T_m metres" even though the cloud lives in
            normalized units. Output keys: ``fscore_physical_<label>``,
            ``precision_physical_<label>``, ``recall_physical_<label>``.
        gt_tree:     optional prebuilt ``scipy.spatial.cKDTree`` over
            ``gt_points``. When supplied the per-call tree-build for the
            ``pred → gt`` direction is skipped, useful when the same GT
            cloud is queried across multiple alignment tracks.
        pred_tree:   optional prebuilt KD-tree over ``pred_points`` (symmetric).
        workers:     forwarded to ``cKDTree.query``. ``-1`` (default) parallelises
            over all available cores; pass ``1`` when the caller wraps this in
            a process pool to avoid oversubscribing the CPU.
        pred_acc_mask: optional boolean array of length ``len(pred_points)``.
            When supplied, the **pred → gt** direction (``accuracy_mean`` /
            ``precision_*``) is computed only over ``pred_points[pred_acc_mask]``
          , i.e. predicted points whose source pixel has valid GT layout
            depth. Predicted points sourced from GT-undefined pixels have no GT
            counterpart to match against (the GT cloud excludes those pixels)
            and would otherwise be penalised as spurious geometry, so they are
            dropped from accuracy/precision. The **gt → pred** direction
            (``completeness_mean`` / ``recall_*``) is NEVER masked, it always
            uses the full predicted cloud, so completeness/recall are
            unaffected. ``None`` (default) reproduces the historical unmasked
            behaviour exactly. If every selected entry is ``False`` (no
            GT-valid-sourced pred points), ``accuracy_mean`` is ``NaN`` and
            ``precision_*`` is ``0.0`` while completeness/recall stay valid.

    Returns:
        dict with keys:
          - ``accuracy_mean`` (pred → gt, mean nearest-neighbour distance).
          - ``completeness_mean`` (gt → pred).
          - ``chamfer_l1_sum`` = ``accuracy + completeness`` (the historical
            sum convention used by this repo). Doubled relative to the
            ``0.5·(a+c)`` mean used by DTU eval / most scene-reconstruction
            papers.
          - ``chamfer_l1_mean`` = ``0.5 · (accuracy + completeness)``,
            paper-style convention. **Prefer this for new headline tables.**
          - ``chamfer_l2_sum`` / ``chamfer_l2_mean``, same idea for the
            squared-distance variant.
          - ``chamfer_l1`` / ``chamfer_l2``, backward-compatible aliases
            pointing at the *sum* (i.e. ``chamfer_l1_sum`` /
            ``chamfer_l2_sum``). Kept so historical JSON output paths
            continue to parse; new code should read the explicit
            ``_sum`` / ``_mean`` keys instead.
          - ``fscore_<t>``, ``precision_<t>``, ``recall_<t>`` for each ``t``
            in ``thresholds``. ``t`` is formatted to two decimals (e.g.
            ``fscore_0.10``).
          - When ``physical_thresholds`` is supplied:
            ``fscore_physical_<label>``, ``precision_physical_<label>``,
            ``recall_physical_<label>`` for each entry.
        Empty/degenerate inputs return NaN for the means and 0.0 for fscores.
    """
    thresholds = tuple(thresholds)
    out: dict = {}

    if len(pred_points) == 0 or len(gt_points) == 0:
        nan = float("nan")
        out.update({
            "accuracy_mean":    nan,
            "completeness_mean": nan,
            "chamfer_l1_sum":   nan,
            "chamfer_l1_mean":  nan,
            "chamfer_l2_sum":   nan,
            "chamfer_l2_mean":  nan,
            # Backward-compat aliases (point at the sum form).
            "chamfer_l1":       nan,
            "chamfer_l2":       nan,
        })
        for t in thresholds:
            out[f"fscore_{t:.2f}"] = 0.0
            out[f"precision_{t:.2f}"] = 0.0
            out[f"recall_{t:.2f}"] = 0.0
        if physical_thresholds:
            for label in physical_thresholds:
                out[f"fscore_physical_{label}"] = 0.0
                out[f"precision_physical_{label}"] = 0.0
                out[f"recall_physical_{label}"] = 0.0
        return out

    # pred → gt (accuracy / precision). Optionally restrict the *query* set to
    # predicted points whose source pixel has valid GT layout depth. GT-undefined
    # pixels contribute no GT points, so pred points sourced there have no fair
    # GT counterpart; masking them mirrors the 2D metrics and seen/unseen splits,
    # which already ignore GT-undefined pixels. The gt → pred direction
    # (completeness / recall) ALWAYS uses the full predicted cloud and is never
    # masked.
    if pred_acc_mask is None:
        pred_acc_points = pred_points
    else:
        m = np.asarray(pred_acc_mask, dtype=bool).reshape(-1)
        if m.shape[0] != len(pred_points):
            raise ValueError(
                f"pred_acc_mask length {m.shape[0]} != len(pred_points) {len(pred_points)}"
            )
        pred_acc_points = pred_points[m]

    d_pred2gt = _nn_distances(pred_acc_points, gt_points, tree=gt_tree, workers=workers)
    d_gt2pred = _nn_distances(gt_points, pred_points, tree=pred_tree, workers=workers)

    # ``n_acc == 0`` only when pred_acc_mask drops every point (the unmasked
    # path is guarded above by the empty-cloud early return, so n_acc >= 1 there
    # and these branches are byte-identical to the historical behaviour).
    n_acc = int(d_pred2gt.shape[0])
    acc = float(d_pred2gt.mean()) if n_acc else float("nan")
    com = float(d_gt2pred.mean())
    l2_acc = float((d_pred2gt ** 2).mean()) if n_acc else float("nan")
    l2_com = float((d_gt2pred ** 2).mean())

    out["accuracy_mean"]    = acc
    out["completeness_mean"] = com
    out["chamfer_l1_sum"]   = acc + com
    out["chamfer_l1_mean"]  = 0.5 * (acc + com)
    out["chamfer_l2_sum"]   = l2_acc + l2_com
    out["chamfer_l2_mean"]  = 0.5 * (l2_acc + l2_com)
    # Backward-compat aliases: existing JSON / CSV / consumers read
    # `chamfer_l1` / `chamfer_l2`. Both alias the SUM form so historical
    # numbers reproduce bit-for-bit. New code should use the explicit
    # `_sum` / `_mean` keys above (prefer `_mean` for paper-style reporting).
    out["chamfer_l1"]       = out["chamfer_l1_sum"]
    out["chamfer_l2"]       = out["chamfer_l2_sum"]

    for t in thresholds:
        precision = float((d_pred2gt < t).mean()) if n_acc else 0.0  # frac pred near GT
        recall = float((d_gt2pred < t).mean())     # frac GT near pred
        if precision + recall < 1e-12:
            fscore = 0.0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        key = f"{t:.2f}"
        out[f"fscore_{key}"] = fscore
        out[f"precision_{key}"] = precision
        out[f"recall_{key}"] = recall

    if physical_thresholds:
        # Same precision/recall/F1 formula, but with thresholds rescaled to
        # represent a physical-distance equivalent in normalized space.
        # ``label`` is a free-form identifier (e.g. ``"0.05m"``); ``t`` is
        # the threshold value already in normalized units
        # (caller computes ``T_metres / vggt_scene_scale``).
        for label, t in physical_thresholds.items():
            precision = float((d_pred2gt < float(t)).mean()) if n_acc else 0.0
            recall    = float((d_gt2pred < float(t)).mean())
            if precision + recall < 1e-12:
                fscore = 0.0
            else:
                fscore = 2 * precision * recall / (precision + recall)
            out[f"fscore_physical_{label}"]    = fscore
            out[f"precision_physical_{label}"] = precision
            out[f"recall_physical_{label}"]    = recall

    return out

