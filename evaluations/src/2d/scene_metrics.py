"""Per-scene 2D metric orchestrator (depth / mask / normals).

One forward pass per scene produces a ``preds`` dict; this module slices it
per frame and dispatches to the per-concern modules
(:mod:`metrics_depth`, :mod:`metrics_mask`, :mod:`metrics_normals`), then
aggregates per-frame-then-mean to match training-time val semantics.
"""
from __future__ import annotations

from _common import to_np, frame, aggregate
from metrics_depth import compute_depth_metrics_with_splits
from metrics_mask import compute_mask_metrics, mask_prob_from_logits
from metrics_normals import (
    compute_normal_metrics,
    compute_normals_numpy,
    normals_from_pred,
)


def compute_2d_metrics_for_scene(sample: dict, preds: dict,
                                 use_depth_as_layout: bool,
                                 has_mask_head: bool,
                                 has_normal_head: bool) -> dict:
    """Per-scene 2D metrics (averaged across the scene's frames).

    Aggregation is per-frame-then-mean, matching
    ``Trainer._accumulate_val_metrics`` → ``Trainer._finalize_val_metrics``.
    """
    if "layout_depth" in preds:
        ld = to_np(preds["layout_depth"])
        if ld.ndim == 4 and ld.shape[-1] == 1:
            ld = ld[..., 0]
        depth_used = "layout_depth"
    elif use_depth_as_layout and "depth" in preds:
        ld = to_np(preds["depth"])
        if ld.ndim == 4 and ld.shape[-1] == 1:
            ld = ld[..., 0]
        depth_used = "depth"
    else:
        raise KeyError("predictions has neither 'layout_depth' nor 'depth'; "
                       "pass --use_depth_as_layout for E0 vanilla eval")

    gt_ld = to_np(sample["layout_depths"])
    gt_dm = to_np(sample.get("layout_depth_masks"))
    lm = to_np(sample.get("layout_masks"))

    S = ld.shape[0]
    depth_records: list[dict] = []
    mask_records: list[dict] = []
    normal_records: list[dict] = []

    pred_mask_prob = None
    if has_mask_head and "layout_mask_logits" in preds:
        pred_mask_prob = mask_prob_from_logits(to_np(preds["layout_mask_logits"]))

    pred_normals_arr = None
    if has_normal_head and "layout_normal" in preds:
        pred_normals_arr = normals_from_pred(to_np(preds["layout_normal"]))
    gt_normals = to_np(sample.get("layout_normals"))
    gt_normal_masks = to_np(sample.get("layout_normal_masks"))

    for s in range(S):
        gt_s = frame(gt_ld, s)
        valid_s = frame(gt_dm, s).astype(bool) if gt_dm is not None else None
        lm_s = frame(lm, s) if lm is not None else None
        depth_records.append(
            compute_depth_metrics_with_splits(ld[s], gt_s, valid_s, lm_s)
        )

        if pred_mask_prob is not None and lm is not None:
            mask_records.append(
                compute_mask_metrics(pred_mask_prob[s], frame(lm, s), threshold=0.5)
            )

        # Normals: prefer head output; otherwise compute from layout depth.
        if gt_normals is not None:
            gt_n_s = frame(gt_normals, s)
            if pred_normals_arr is not None:
                pred_n_s = pred_normals_arr[s]
            else:
                pred_n_s, _ = compute_normals_numpy(ld[s])
            valid_n = None
            if gt_normal_masks is not None:
                valid_n = frame(gt_normal_masks, s).astype(bool)
            normal_records.append(compute_normal_metrics(pred_n_s, gt_n_s, valid_n))

    out = {"depth_used": depth_used}
    out.update(aggregate(depth_records))
    if mask_records:
        magg = aggregate(mask_records)
        for k, v in magg.items():
            out[f"mask_{k}"] = v
    if normal_records:
        nagg = aggregate(normal_records)
        for k, v in nagg.items():
            out[f"normal_{k}"] = v
    return out
