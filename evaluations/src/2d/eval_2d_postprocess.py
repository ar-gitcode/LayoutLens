#!/usr/bin/env python3
"""Manifest-driven 2D layout-depth eval with 3D geometric post-processing.

Runs **one forward pass per scene** through the VGGT layout-depth head, then
post-processes the predicted depth four ways (raw / RANSAC planes / Manhattan
planes / cuboid box), plus two GT-driven oracle methods (oracle_gt_planes,
oracle_gt_cuboid) for an upper-bound diagnostic, and re-renders each
geometry back to per-view layout-depth, feeding it through the same metric
code used everywhere else in this repo.

Single subcommand::

    run        Iterate the manifest, write per (method, hole_policy) JSONs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# --- repo path setup (flat sys.path bootstrap; see common/_paths.py) --------
_d = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_d) != "src":
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "common"))
import _paths  # noqa: E402: adds repo root, training, all eval subdirs to sys.path
from _paths import REPO_ROOT as _repo_root, TRAINING_DIR as _training_dir  # noqa: E402
os.chdir(_training_dir)

from _common import (                                          # noqa: E402
    to_np,
    frame,
    load_model_and_cfg,
    enable_unique_scene_mode,
    select_split,
    aggregate,
)
from scene_metrics import compute_2d_metrics_for_scene          # noqa: E402
from _oca_eval_helpers import forward_model, cameras_from_sample  # noqa: E402
from eval_metrics import compute_depth_metrics_with_splits      # noqa: E402

# Manifest helpers come from common/manifest.py; per-frame scale align lives in
# eval_2d.py. Reuse both to stay consistent with the rest of the eval surface.
from manifest import (                                         # noqa: E402
    _load_eval_manifest,
    _check_manifest_split,
    _manifest_iter_items,
    _find_room_envelopes_child,
    _resolve_seq_index,
)
from eval_2d import _scale_align_preds_per_frame                # noqa: E402

from training.geometry.room_envelope_geometry import (         # noqa: E402
    depth_to_world_points,
    mask_valid_points,
    sample_points,
    save_point_cloud_ply,
)
from training.geometry.postprocess_planes import (             # noqa: E402
    fit_ransac_envelope,
    snap_to_manhattan,
)
from training.geometry.postprocess_cuboid import fit_cuboid_room  # noqa: E402
from training.geometry.render_planes_to_depth import (         # noqa: E402
    render_planes_to_zdepth_batch,
)


METHODS_ALL = ("raw", "ransac", "manhattan", "cuboid",
               "oracle_planes", "oracle_cuboid")

# Methods that consume model predictions (failure → fall back to raw_pred
# for headline metric parity, fix #3 / correction #3).
MODEL_METHODS = ("ransac", "manhattan", "cuboid")
# Methods that consume GT (failure → record NaN and oracle_failure=True;
# never fall back to raw_pred, fix #3).
ORACLE_METHODS = ("oracle_planes", "oracle_cuboid")


# ---------------------------------------------------------------------------
# Scale alignment
# ---------------------------------------------------------------------------

def _scale_shift_per_frame(pred_ld_S: np.ndarray, gt_ld_S: np.ndarray,
                            gt_valid_S: np.ndarray | None,
                            ) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    """Per-frame linear scale+shift: ``aligned = s · pred + t``.

    Ports the linear lstsq fit from
    ``vggt_layout_baselines/eval/metrics.py:align_depth(mode='scale_shift')``.
    After applying the fit, non-positive aligned depths are clamped to 0 so
    they are treated as invalid by everything downstream (fix #8).

    Returns ``(aligned_S, s_list, t_list, clipped_frac_list)``.
    """
    S, H, W = pred_ld_S.shape
    aligned = pred_ld_S.copy().astype(np.float32)
    ss: list[float] = []
    tt: list[float] = []
    clipped: list[float] = []
    for s in range(S):
        gt_s = gt_ld_S[s]
        if gt_valid_S is not None:
            valid = gt_valid_S[s].astype(bool) & (gt_s > 1e-6) & (pred_ld_S[s] > 1e-6)
        else:
            valid = (gt_s > 1e-6) & (pred_ld_S[s] > 1e-6)
        n = int(valid.sum())
        if n < 32:
            ss.append(float("nan"))
            tt.append(float("nan"))
            clipped.append(0.0)
            continue
        p = pred_ld_S[s][valid].astype(np.float64)
        g = gt_s[valid].astype(np.float64)
        A = np.stack([p, np.ones_like(p)], axis=1)
        sol, *_ = np.linalg.lstsq(A, g, rcond=None)
        s_val, t_val = float(sol[0]), float(sol[1])
        a = pred_ld_S[s].astype(np.float64) * s_val + t_val
        # Fix #8: clamp non-positives so they don't break downstream geometry.
        clipped_frac = float((a <= 1e-3).mean())
        a = np.where(a > 1e-3, a, 0.0).astype(np.float32)
        aligned[s] = a
        ss.append(s_val)
        tt.append(t_val)
        clipped.append(clipped_frac)
    return aligned, ss, tt, clipped


def apply_scale_align(pred_ld_S: np.ndarray, gt_ld_S: np.ndarray,
                       gt_valid_S: np.ndarray | None, mode: str
                       ) -> tuple[np.ndarray, dict]:
    """Apply --scale_align mode to the predicted depth, BEFORE unprojection.

    Returns ``(aligned, diagnostics)``. Modes::

        none: passthrough.
        per_frame: median(gt/pred) per frame (matches eval_2d.py default
                     for monocular checkpoints).
        per_scene: single median(gt/pred) over the whole scene.
        scale_shift, linear lstsq per frame (old-repo default), with
                     non-positive depths clamped to 0 (fix #8).
    """
    if mode == "none":
        return pred_ld_S.astype(np.float32), {"scale_align": "none"}

    if mode == "per_frame":
        # Reuse the helper from eval_2d.py. It expects a preds_one dict.
        preds_one = {"layout_depth": pred_ld_S[..., None]}    # (S,H,W,1)
        sample_like = {"layout_depths": gt_ld_S,
                       "layout_depth_masks": gt_valid_S}
        out, scales = _scale_align_preds_per_frame(
            preds_one, sample_like, use_depth_as_layout=False,
        )
        aligned = out["layout_depth"][..., 0]
        return aligned.astype(np.float32), {
            "scale_align": "per_frame",
            "scales": scales,
            "median_scale": float(np.nanmedian(scales)) if scales else float("nan"),
        }

    if mode == "per_scene":
        if gt_valid_S is not None:
            valid = gt_valid_S.astype(bool) & (gt_ld_S > 1e-6) & (pred_ld_S > 1e-6)
        else:
            valid = (gt_ld_S > 1e-6) & (pred_ld_S > 1e-6)
        if int(valid.sum()) < 32:
            return pred_ld_S.astype(np.float32), {
                "scale_align": "per_scene", "scale": float("nan")
            }
        p = pred_ld_S[valid].astype(np.float64)
        g = gt_ld_S[valid].astype(np.float64)
        scale = float(np.median(g / np.clip(p, 1e-6, None)))
        aligned = pred_ld_S.astype(np.float32) * scale
        return aligned, {"scale_align": "per_scene", "scale": scale}

    if mode == "scale_shift":
        aligned, ss, tt, clipped = _scale_shift_per_frame(
            pred_ld_S, gt_ld_S, gt_valid_S,
        )
        return aligned, {
            "scale_align":              "scale_shift",
            "s":                        ss,
            "t":                        tt,
            "scale_shift_clipped_frac": clipped,
            "mean_clipped_frac":        float(np.mean(clipped) if clipped else float("nan")),
        }

    raise ValueError(f"unknown --scale_align mode: {mode!r}")


# ---------------------------------------------------------------------------
# Fused point cloud + room interior anchor
# ---------------------------------------------------------------------------

def _fuse_world_cloud(depth_S: np.ndarray,
                      K_S: np.ndarray,
                      E_S: np.ndarray,
                      *,
                      max_points: int,
                      seed: int = 0,
                      ) -> np.ndarray:
    """Unproject every valid pixel of every view to world frame, concatenate,
    then randomly downsample to ``max_points``.
    """
    chunks: list[np.ndarray] = []
    for s in range(depth_S.shape[0]):
        d = depth_S[s]
        valid = d > 1e-6
        if not valid.any():
            continue
        world = depth_to_world_points(d, K_S[s], E_S[s])
        chunks.append(mask_valid_points(world, valid))
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.concatenate(chunks, axis=0)
    return sample_points(pts, max_points=max_points, seed=seed)


def _camera_centres(E_S: np.ndarray) -> np.ndarray:
    """Return camera centres in world frame, shape (S, 3)."""
    from vggt.utils.geometry import closed_form_inverse_se3
    S = E_S.shape[0]
    E4 = np.tile(np.eye(4, dtype=np.float32)[None], (S, 1, 1))
    E4[:, :3, :] = E_S.astype(np.float32)
    c2w = closed_form_inverse_se3(E4)
    return c2w[:, :3, 3].astype(np.float32)


# ---------------------------------------------------------------------------
# Mask-policy metric helper (correction #2, fix #7)
# ---------------------------------------------------------------------------

def _compute_2d_metrics_with_render_mask(sample: dict,
                                          depth_S: np.ndarray,
                                          render_valid_S: np.ndarray) -> dict:
    """Per-scene 2D depth metrics with an extra AND-ed render_valid_mask.

    Mirrors ``_common.compute_2d_metrics_for_scene`` for the depth-only path,
    but combines ``layout_depth_mask`` with ``render_valid_mask`` before
    feeding the per-frame ``compute_depth_metrics_with_splits``. We do this
    locally (not by changing ``compute_2d_metrics_for_scene``) so the
    canonical metric path used by the rest of the repo stays untouched.
    """
    gt_ld = to_np(sample["layout_depths"])
    gt_dm = to_np(sample.get("layout_depth_masks"))
    lm = to_np(sample.get("layout_masks"))

    S = depth_S.shape[0]
    records: list[dict] = []
    for s in range(S):
        gt_s = frame(gt_ld, s)
        gt_valid_s = frame(gt_dm, s).astype(bool) if gt_dm is not None else None
        render_valid_s = render_valid_S[s].astype(bool)
        if gt_valid_s is None:
            combined = render_valid_s
        else:
            combined = gt_valid_s & render_valid_s
        lm_s = frame(lm, s) if lm is not None else None
        records.append(
            compute_depth_metrics_with_splits(depth_S[s], gt_s, combined, lm_s)
        )
    out: dict = {"depth_used": "layout_depth_post"}
    out.update(aggregate(records))
    return out


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

def _post_planes(fused_pts: np.ndarray, args) -> tuple[list[dict], dict]:
    """RANSAC plane fit on a world-frame point cloud."""
    planes = fit_ransac_envelope(
        fused_pts,
        max_planes=args.ransac_max_planes,
        thresh=args.ransac_thresh,
        min_inliers=args.ransac_min_inliers,
        max_iters=args.ransac_max_iters,
        seed=args.seed,
        extent_quantiles=tuple(args.plane_extent_quantile),
        vectorized=getattr(args, "ransac_vectorized", False),
    )
    if not planes:
        return [], {"plane_status": "no_planes", "n_planes": 0}
    return planes, {
        "plane_status": "ok",
        "n_planes": len(planes),
        "mean_inlier_ratio": float(np.mean([p["inlier_ratio"] for p in planes])),
        "mean_residual": float(np.mean([p["mean_residual"] for p in planes])),
    }


def _post_manhattan(fused_pts: np.ndarray, args) -> tuple[list[dict], dict]:
    base_planes, base_status = _post_planes(fused_pts, args)
    if not base_planes:
        return [], {"manhattan_status": "no_input_planes", **base_status}
    snapped, mstatus = snap_to_manhattan(
        base_planes,
        fused_pts,
        angle_tol_deg=args.manhattan_angle_tol_deg,
        merge_tol=args.manhattan_merge_tol,
        extent_quantiles=tuple(args.plane_extent_quantile),
    )
    diag = {**base_status, **mstatus, "n_planes": len(snapped)}
    if not snapped:
        return [], diag
    return snapped, diag


def _post_cuboid(fused_pts: np.ndarray, args,
                  manhattan_basis: np.ndarray | None) -> tuple[list[dict], dict]:
    faces, status = fit_cuboid_room(
        fused_pts,
        method=args.cuboid_method,
        manhattan_basis=manhattan_basis,
        quantile=tuple(args.cuboid_quantile),
        inlier_thresh=args.ransac_thresh,
        min_points=args.ransac_min_inliers,
        min_box_dim=args.cuboid_min_box_dim,
    )
    if status.get("cuboid_status") != "ok":
        return [], status
    return faces, status


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def _process_scene(sample: dict, model, device, args
                   ) -> dict | None:
    """Run forward + post-processing + metrics for one scene.

    Returns a dict keyed by ``f"{method}__{hole_policy}"`` whose values are
    per-scene metric records produced by ``compute_2d_metrics_for_scene``
    (or the local mask-helper). Returns None if forward fails or layout
    depth is absent.
    """
    import torch

    imgs = np.array(sample["images"])
    if imgs.ndim != 4:
        return None
    imgs_t = (torch.tensor(imgs, dtype=torch.float32)
              .permute(0, 3, 1, 2) / 255.0).unsqueeze(0).to(device)

    K_t, E_t = cameras_from_sample(sample, device=device)
    with torch.no_grad():
        preds = forward_model(model, imgs_t, intrinsics=K_t, extrinsics=E_t)

    # Pick the depth tensor: layout_depth head if present, else fall back to
    # the regular depth head when --use_depth_as_layout is set (E0 vanilla).
    depth_key = None
    if "layout_depth" in preds:
        depth_key = "layout_depth"
    elif args.use_depth_as_layout and "depth" in preds:
        depth_key = "depth"
    if depth_key is None:
        return None

    pred_ld_S = preds[depth_key][0].squeeze(-1).cpu().numpy()  # (S, H, W)
    S, H, W = pred_ld_S.shape
    gt_ld_S = np.asarray(sample["layout_depths"], dtype=np.float32)
    gt_dm_S = np.asarray(sample.get("layout_depth_masks"))           # may be None
    K_S = np.stack([np.asarray(k, dtype=np.float32) for k in sample["intrinsics"]], 0)
    E_S = np.stack([np.asarray(e, dtype=np.float32) for e in sample["extrinsics"]], 0)

    # --- scale alignment BEFORE unprojection (fix #8) ----------------------
    pred_aligned, scale_diag = apply_scale_align(
        pred_ld_S, gt_ld_S,
        gt_dm_S if sample.get("layout_depth_masks") is not None else None,
        args.scale_align,
    )

    # --- fuse + interior anchor --------------------------------------------
    fused_pred = _fuse_world_cloud(pred_aligned, K_S, E_S,
                                    max_points=args.max_fuse_points, seed=args.seed)
    fused_gt = _fuse_world_cloud(gt_ld_S.astype(np.float32), K_S, E_S,
                                  max_points=args.max_fuse_points, seed=args.seed)
    camera_centres = _camera_centres(E_S)
    interior_pt = camera_centres.mean(axis=0)

    # --- run the four model-based geometries -------------------------------
    plane_sets: dict[str, dict] = {}

    if "ransac" in args.methods or "manhattan" in args.methods or "cuboid" in args.methods:
        if "ransac" in args.methods:
            planes_r, diag_r = _post_planes(fused_pred, args)
            plane_sets["ransac"] = {"planes": planes_r, "diag": diag_r}
        if "manhattan" in args.methods:
            planes_m, diag_m = _post_manhattan(fused_pred, args)
            plane_sets["manhattan"] = {"planes": planes_m, "diag": diag_m}
        if "cuboid" in args.methods:
            manhattan_basis = None
            if args.cuboid_method == "from_manhattan" and "manhattan" in plane_sets:
                # Reuse the basis discovered for the Manhattan path.
                mb = plane_sets["manhattan"]["diag"].get("basis")
                if mb is not None:
                    manhattan_basis = np.asarray(mb)
            faces_c, diag_c = _post_cuboid(fused_pred, args, manhattan_basis)
            plane_sets["cuboid"] = {"planes": faces_c, "diag": diag_c}

    # --- and the two GT-driven oracles -------------------------------------
    if "oracle_planes" in args.methods:
        planes_or, diag_or = _post_planes(fused_gt, args)
        plane_sets["oracle_planes"] = {"planes": planes_or, "diag": diag_or}
    if "oracle_cuboid" in args.methods:
        faces_oc, diag_oc = fit_cuboid_room(
            fused_gt, method=args.cuboid_method,
            quantile=tuple(args.cuboid_quantile),
            inlier_thresh=args.ransac_thresh,
            min_points=args.ransac_min_inliers,
            min_box_dim=args.cuboid_min_box_dim,
        )
        plane_sets["oracle_cuboid"] = {"planes": faces_oc, "diag": diag_oc}

    # --- render every plane set into every view ----------------------------
    rendered_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, entry in plane_sets.items():
        planes = entry["planes"]
        if not planes:
            rendered_cache[name] = (None, None)  # failure marker
            continue
        anchor = interior_pt if name in MODEL_METHODS else camera_centres.mean(axis=0)
        d, m = render_planes_to_zdepth_batch(
            planes, K_S, E_S, H, W, anchor,
            min_depth=args.render_min_depth, max_depth=args.render_max_depth,
        )
        rendered_cache[name] = (d, m)
        entry["diag"]["mean_render_coverage"] = float(m.mean())

    # --- per (method, hole_policy) metric records --------------------------
    records: dict[str, dict] = {}

    # `raw` (baseline), identical for fill and mask.
    if "raw" in args.methods:
        preds_for_metric = dict(preds)
        preds_for_metric["layout_depth"] = pred_aligned[..., None]
        m_raw = compute_2d_metrics_for_scene(
            sample,
            {"layout_depth": pred_aligned[..., None]},
            use_depth_as_layout=False,
            has_mask_head=False,
            has_normal_head=False,
        )  # we always feed "layout_depth" into the metric, regardless of head source
        m_raw["failure"] = False
        m_raw["render_coverage"] = 1.0
        for policy in args.render_holes:
            records[f"raw__{policy}"] = dict(m_raw)
            records[f"raw__{policy}"]["scale_diag"] = scale_diag

    # Other methods.
    for method in args.methods:
        if method == "raw":
            continue
        entry = plane_sets.get(method, {})
        rendered, render_valid = rendered_cache.get(method, (None, None))
        is_oracle = method in ORACLE_METHODS
        diag = entry.get("diag", {})
        failure = (rendered is None)

        for policy in args.render_holes:
            key = f"{method}__{policy}"
            if failure:
                if is_oracle:
                    # Oracles do NOT fall back to raw (fix #3). Emit NaNs.
                    metric = _nan_record(sample, S, has_visible="layout_masks" in sample)
                    metric["oracle_failure"] = True
                else:
                    # Model-based methods fall back to raw_pred for that scene.
                    metric = compute_2d_metrics_for_scene(
                        sample,
                        {"layout_depth": pred_aligned[..., None]},
                        use_depth_as_layout=False,
                        has_mask_head=False,
                        has_normal_head=False,
                    )
                    metric["failure"] = True
                metric["render_coverage"] = 0.0
                metric["method_diag"] = diag
                metric["scale_diag"] = scale_diag
                records[key] = metric
                continue

            # We have rendered geometry. Apply the hole policy.
            if policy == "fill":
                depth_for_metric = np.where(render_valid, rendered, pred_aligned)
                metric = compute_2d_metrics_for_scene(
                    sample,
                    {"layout_depth": depth_for_metric[..., None]},
                    use_depth_as_layout=False,
                    has_mask_head=False,
                    has_normal_head=False,
                )
            elif policy == "mask":
                metric = _compute_2d_metrics_with_render_mask(
                    sample, rendered, render_valid,
                )
            elif policy == "zero":
                metric = compute_2d_metrics_for_scene(
                    sample,
                    {"layout_depth": rendered[..., None]},
                    use_depth_as_layout=False,
                    has_mask_head=False,
                    has_normal_head=False,
                )
            else:
                raise ValueError(f"unknown hole policy: {policy!r}")
            metric["failure"] = False
            if is_oracle:
                metric["oracle_failure"] = False
            metric["render_coverage"] = diag.get("mean_render_coverage", float("nan"))
            metric["method_diag"] = diag
            metric["scale_diag"] = scale_diag
            records[key] = metric

    return records


def _nan_record(sample: dict, S: int, has_visible: bool) -> dict:
    """A metric record full of NaNs (for oracle failures, fix #3)."""
    nan = float("nan")
    out = {"depth_used": "oracle_failed"}
    metric_keys = ("absrel", "rmse", "log_rmse", "delta1", "delta2", "delta3", "silog")
    for k in metric_keys:
        for sub in ("all", "visible", "occluded"):
            out[f"{k}_{sub}"] = nan
    for sub in ("all", "visible", "occluded"):
        out[f"n_valid_{sub}"] = 0
    return out


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    import torch
    from hydra.utils import instantiate

    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.render_holes = [h.strip() for h in args.render_holes.split(",") if h.strip()]
    args.plane_extent_quantile = [float(x) for x in args.plane_extent_quantile.split(",")]
    args.cuboid_quantile = [float(x) for x in args.cuboid_quantile.split(",")]

    # Normalize: "ransac" / "manhattan" / "cuboid" / "raw" / "oracle_planes" / "oracle_cuboid".
    unknown = [m for m in args.methods if m not in METHODS_ALL]
    if unknown:
        print(f"[fatal] unknown methods: {unknown}; choose from {METHODS_ALL}",
              file=sys.stderr)
        return 2

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg, model, device, ckpt_meta = load_model_and_cfg(
        args.config, args.checkpoint, args.device,
    )
    has_layout_head = bool(getattr(cfg.model, "enable_layout_depth", False))
    if not has_layout_head and not args.use_depth_as_layout:
        print("[fatal] cfg.model.enable_layout_depth=False and "
              "--use_depth_as_layout not set. Pass --use_depth_as_layout "
              "to evaluate vanilla VGGT (E0) using the regular depth head.",
              file=sys.stderr)
        return 2
    print(f"[heads] layout_depth={has_layout_head} "
          f"use_depth_as_layout={args.use_depth_as_layout}")

    split_cfg = select_split(cfg, args.split)
    inner_ds = instantiate(
        split_cfg.dataset,
        common_config=split_cfg.common_config,
        _recursive_=False,
    )
    n_unique = enable_unique_scene_mode(inner_ds)

    manifest = _load_eval_manifest(args.manifest)
    _check_manifest_split(manifest, args.split, allow_split_mismatch=False)
    manifest_items, manifest_mode, manifest_num_views = _manifest_iter_items(manifest, None)
    re_child = _find_room_envelopes_child(inner_ds)
    re_child.inside_random = False
    scene_cam_lookup = {seq["scene_cam"]: i for i, seq in enumerate(re_child.sequences)}

    n_total = len(manifest_items)
    if args.max_samples is not None and args.max_samples > 0:
        n_total = min(n_total, args.max_samples)

    print("=" * 70)
    print(f"  config          : {args.config}")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  split           : {args.split}  ({n_unique} unique seq, manifest items: {len(manifest_items)})")
    print(f"  manifest        : {args.manifest}  mode={manifest_mode}")
    print(f"  methods         : {args.methods}")
    print(f"  render_holes    : {args.render_holes}")
    print(f"  scale_align     : {args.scale_align}")
    print(f"  ransac defaults : max_planes={args.ransac_max_planes} thresh={args.ransac_thresh} "
          f"min_inliers={args.ransac_min_inliers} iters={args.ransac_max_iters}")
    print(f"  manhattan       : angle_tol={args.manhattan_angle_tol_deg}° "
          f"merge_tol={args.manhattan_merge_tol}")
    print(f"  cuboid          : method={args.cuboid_method} q={args.cuboid_quantile}")
    print(f"  max_samples     : {n_total}")
    print(f"  output_dir      : {args.output_dir}")
    print(f"  save_debug_every: {args.save_debug_every}")
    print("=" * 70)

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug_every and args.save_debug_every > 0:
        (out_dir / "debug_visuals").mkdir(exist_ok=True)

    # per_method_records[key] = list of per-scene metric dicts
    per_method_records: dict[str, list[dict]] = {}
    n_skipped = 0
    t_start = time.time()

    for i in range(n_total):
        item = manifest_items[i]
        try:
            seq_index = _resolve_seq_index(item, scene_cam_lookup)
            sample = re_child.get_data(
                seq_index=seq_index,
                img_per_seq=int(item["num_views"]),
                ids=list(item["ids"]),
                aspect_ratio=1.0,
            )
        except Exception as e:
            n_skipped += 1
            print(f"[manifest item {i}] dataset error ({e}); skipping")
            continue

        try:
            records = _process_scene(sample, model, device, args)
        except Exception as e:
            n_skipped += 1
            print(f"[scene {i} scene_cam={item.get('scene_cam')}] process error ({e}); skipping",
                  flush=True)
            continue
        if records is None:
            n_skipped += 1
            continue

        for key, rec in records.items():
            per_method_records.setdefault(key, []).append(rec)

        if args.save_debug_every > 0 and (i % args.save_debug_every) == 0:
            try:
                _dump_debug(out_dir / "debug_visuals" / f"sample{i:04d}",
                            sample, records)
            except Exception as e:
                print(f"[debug {i}] dump error ({e}); continuing", flush=True)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  {i+1}/{n_total} done in {elapsed:.1f}s "
                  f"({(i+1)/max(elapsed,1e-6):.2f} scenes/s, skipped={n_skipped})")

    # --- write per (method, hole_policy) JSONs -----------------------------
    method_summaries: dict[str, dict] = {}
    for key, recs in per_method_records.items():
        method, hole = key.split("__", 1)
        is_oracle = method in ORACLE_METHODS
        agg = aggregate(recs)
        n_failures = sum(1 for r in recs if r.get("failure", False))
        n_oracle_failures = sum(1 for r in recs if r.get("oracle_failure", False))

        # Success-only diagnostic table (over rows without method-side failure).
        if is_oracle:
            ok_recs = [r for r in recs if not r.get("oracle_failure", False)]
        else:
            ok_recs = [r for r in recs if not r.get("failure", False)]
        agg_ok = aggregate(ok_recs)

        summary = {
            "method":           method,
            "render_holes":     hole,
            "config":           args.config,
            "checkpoint":       args.checkpoint,
            "split":            args.split,
            "manifest":         args.manifest,
            "scale_align":      args.scale_align,
            "n_samples":        n_total,
            "n_evaluated":      len(recs),
            "n_failures":       n_failures,
            "failure_rate":     float(n_failures / max(len(recs), 1)),
            "n_oracle_failures": n_oracle_failures,
            "oracle_failure_rate": float(n_oracle_failures / max(len(recs), 1)),
            "headline_metrics": agg,
            "success_only_diagnostics": agg_ok,
            "render_coverage_mean": _safe_mean(recs, "render_coverage"),
            "config_overrides": vars(args),
        }
        method_summaries[key] = summary

        json_path = out_dir / f"{method}__{hole}.json"
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=_json_safe)

    # Synthesize raw_pred__mask as a copy of raw_pred__fill (fix #2, table
    # symmetry; raw has no render_valid_mask so the policy is a no-op).
    if "raw" in args.methods and "fill" in args.render_holes and "mask" in args.render_holes:
        src = method_summaries.get("raw__fill")
        if src is not None and "raw__mask" in method_summaries:
            # We already created raw__mask = raw__fill in _process_scene; here
            # we explicitly mark it for posterity.
            method_summaries["raw__mask"]["note"] = (
                "identical to raw__fill; raw_pred has no render_valid_mask. "
                "Emitted for table symmetry (plan fix #2)."
            )
            method_summaries["raw__mask"]["render_coverage_mean"] = 1.0
            with open(out_dir / "raw__mask.json", "w") as fh:
                json.dump(method_summaries["raw__mask"], fh, indent=2, default=_json_safe)

    elapsed = time.time() - t_start
    print(f"\n[done] {n_total} scenes, skipped {n_skipped}, in {elapsed:.1f}s")
    print(f"[done] wrote {len(method_summaries)} JSONs to {out_dir}")
    return 0


def _safe_mean(recs: list[dict], key: str) -> float:
    vals = [float(r.get(key, np.nan)) for r in recs]
    arr = np.array(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"unserializable: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Debug viz
# ---------------------------------------------------------------------------

def _dump_debug(dirpath: Path, sample: dict, records: dict) -> None:
    """Write a few PNGs + small JSON files for a single scene's first view.

    We avoid matplotlib to keep dependencies minimal, use numpy + imageio if
    available, else skip the visualisation.
    """
    import imageio.v2 as imageio
    dirpath.mkdir(parents=True, exist_ok=True)

    imgs = np.asarray(sample["images"])[0]  # (H, W, 3) uint8
    gt = np.asarray(sample["layout_depths"])[0]
    imageio.imwrite(dirpath / "00_rgb.png", imgs.astype(np.uint8))
    imageio.imwrite(dirpath / "01_gt_layout_depth.png",
                    _depth_to_u16(gt))
    # Per-method depth (first frame) is not directly carried in `records`
    # (we threw the per-frame arrays away). Dump the per-method metric
    # records so debug visuals still contain the headline numbers.
    with open(dirpath / "metrics.json", "w") as fh:
        json.dump({k: {kk: vv for kk, vv in r.items() if isinstance(vv, (int, float))}
                   for k, r in records.items()},
                  fh, indent=2, default=_json_safe)


def _depth_to_u16(depth_m: np.ndarray, max_m: float = 10.0) -> np.ndarray:
    d = np.clip(depth_m, 0, max_m) / max_m
    return (d * 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run one evaluation pass")
    pr.add_argument("--config", required=True,
                    help="Hydra config name, e.g. room_envelopes/e4_layout_depth_mask_normals_frozen")
    pr.add_argument("--checkpoint", default=None,
                    help="Optional. E0 vanilla VGGT does not require one.")
    pr.add_argument("--use_depth_as_layout", action="store_true", default=False,
                    help="When the model has no layout-depth head (E0), use "
                         "preds['depth'] as the layout-depth proxy.")
    pr.add_argument("--split", default="val")
    pr.add_argument("--manifest", required=True,
                    help="path to the eval manifest JSON")
    pr.add_argument("--extrinsics_manifest", required=False,
                    help="path to extrinsics_manifest.npz (informational; the "
                         "manifest is consumed via the dataset class)")
    pr.add_argument("--methods", default="raw,ransac,manhattan,cuboid,oracle_planes,oracle_cuboid",
                    help="comma-separated subset of "
                         "{raw, ransac, manhattan, cuboid, oracle_planes, oracle_cuboid}")
    pr.add_argument("--render_holes", default="fill,mask",
                    help="comma-separated subset of {fill, mask, zero}")
    pr.add_argument("--scale_align", default="none",
                    choices=("none", "per_frame", "per_scene", "scale_shift"))
    # RANSAC defaults match vggt_layout_baselines (fix #1).
    pr.add_argument("--ransac_max_planes", type=int, default=6)
    pr.add_argument("--ransac_thresh", type=float, default=0.03)
    pr.add_argument("--ransac_min_inliers", type=int, default=500)
    pr.add_argument("--ransac_max_iters", type=int, default=1000)
    pr.add_argument("--ransac_vectorized", action="store_true",
                    help="Use batched/adaptive RANSAC (scores K hypotheses per "
                         "matmul + Fischler-Bolles early-stop at p=0.99). "
                         "Logically equivalent to the scalar path but RNG draw "
                         "order differs, so inlier masks are not byte-identical.")
    pr.add_argument("--manhattan_angle_tol_deg", type=float, default=20.0)
    pr.add_argument("--manhattan_merge_tol", type=float, default=0.06)
    pr.add_argument("--plane_extent_quantile", default="0.01,0.99")
    pr.add_argument("--cuboid_method", default="pca_aabb",
                    choices=("pca_aabb", "from_manhattan"))
    pr.add_argument("--cuboid_quantile", default="0.01,0.99")
    pr.add_argument("--cuboid_min_box_dim", type=float, default=0.5)
    pr.add_argument("--render_min_depth", type=float, default=0.05)
    pr.add_argument("--render_max_depth", type=float, default=50.0)
    pr.add_argument("--max_fuse_points", type=int, default=200_000)
    pr.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    pr.add_argument("--save-debug-every", dest="save_debug_every", type=int, default=0)
    pr.add_argument("--output-dir", dest="output_dir", required=True)
    pr.add_argument("--device", default=None)
    pr.add_argument("--seed", type=int, default=42)
    pr.set_defaults(func=cmd_run)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
