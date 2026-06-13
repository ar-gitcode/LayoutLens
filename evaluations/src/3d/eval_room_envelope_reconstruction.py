#!/usr/bin/env python3
"""End-to-end room-envelope evaluation for VGGT on the Room Envelopes dataset.

Computes per-view 2D metrics (layout depth / mask / normals on the
all/visible/occluded splits) and per-scene 3D reconstruction metrics
(chamfer, F-score; plus seen/unseen splits) by back-projecting predicted
layout depth into the world frame using GT cameras. The predicted cloud covers
the full amodal envelope (gated only by depth validity); it is never filtered by
a *visibility* (layout) mask. For the overall 3D metrics, the pred→GT side
(accuracy / precision) ignores predicted points whose source pixel has
*undefined* GT layout depth, those pixels have no GT to compare against (they
are excluded from the GT cloud), so plausible predictions there are not
penalised as spurious geometry. The GT→pred side (completeness / recall) still
uses the full predicted cloud and is unchanged.

Camera convention: dataset extrinsics are (3,4) **camera-from-world**
OpenCV matrices. See evaluations/src/3d/pointcloud.py for details.

Smoke test (run from repo root):

  python evaluations/src/3d/eval_room_envelope_reconstruction.py \\
      --config room_envelopes/e1b_uf12_layout_depth_only_original_regression \\
      --checkpoint /path/to/checkpoint.pt \\
      --split val \\
      --max_batches 2 \\
      --output_dir outputs/eval_smoke \\
      --save_debug_every 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- repo path setup (flat sys.path bootstrap; see common/_paths.py) --------
_d = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_d) != "src":
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "common"))
import _paths  # noqa: E402: adds repo root, training, all eval subdirs to sys.path
from _paths import REPO_ROOT as _repo_root, TRAINING_DIR as _training_dir  # noqa: E402
# Hydra config_path resolution and certain imports expect cwd == training/.
os.chdir(_training_dir)

# Shared eval helpers: model + cfg loading, split selection, unique-scene
# determinism, NaN-safe aggregation (common/_common.py); the per-scene 2D
# metric orchestrator lives in 2d/scene_metrics.py.
from _common import (  # noqa: E402
    to_np as _to_np,
    frame as _frame,
    load_model_and_cfg as _load_model_and_cfg_impl,
    enable_unique_scene_mode as _enable_unique_scene_mode,
    select_split as _select_split,
    aggregate as _aggregate,
)
from scene_metrics import compute_2d_metrics_for_scene as _compute_2d_metrics_for_scene  # noqa: E402

from _oca_eval_helpers import forward_model, cameras_from_sample  # noqa: E402
from tensor_utils import _to_image_tensor, _strip_batch_dim  # noqa: E402

from _timing import StageTimer  # noqa: E402
import _cli  # noqa: E402: shared argparse flag helpers

from pointcloud import (  # noqa: E402
    backproject_depth_to_world,
    build_kdtree,
    build_scene_pointcloud_from_batch,
    pred_cloud_gtvalid_mask,
    sample_pointcloud,
    sample_pointcloud_with_companion,
)
from metrics_chamfer import chamfer_and_fscore  # noqa: E402
from alignment import (  # noqa: E402
    align_pointcloud_scale_only,
    align_pred_cameras_to_gt,
    fit_xyz_scale_zshift_lstsq,
)
from pose import (  # noqa: E402
    compute_pose_error_metrics,
    decode_pred_pose_enc,
)
from diagnostics import bbox_size  # noqa: E402
from depth_scale import (  # noqa: E402
    compute_median_depth_scale,
    summarize_depth_scale,
)
from normalization import normalize_sample_vggt_scene  # noqa: E402
from ply_io import prefix_metrics, write_ply  # noqa: E402


def _load_model_and_cfg(args):
    """Backwards-compatible wrapper around ``_common.load_model_and_cfg``.

    Drops the ``ckpt_meta`` block (this script doesn't use it) so the
    return signature matches the pre-extraction code exactly.
    """
    cfg, model, device, _ = _load_model_and_cfg_impl(
        args.config, args.checkpoint, args.device,
    )
    return cfg, model, device


# ---------------------------------------------------------------------------
# Alignment-track resolver
# ---------------------------------------------------------------------------

def _resolve_alignment_tracks(alignment_arg: str, camera_mode: str, view_count=None):
    """Map ``--alignment`` + ``--camera_mode`` (+ optional ``view_count``) → flags
    tuple ``(do_raw, do_scale, do_sim3, do_scale_shift_cam, headline)``.

    The ``auto`` policy is **view-count aware** when ``view_count`` is supplied
    (the orchestrator passes the per-manifest N). When ``view_count`` is ``None``
    (unknown / mixed / the standalone runner) it falls back to the historic
    camera-mode-only policy, so existing standalone-engine results reproduce.

    auto + gt   + 1-view     → raw + scale + scale_shift_cam; headline = scale
    auto + gt   + multi/None → raw + scale;                   headline = scale
    auto + pred + 1-view     → raw + scale + scale_shift_cam; headline = scale
                               (scale_shift_cam is a *diagnostic* here: it is fit
                                in the predicted camera frame, so its quality is
                                bounded by predicted intrinsics/pose reliability,
                                the intrinsics-free ``scale`` stays the headline.)
    auto + pred + multi/None → raw + sim3;                    headline = sim3
    none               → raw only;                     headline = raw
    scale              → raw + scale;                  headline = scale
    sim3               → raw + sim3;                   headline = sim3
    scale_shift /
    scale_shift_cam    → raw + scale_shift_cam;        headline = scale_shift_cam
    all                → raw + scale + sim3;           headline = scale (gt) / sim3 (pred)
    all_cam            → raw + scale + scale_shift_cam; headline = scale_shift_cam

    ``scale_shift`` is the public alias for ``scale_shift_cam``: the camera-frame
    Room-Envelopes / LaRI alignment ``min Σ ||s·P_cam + (0,0,t_z) − G_cam||²`` on
    per-pixel point-map values. **1-view only**; on multi-view inputs the track
    sets a ``multi_view_not_supported`` reason and emits NaN sentinels rather than
    raising. The view-aware ``auto`` policy only selects it for ``view_count == 1``.

    ``all`` is left unchanged (no scale_shift_cam) so existing results
    reproduce; use ``all_cam`` to opt into the camera-frame track.
    """
    is_single = (view_count == 1)
    if alignment_arg == "auto":
        if camera_mode == "gt":
            if is_single:
                return True, True, False, True, "scale"
            return True, True, False, False, "scale"
        # camera_mode == "pred"
        if is_single:
            return True, True, False, True, "scale"
        return True, False, True, False, "sim3"
    if alignment_arg == "none":
        return True, False, False, False, "raw"
    if alignment_arg == "scale":
        return True, True, False, False, "scale"
    if alignment_arg == "sim3":
        return True, False, True, False, "sim3"
    if alignment_arg in ("scale_shift", "scale_shift_cam"):
        return True, False, False, True, "scale_shift_cam"
    if alignment_arg == "all":
        return True, True, True, False, "scale" if camera_mode == "gt" else "sim3"
    if alignment_arg == "all_cam":
        return True, True, False, True, "scale_shift_cam"
    raise ValueError(f"unknown --alignment value: {alignment_arg!r}")


# ---------------------------------------------------------------------------
# 3D metrics for one scene, raw + scale-only + Sim(3) tracks
# ---------------------------------------------------------------------------

def _stack_per_frame(arr_or_list, S: int) -> np.ndarray:
    """Return (S, ...) numpy array from a list-of-arrays or stacked array."""
    if arr_or_list is None:
        return None
    if isinstance(arr_or_list, list):
        return np.stack([np.asarray(arr_or_list[s]) for s in range(S)], axis=0)
    a = np.asarray(arr_or_list)
    return a


def _resolve_pred_layout_depth(preds_one: dict, use_depth_as_layout: bool) -> np.ndarray | None:
    """Return per-frame predicted layout depth as (S, H, W) numpy, or None."""
    if "layout_depth" in preds_one:
        ld = _to_np(preds_one["layout_depth"])
    elif use_depth_as_layout and "depth" in preds_one:
        ld = _to_np(preds_one["depth"])
    else:
        return None
    if ld.ndim == 4 and ld.shape[-1] == 1:
        ld = ld[..., 0]
    return ld


def _build_scaled_pred_cloud(preds_one: dict,
                             sample_for_cams: dict,
                             scale: float,
                             use_depth_as_layout: bool,
                             extrinsics_convention: str) -> np.ndarray:
    """Rebuild the predicted cloud after multiplying pred layout depth by ``scale``.

    Camera intrinsics/extrinsics are taken from ``sample_for_cams``; this lets
    callers pass GT cameras (for scale-only alignment) or aligned predicted
    cameras (for Sim(3) alignment). The cloud is gated by depth validity only
    (never by a layout mask).
    """
    if not np.isfinite(scale):
        return np.zeros((0, 3), dtype=np.float32)
    scaled_preds = dict(preds_one)
    if "layout_depth" in preds_one:
        ld = _to_np(preds_one["layout_depth"]) * float(scale)
        scaled_preds["layout_depth"] = ld
    if use_depth_as_layout and "depth" in preds_one:
        d = _to_np(preds_one["depth"]) * float(scale)
        scaled_preds["depth"] = d
    return build_scene_pointcloud_from_batch(
        sample_for_cams, scaled_preds, mode="pred",
        use_depth_as_layout=use_depth_as_layout,
        extrinsics_convention=extrinsics_convention,
    )


def _build_corresponded_points(pred_ld: np.ndarray,
                               gt_ld: np.ndarray,
                               sample_for_pred: dict,
                               sample_gt: dict,
                               overlap_mask: np.ndarray | None,
                               S: int,
                               extrinsics_convention: str,
                               max_pairs: int = 0,
                               seed: int = 0xE3,
                               frame: str = "world") -> tuple[np.ndarray, np.ndarray]:
    """Build pixel-corresponded predicted/GT 3-D points for the scale_shift fit.

    For each frame we backproject the predicted and GT layout depth through the
    same per-frame cameras using an *identical* validity mask
    ``combined = overlap & (pred>eps) & (gt>eps)``. Because the backproject
    helpers keep ``(depth>1e-6) & mask`` pixels in row-major order and
    ``combined`` is a subset of both depth-validity masks, the two returned
    arrays are the same length and corresponded element-wise (``P_i`` and
    ``G_i`` are the same pixel).

    Cameras: ``sample_for_pred`` for the predicted side (so the points match the
    raw predicted cloud) and ``sample_gt`` for the GT side. Under
    ``--camera-mode gt`` these are identical.

    ``frame``:
        - ``"world"``: points returned in the world frame via per-view
          extrinsics. Note that this introduces a ``(s-1)·C`` camera-centre
          residual in any scale_shift fit, which is why ``scale_shift_cam``
          uses the camera frame instead.
        - ``"camera"`` (scale_shift_cam): points returned in each view's
          *camera-local* frame, i.e. ``P_cam = K^{-1} u · z``. This is the
          LaRI / Room-Envelopes point-map space; the camera centre is at the
          origin, so the world-frame bias disappears. **Single-view only**:
          for multi-view, the per-view camera frames differ and a single
          ``(s, t_z)`` no longer corresponds to a consistent world-frame
          transform, callers must restrict to ``S == 1`` when using this.

    Returns ``(P, G)`` each ``(M, 3)`` float32 (possibly empty). When
    ``max_pairs > 0`` and there are more than ``max_pairs`` correspondences,
    a deterministic random subset is returned (same indices for P and G).
    """
    if frame not in ("world", "camera"):
        raise ValueError(f"frame must be 'world' or 'camera'; got {frame!r}")

    intr_pred = _stack_per_frame(sample_for_pred.get("intrinsics"), S)
    extr_pred = _stack_per_frame(sample_for_pred.get("extrinsics"), S)
    intr_gt   = _stack_per_frame(sample_gt.get("intrinsics"), S)
    extr_gt   = _stack_per_frame(sample_gt.get("extrinsics"), S)

    P_chunks: list[np.ndarray] = []
    G_chunks: list[np.ndarray] = []
    for s in range(S):
        p_d = np.asarray(pred_ld[s], dtype=np.float32)
        g_d = np.asarray(gt_ld[s], dtype=np.float32)
        # Reject NaN AND +inf (exp activation can overflow); ``> 1e-6``
        # already rejects NaN but not +inf, so ``np.isfinite`` is required.
        combined = (np.isfinite(p_d) & (p_d > 1e-6)
                    & np.isfinite(g_d) & (g_d > 1e-6))
        if overlap_mask is not None:
            combined = combined & np.asarray(overlap_mask[s], dtype=bool)
        if not combined.any():
            continue
        if frame == "world":
            P_s = backproject_depth_to_world(
                p_d, intr_pred[s], extr_pred[s], mask=combined,
                extrinsics_convention=extrinsics_convention,
            )
            G_s = backproject_depth_to_world(
                g_d, intr_gt[s], extr_gt[s], mask=combined,
                extrinsics_convention=extrinsics_convention,
            )
        else:  # frame == "camera"
            # Camera-frame point map: K^{-1} u · z, no extrinsics applied.
            # Both pred and GT use the SAME camera intrinsics (under
            # --camera-mode gt the pred camera == GT camera; under
            # --camera-mode pred the predicted K is what the pred cloud lives
            # in, so it matches what scale_shift_cam will align). Validity
            # mask is identical to the world-frame branch.
            from training.geometry.room_envelope_geometry import (
                unproject_depth_to_camera_points,
            )
            P_full = unproject_depth_to_camera_points(p_d, intr_pred[s])
            G_full = unproject_depth_to_camera_points(g_d, intr_gt[s])
            mask_flat = combined.reshape(-1)
            P_s = P_full.reshape(-1, 3)[mask_flat]
            G_s = G_full.reshape(-1, 3)[mask_flat]
        if len(P_s) and len(P_s) == len(G_s):
            P_chunks.append(P_s)
            G_chunks.append(G_s)

    if not P_chunks:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty
    P = np.concatenate(P_chunks, axis=0)
    G = np.concatenate(G_chunks, axis=0)
    if max_pairs and len(P) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(P), max_pairs, replace=False)
        P, G = P[idx], G[idx]
    return P.astype(np.float32), G.astype(np.float32)


def _backproject_labeled(depth_S: np.ndarray,
                         intr_S: np.ndarray,
                         extr_S: np.ndarray,
                         keep_S: np.ndarray | None,
                         label_S: np.ndarray,
                         S: int,
                         extrinsics_convention: str) -> tuple[np.ndarray, np.ndarray]:
    """Backproject per-frame depth to world points, carrying a per-point label.

    Each kept pixel's world point is paired with that pixel's value from
    ``label_S`` (an int array, e.g. 1=seen / 0=unseen / -1=not-layout). The
    final per-frame mask is ``(depth>1e-6) & keep``, the same expression
    :func:`backproject_depth_to_world` applies internally, so the returned
    points and labels are aligned element-wise in row-major order.

    Args:
        depth_S:  (S,H,W) depth maps.
        intr_S:   (S,3,3) intrinsics.
        extr_S:   (S,3,4)/(S,4,4) extrinsics.
        keep_S:   (S,H,W) bool keep-mask, or ``None`` (depth-validity only).
        label_S:  (S,H,W) integer label per pixel.
        S:        frame count.

    Returns:
        ``(pts (N,3) float32, labels (N,) int16)``.
    """
    pts_chunks: list[np.ndarray] = []
    lab_chunks: list[np.ndarray] = []
    for s in range(S):
        d = np.asarray(depth_S[s], dtype=np.float32)
        # Match backproject_depth_to_world's filter so labels stay aligned
        # with returned points (NaN AND +inf rejected together).
        final = np.isfinite(d) & (d > 1e-6)
        if keep_S is not None:
            ks = np.asarray(keep_S[s], dtype=bool)
            if ks.shape == final.shape:
                final = final & ks
        if not final.any():
            continue
        pts = backproject_depth_to_world(
            d, intr_S[s], extr_S[s], mask=final,
            extrinsics_convention=extrinsics_convention,
        )
        lab = np.asarray(label_S[s]).reshape(-1)[final.reshape(-1)]
        if len(pts) and len(pts) == len(lab):
            pts_chunks.append(pts)
            lab_chunks.append(lab)
    if not pts_chunks:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int16))
    return (np.concatenate(pts_chunks, axis=0).astype(np.float32),
            np.concatenate(lab_chunks, axis=0).astype(np.int16))


def _bbox_size_size_arr(bbox_dict: dict) -> np.ndarray:
    """Return bbox size as a 3-vector numpy array, NaN-safe."""
    return np.asarray(bbox_dict.get("bbox_size", [np.nan, np.nan, np.nan]),
                      dtype=np.float64)


def _safe_size_ratio(p_size: np.ndarray, g_size: np.ndarray) -> list:
    """Per-axis ratio p/g; NaN-safe and handles tiny GT sizes."""
    g = np.where(np.abs(g_size) > 1e-9, g_size, np.nan)
    return (p_size / g).tolist()


def _compute_3d_metrics_for_scene(
    *,
    sample: dict,
    preds_one: dict,
    args,
    image_hw: tuple[int, int],
    do_raw: bool,
    do_scale: bool,
    do_sim3: bool,
    do_scale_shift_cam: bool = False,
    timer: StageTimer | None = None,
    vggt_scene_scale: float | None = None,
    gt_cache: dict | None = None,
) -> tuple[dict, dict]:
    """Compute raw / scale-aligned / Sim(3) / scale_shift_cam-aligned 3D metrics.

    Returns ``(record, plys)`` where ``plys`` maps PLY filename suffixes
    ("gt", "pred_raw", "pred_scale_aligned", "pred_sim3_aligned",
    "pred_scale_shift_cam_aligned") to point arrays for the caller to
    optionally save.

    ``vggt_scene_scale`` (optional), when supplied (only meaningful for the
    vggt_scene eval pass), the chamfer/F-score calls also report
    ``fscore_physical_{0.05,0.10,0.20}m`` keys representing the F-score in
    a normalized space at thresholds that correspond to 5/10/20 cm of
    *physical* distance. The default ``fscore_{0.05,0.10,0.20}`` keys for
    the vggt_scene pass remain in normalized-space units (i.e. 0.20 in that
    space ≈ ``0.20 * vggt_scene_scale`` metres physical, typically ~1-2 m).
    """
    record: dict = {}
    plys: dict = {}

    # Physical-equivalent F-score thresholds for the vggt_scene eval pass.
    # Distances in vggt_scene space equal physical distances divided by
    # ``vggt_scene_scale`` (the per-sample mean valid-point distance used by
    # the trainer's ``_process_batch`` normalization). To recover a physical
    # threshold ``T_m`` we therefore use ``T_m / vggt_scene_scale`` as the
    # threshold in normalized space. Passed to every ``chamfer_and_fscore``
    # call in this function so headline / scale_aligned / scale_shift /
    # sim3 all get the extra ``fscore_physical_*`` keys.
    physical_thresholds = None
    if vggt_scene_scale is not None and np.isfinite(vggt_scene_scale) and vggt_scene_scale > 0:
        physical_thresholds = {
            "0.05m": 0.05 / float(vggt_scene_scale),
            "0.10m": 0.10 / float(vggt_scene_scale),
            "0.20m": 0.20 / float(vggt_scene_scale),
        }
        record["vggt_scene_physical_threshold_0.05m_norm"] = physical_thresholds["0.05m"]
        record["vggt_scene_physical_threshold_0.10m_norm"] = physical_thresholds["0.10m"]
        record["vggt_scene_physical_threshold_0.20m_norm"] = physical_thresholds["0.20m"]

    # No-op fallback so unconditional ``with timer.time(...)`` blocks below
    # remain valid when the caller doesn't pass a timer.
    _timer = timer if timer is not None else StageTimer(enabled=False)

    # cKDTree.query workers; -1 = all cores. getattr keeps backward-compat
    # for callers that haven't been updated to forward the flag.
    _kdtw = getattr(args, "kdtree_workers", -1)

    # GT-side cache (per scene, per eval-space). The GT cloud / KD-tree / bbox
    # and the seen/unseen GT split clouds+trees depend ONLY on ``sample`` (GT),
    # not on the prediction or the postprocess method, so the orchestrator can
    # hand the same dict to raw/ransac/manhattan/cuboid for one (scene, space)
    # and we build them once. ``id(sample)`` keys the metric (``sample``) vs
    # vggt_scene (``normalized_sample``) entries, they are distinct objects,
    # and the orchestrator resets the dict each scene so ids never collide.
    # Pred-side clouds/trees are NEVER cached (each method differs).
    _gc = gt_cache.setdefault(id(sample), {}) if gt_cache is not None else None

    # ---- Per-frame inputs ----
    extr_gt_full = _to_np(sample.get("extrinsics"))
    intr_gt_full = _to_np(sample.get("intrinsics"))
    if extr_gt_full is None or intr_gt_full is None:
        raise KeyError("sample missing 'extrinsics'/'intrinsics'")
    S = len(extr_gt_full) if isinstance(sample.get("extrinsics"), list) else int(extr_gt_full.shape[0])

    pred_ld = _resolve_pred_layout_depth(preds_one, args.use_depth_as_layout)
    if pred_ld is None:
        raise KeyError("predictions has no usable layout_depth/depth")

    # ---- (Phase 2A) Materialize prediction tensors to numpy ONCE.
    # Downstream point-cloud builders re-run ``.detach().cpu().numpy()`` on
    # every call; pre-converting turns those into no-ops (``_to_np`` and
    # ``np.asarray`` short-circuit on numpy inputs) without changing any
    # metric values. ``preds_np`` is then passed everywhere in place of
    # ``preds_one``.
    preds_np = dict(preds_one)
    for _k in ("layout_depth", "depth", "layout_mask_logits"):
        if _k in preds_one:
            preds_np[_k] = _to_np(preds_one[_k])

    gt_ld = _stack_per_frame(sample.get("layout_depths"), S)
    gt_dm = _stack_per_frame(sample.get("layout_depth_masks"), S)

    # The predicted cloud is never mask-filtered (it covers the full amodal
    # envelope); the depth-scale overlap is the GT depth-validity region.
    pred_keep = None
    overlap_mask = gt_dm.astype(bool) if gt_dm is not None else None

    # Per-pixel GT-layout-depth validity (gt_ld > 1e-6 AND layout_depth_masks).
    # This is exactly ``label_S != -1`` from the seen/unseen block below; compute
    # it once here so the OVERALL pred→GT metrics (accuracy / precision) can drop
    # predicted points whose source pixel has undefined GT layout depth,
    # consistent with the 2D metrics and the seen/unseen splits, which already
    # ignore those pixels. Built per (scene, eval-space); cheap boolean ops.
    if gt_ld is not None:
        gt_dm_bool = overlap_mask if overlap_mask is not None else (gt_ld > 1e-6)
        gt_valid_S = (gt_ld > 1e-6) & gt_dm_bool
    else:
        gt_dm_bool = None
        gt_valid_S = None

    # Depth-scale diagnostics over the same overlap (used both for the
    # scale-only alignment and for the diagnostic block).
    diag = summarize_depth_scale(pred_ld, gt_ld, overlap_mask)
    record["median_pred_depth"]    = diag["median_pred_depth"]
    record["median_gt_depth"]      = diag["median_gt_depth"]
    record["median_pred_gt_ratio"] = diag["median_pred_gt_ratio"]
    record["median_gt_pred_scale"] = diag["median_gt_pred_scale"]
    record["depth_overlap_pixels"] = int(diag.get("n_overlap", 0))

    # ---- Build GT cloud once (always in GT world frame) ----
    # Reuse across methods within a (scene, eval-space) via the GT cache.
    if _gc is not None and "gt_pc_s" in _gc:
        gt_pc_s = _gc["gt_pc_s"]
        gt_tree = _gc["gt_tree"]
        gt_bb = _gc["gt_bb"]
    else:
        with _timer.time("gt_cloud_build"):
            gt_pc = build_scene_pointcloud_from_batch(
                sample, preds_np, mode="gt",
                extrinsics_convention=args.extrinsics_convention,
            )
            gt_pc_s = sample_pointcloud(gt_pc, args.max_points_per_scene, seed=0xBAD)
        # (Phase 2B) Build GT KD-tree ONCE per scene; reused across
        # raw / scale / sim3 / scale_shift chamfer calls (and the seen/unseen
        # split block builds its own per-(split) trees lazily, also cached).
        gt_tree = build_kdtree(gt_pc_s)
        gt_bb = bbox_size(gt_pc_s)
        if _gc is not None:
            _gc["gt_pc_s"] = gt_pc_s
            _gc["gt_tree"] = gt_tree
            _gc["gt_bb"] = gt_bb
    record["gt_bbox_min"]  = gt_bb["bbox_min"]
    record["gt_bbox_max"]  = gt_bb["bbox_max"]
    record["gt_bbox_size"] = gt_bb["bbox_size"]
    record["n_gt_points"]  = int(len(gt_pc_s))
    plys["gt"] = gt_pc_s
    g_size = _bbox_size_size_arr(gt_bb)

    # ---- Decode predicted cameras when a camera/pose head is present ----
    # The camera head (``pose_enc``) serves two independent purposes here:
    #   (a) head-aware pose/intrinsics error metrics, emitted whenever the
    #       head exists, *regardless* of --camera-mode, so a checkpoint with a
    #       camera head reports pose error even on the default gt path; and
    #   (b) driving 3D reconstruction from predicted cameras, only when
    #       --camera-mode pred is requested.
    sample_for_pred = sample
    extr_pred_decoded = None
    intr_pred_decoded = None
    has_pose_enc = "pose_enc" in preds_one
    if has_pose_enc:
        H_img, W_img = int(image_hw[0]), int(image_hw[1])
        ep, ip = decode_pred_pose_enc(preds_one["pose_enc"], (H_img, W_img))
        extr_pred_decoded = ep[0] if ep.ndim == 4 else ep
        intr_pred_decoded = ip[0] if ip.ndim == 4 else ip

        # (a) Pose / intrinsics error metrics, head-aware, camera-mode-agnostic.
        extr_gt_for_pose = _stack_per_frame(sample.get("extrinsics"), S)
        if extr_gt_for_pose is not None:
            if extr_gt_for_pose.shape[-2:] == (4, 4):
                extr_gt_for_pose = extr_gt_for_pose[..., :3, :]
            intr_gt_for_pose = _stack_per_frame(sample.get("intrinsics"), S)
            try:
                pose_err = compute_pose_error_metrics(
                    extr_pred_decoded, extr_gt_for_pose,
                    intr_pred=intr_pred_decoded, intr_gt=intr_gt_for_pose,
                    image_hw=(int(image_hw[0]), int(image_hw[1])),
                )
                record.update(prefix_metrics(pose_err, "pose_"))
            except Exception as e:  # pragma: no cover, defensive
                record["pose_error_msg"] = repr(e)

    # (b) Predicted-camera reconstruction only under --camera-mode pred.
    if args.camera_mode == "pred":
        if not has_pose_enc:
            raise KeyError("camera_mode=pred but 'pose_enc' not in predictions")
        sample_for_pred = dict(sample)
        sample_for_pred["extrinsics"] = [extr_pred_decoded[s] for s in range(S)]
        sample_for_pred["intrinsics"] = [intr_pred_decoded[s] for s in range(S)]

    # ---- RAW pred cloud (no alignment) ----
    with _timer.time("pred_raw_build"):
        pred_raw = build_scene_pointcloud_from_batch(
            sample_for_pred, preds_np, mode="pred",
            use_depth_as_layout=args.use_depth_as_layout,
            extrinsics_convention=args.extrinsics_convention,
        )
        # GT-validity mask aligned row-for-row with pred_raw (identical
        # depth-validity gate); co-sample with the same RNG so pred_raw_s is
        # byte-identical to the historical sample.
        pred_raw_gtvalid = pred_cloud_gtvalid_mask(pred_ld, gt_valid_S, pred_keep)
        if pred_raw_gtvalid is not None:
            pred_raw_s, pred_raw_gtvalid_s = sample_pointcloud_with_companion(
                pred_raw, pred_raw_gtvalid, args.max_points_per_scene, seed=0xC1)
        else:
            pred_raw_s = sample_pointcloud(pred_raw, args.max_points_per_scene, seed=0xC1)
            pred_raw_gtvalid_s = None
    raw_bb = bbox_size(pred_raw_s)
    record["pred_raw_bbox_min"]  = raw_bb["bbox_min"]
    record["pred_raw_bbox_max"]  = raw_bb["bbox_max"]
    record["pred_raw_bbox_size"] = raw_bb["bbox_size"]
    record["bbox_size_ratio_raw"] = _safe_size_ratio(_bbox_size_size_arr(raw_bb), g_size)
    record["n_pred_points"] = int(len(pred_raw_s))
    plys["pred_raw"] = pred_raw_s
    if do_raw:
        with _timer.time("chamfer_raw"):
            cf_raw = chamfer_and_fscore(
                pred_raw_s, gt_pc_s, gt_tree=gt_tree, workers=_kdtw,
                physical_thresholds=physical_thresholds,
                pred_acc_mask=pred_raw_gtvalid_s,
            )
        record.update(prefix_metrics(cf_raw, "raw_"))

    # ---- SCALE-ONLY aligned pred cloud (uses same cameras as raw) ----
    if do_scale:
        scale_factor = float("nan")
        pred_scale_gtvalid_s = None
        if args.scale_alignment == "median_depth":
            scale_factor = compute_median_depth_scale(pred_ld, gt_ld, overlap_mask)
            scale_source = "median_depth_per_pixel"
        else:
            scale_source = "pointcloud_rms"

        if args.scale_alignment == "median_depth" and np.isfinite(scale_factor):
            with _timer.time("scale_cloud_build"):
                pred_scale = _build_scaled_pred_cloud(
                    preds_one=preds_np,
                    sample_for_cams=sample_for_pred,
                    scale=scale_factor,
                    use_depth_as_layout=args.use_depth_as_layout,
                    extrinsics_convention=args.extrinsics_convention,
                )
                # Mask gate uses the SAME scaled depth as the cloud build.
                _scale_gtvalid = pred_cloud_gtvalid_mask(
                    pred_ld * float(scale_factor), gt_valid_S, pred_keep)
                if _scale_gtvalid is not None:
                    pred_scale_s, pred_scale_gtvalid_s = sample_pointcloud_with_companion(
                        pred_scale, _scale_gtvalid, args.max_points_per_scene, seed=0xC2)
                else:
                    pred_scale_s = sample_pointcloud(pred_scale, args.max_points_per_scene, seed=0xC2)
        else:
            # Fallback: scale via pointcloud RMS over the fused raw cloud.
            with _timer.time("scale_cloud_build"):
                pred_scale_s, scale_factor = align_pointcloud_scale_only(
                    pred_raw_s, gt_pc_s, mode="pointcloud_rms",
                )
                pred_scale_s = sample_pointcloud(pred_scale_s, args.max_points_per_scene, seed=0xC3)
            scale_source = "pointcloud_rms"
            # RMS alignment is a per-point affine transform of pred_raw_s (order
            # and count preserved; the trailing sample is a no-op as
            # len(pred_raw_s) <= max_points), so the raw mask applies directly.
            pred_scale_gtvalid_s = pred_raw_gtvalid_s

        scale_bb = bbox_size(pred_scale_s)
        record["pred_scale_aligned_bbox_min"]  = scale_bb["bbox_min"]
        record["pred_scale_aligned_bbox_max"]  = scale_bb["bbox_max"]
        record["pred_scale_aligned_bbox_size"] = scale_bb["bbox_size"]
        record["bbox_size_ratio_scale_aligned"] = _safe_size_ratio(
            _bbox_size_size_arr(scale_bb), g_size,
        )
        record["scale_alignment_factor"]   = float(scale_factor) if np.isfinite(scale_factor) else float("nan")
        record["scale_alignment_source"]   = scale_source
        plys["pred_scale_aligned"] = pred_scale_s

        with _timer.time("chamfer_scale"):
            cf_scale = chamfer_and_fscore(
                pred_scale_s, gt_pc_s, gt_tree=gt_tree, workers=_kdtw,
                physical_thresholds=physical_thresholds,
                pred_acc_mask=pred_scale_gtvalid_s,
            )
        record.update(prefix_metrics(cf_scale, "scale_aligned_"))

    # ---- SIM(3) aligned pred cloud ----
    if do_sim3:
        sim3_record: dict = {}
        sim3_source = None
        pred_sim3_gtvalid_s = None
        # Prefer camera-centre Umeyama when we already decoded pred cameras.
        if extr_pred_decoded is not None:
            extr_gt_3x4 = _stack_per_frame(sample.get("extrinsics"), S)
            if extr_gt_3x4.shape[-2:] == (4, 4):
                extr_gt_3x4 = extr_gt_3x4[..., :3, :]
            try:
                with _timer.time("sim3_fit"):
                    extr_aligned, sim3_scale = align_pred_cameras_to_gt(
                        extr_pred_decoded, extr_gt_3x4, with_scale=True,
                    )
                # Build aligned pred cloud: pred depth × sim3_scale, with
                # aligned extrinsics + predicted intrinsics.
                aligned_sample = dict(sample)
                aligned_sample["extrinsics"] = [extr_aligned[s] for s in range(S)]
                aligned_sample["intrinsics"] = [intr_pred_decoded[s] for s in range(S)]
                with _timer.time("sim3_cloud_build"):
                    pred_sim3 = _build_scaled_pred_cloud(
                        preds_one=preds_np,
                        sample_for_cams=aligned_sample,
                        scale=sim3_scale,
                        use_depth_as_layout=args.use_depth_as_layout,
                        extrinsics_convention=args.extrinsics_convention,
                    )
                    # Aligned cameras only rotate/translate points; the
                    # depth-validity gate (hence the mask) uses pred depth ×
                    # sim3_scale, matching the cloud build.
                    _sim3_gtvalid = pred_cloud_gtvalid_mask(
                        pred_ld * float(sim3_scale), gt_valid_S, pred_keep)
                    if _sim3_gtvalid is not None:
                        pred_sim3_s, pred_sim3_gtvalid_s = sample_pointcloud_with_companion(
                            pred_sim3, _sim3_gtvalid, args.max_points_per_scene, seed=0xD1)
                    else:
                        pred_sim3_s = sample_pointcloud(pred_sim3, args.max_points_per_scene, seed=0xD1)
                sim3_record["sim3_scale"] = float(sim3_scale)
                # Camera-centre Umeyama returns rigid R, t, det(R)=±1, t in
                # world units. Recover them for diagnostics.
                # The function returned ``aligned`` extrinsics + scale; we
                # don't get R/t back directly. Recompute via
                # ``umeyama_similarity`` on the camera centres for logging.
                from alignment import umeyama_similarity

                def _cam_center(extr_3x4):
                    R = np.asarray(extr_3x4[:3, :3], dtype=np.float64)
                    t = np.asarray(extr_3x4[:3, 3], dtype=np.float64)
                    return -R.T @ t

                pred_C = np.stack([_cam_center(extr_pred_decoded[s]) for s in range(S)], axis=0)
                gt_C = np.stack([_cam_center(extr_gt_3x4[s]) for s in range(S)], axis=0)
                _, R_um, t_um = umeyama_similarity(pred_C, gt_C, with_scale=True)
                sim3_record["sim3_rotation_det"]   = float(np.linalg.det(R_um))
                sim3_record["sim3_translation_norm"] = float(np.linalg.norm(t_um))
                sim3_source = "camera_centres"
            except Exception as e:  # pragma: no cover, defensive
                sim3_record["sim3_error"] = f"camera_centre_umeyama: {e}"
                pred_sim3_s = np.zeros((0, 3), dtype=np.float32)
                sim3_source = None

        if sim3_source is None:
            # No camera correspondences are available (typically because
            # ``--camera_mode gt`` was selected, so we never decoded a
            # pred pose, and there is no usable second camera frame to
            # Umeyama-align onto GT cameras). The previous fallback ran
            # Sim(3) on ``pred_raw_s[:n]`` against ``gt_pc_s[:n]``, two
            # uncorresponded arbitrarily-ordered fused clouds, which
            # produced a geometrically meaningless Sim(3). Per the audit
            # (E-4), make this loud: skip the sim3 track, emit a clear
            # error key, and leave the aligned cloud empty so chamfer
            # downstream returns NaN/0 (the standard empty-cloud signal).
            print(
                f"[sim3] skipped: no camera correspondences available "
                f"(camera_mode={args.camera_mode}). Use --camera_mode pred "
                f"to get a sim3 track, or drop --alignment sim3/all."
            )
            pred_sim3_s = np.zeros((0, 3), dtype=np.float32)
            sim3_record["sim3_error"] = (
                "skipped_no_camera_correspondences: refuses to align "
                "two uncorresponded fused clouds via prefix p[:n]/g[:n]"
            )
            sim3_source = "skipped_no_correspondence"

        sim3_bb = bbox_size(pred_sim3_s)
        sim3_record["pred_sim3_aligned_bbox_min"]  = sim3_bb["bbox_min"]
        sim3_record["pred_sim3_aligned_bbox_max"]  = sim3_bb["bbox_max"]
        sim3_record["pred_sim3_aligned_bbox_size"] = sim3_bb["bbox_size"]
        sim3_record["bbox_size_ratio_sim3_aligned"] = _safe_size_ratio(
            _bbox_size_size_arr(sim3_bb), g_size,
        )
        sim3_record["sim3_alignment_source"] = sim3_source
        record.update(sim3_record)
        plys["pred_sim3_aligned"] = pred_sim3_s

        with _timer.time("chamfer_sim3"):
            cf_sim3 = chamfer_and_fscore(
                pred_sim3_s, gt_pc_s, gt_tree=gt_tree, workers=_kdtw,
                physical_thresholds=physical_thresholds,
                pred_acc_mask=pred_sim3_gtvalid_s,
            )
        record.update(prefix_metrics(cf_sim3, "sim3_aligned_"))

    # ---- SCALE+SHIFT (camera-frame; LaRI / Room Envelopes correct variant) ----
    # A naive world-frame fit ``min ||s·P_world + (0,0,t_z) − G_world||²`` is
    # biased: substituting ``P_world = R·r·z_pred + C`` shows the residual
    # contains an irreducible ``(s-1)·C`` term (camera centre), which pulls the
    # LSQ toward s=1. The LaRI / Room-Envelopes alignment instead operates on
    # the **camera-local point map** ``P_cam = K^{-1}·u·z``, where the camera
    # centre is at the origin and the bias disappears. We then re-apply
    # ``(s, t_z)`` to the world cloud by scaling pred depth (preserves the pixel
    # ray) and translating the resulting world points by ``R_c2w · (0, 0, t_z)``
    #, the world-frame equivalent of a camera-local optical-axis shift.
    #
    # **1-view only.** A single ``(s, t_z)`` does not produce a consistent
    # multi-view world-frame transform (each view has its own R_c2w), so we
    # explicitly skip and emit NaN sentinels for ``S > 1`` rather than
    # silently producing geometrically meaningless results.
    s_cam = float("nan")
    tz_cam = float("nan")
    ok_cam = False
    reason_cam = "skipped"
    n_pairs_cam = 0
    cam_residual_rmse = float("nan")
    cam_residual_rel  = float("nan")
    cam_cond = float("nan")
    pred_cam_aligned_full = np.zeros((0, 3), dtype=np.float32)
    pred_cam_gtvalid_s = None
    if do_scale_shift_cam:
        if S != 1:
            ok_cam = False
            reason_cam = "multi_view_not_supported"
            pred_cam_s = np.zeros((0, 3), dtype=np.float32)
        else:
            with _timer.time("scale_shift_cam_pairs"):
                P_cam_corr, G_cam_corr = _build_corresponded_points(
                    pred_ld=pred_ld,
                    gt_ld=gt_ld,
                    sample_for_pred=sample_for_pred,
                    sample_gt=sample,
                    overlap_mask=overlap_mask,
                    S=S,
                    extrinsics_convention=args.extrinsics_convention,
                    max_pairs=args.max_points_per_scene,
                    seed=0xF7,
                    frame="camera",
                )
            with _timer.time("scale_shift_cam_fit"):
                s_cam, tz_cam, ok_cam, reason_cam, n_pairs_cam = fit_xyz_scale_zshift_lstsq(
                    P_cam_corr, G_cam_corr,
                )

            # Stability diagnostics: residual + 2x2 normal-equation condition.
            if n_pairs_cam > 0:
                P64 = np.asarray(P_cam_corr, dtype=np.float64)
                G64 = np.asarray(G_cam_corr, dtype=np.float64)
                if np.isfinite(s_cam) and np.isfinite(tz_cam):
                    resid = s_cam * P64 + np.array([0.0, 0.0, tz_cam]) - G64
                    cam_residual_rmse = float(np.sqrt((resid * resid).sum() / len(P64)))
                    g_rms = float(np.sqrt((G64 * G64).sum() / len(G64)))
                    cam_residual_rel = cam_residual_rmse / g_rms if g_rms > 0 else float("nan")
                Q = float((P64 * P64).sum())
                Pz = float(P64[:, 2].sum())
                N = float(len(P64))
                tr = Q + N
                det = Q * N - Pz * Pz
                disc = max(0.0, tr * tr - 4.0 * det)
                l1 = 0.5 * (tr + np.sqrt(disc))
                l2 = 0.5 * (tr - np.sqrt(disc))
                if l2 > 1e-30 and np.isfinite(l1):
                    cam_cond = float(l1 / l2)

            if ok_cam:
                with _timer.time("scale_shift_cam_cloud_build"):
                    pred_cam_scaled = _build_scaled_pred_cloud(
                        preds_one=preds_np,
                        sample_for_cams=sample_for_pred,
                        scale=s_cam,
                        use_depth_as_layout=args.use_depth_as_layout,
                        extrinsics_convention=args.extrinsics_convention,
                    )
                # World-frame equivalent of a camera-local (0, 0, t_z) shift
                # for this single view. Both extrinsics conventions reduce
                # to ``R_c2w · (0, 0, t_z)`` once we close-form-invert the
                # w2c extrinsic into a c2w rotation.
                extr_pred_S = _stack_per_frame(sample_for_pred.get("extrinsics"), S)
                ext_3x4 = np.asarray(extr_pred_S[0], dtype=np.float32)
                if ext_3x4.shape == (4, 4):
                    ext_3x4 = ext_3x4[:3, :]
                if args.extrinsics_convention == "w2c":
                    from vggt.utils.geometry import closed_form_inverse_se3
                    ext_4x4 = np.eye(4, dtype=np.float32)
                    ext_4x4[:3, :] = ext_3x4
                    c2w = closed_form_inverse_se3(ext_4x4[None])[0]
                    R_c2w = c2w[:3, :3]
                else:  # c2w
                    R_c2w = ext_3x4[:3, :3]
                world_shift = (R_c2w @ np.array([0.0, 0.0, float(tz_cam)],
                                                dtype=np.float32))
                if len(pred_cam_scaled):
                    pred_cam_aligned_full = (
                        pred_cam_scaled.astype(np.float64) + world_shift
                    ).astype(np.float32)
                # The world_shift is a constant translation (order/count
                # preserved), so the mask gate uses pred depth × s_cam, the
                # same depth the scaled cloud was built from.
                _cam_gtvalid = pred_cloud_gtvalid_mask(
                    pred_ld * float(s_cam), gt_valid_S, pred_keep)
                if (_cam_gtvalid is not None
                        and len(_cam_gtvalid) == len(pred_cam_aligned_full)):
                    pred_cam_s, pred_cam_gtvalid_s = sample_pointcloud_with_companion(
                        pred_cam_aligned_full, _cam_gtvalid,
                        args.max_points_per_scene, seed=0xF8,
                    )
                else:
                    pred_cam_s = sample_pointcloud(
                        pred_cam_aligned_full, args.max_points_per_scene, seed=0xF8,
                    )
                    pred_cam_gtvalid_s = None
                if len(pred_cam_s) == 0:
                    ok_cam = False
                    reason_cam = "empty_pred_cloud"
            if not ok_cam:
                pred_cam_s = np.zeros((0, 3), dtype=np.float32)
                pred_cam_aligned_full = np.zeros((0, 3), dtype=np.float32)
                pred_cam_gtvalid_s = None  # empty cloud → unmasked (inert; early-returns)

        cam_bb = bbox_size(pred_cam_s)
        record["pred_scale_shift_cam_aligned_bbox_min"]  = cam_bb["bbox_min"]
        record["pred_scale_shift_cam_aligned_bbox_max"]  = cam_bb["bbox_max"]
        record["pred_scale_shift_cam_aligned_bbox_size"] = cam_bb["bbox_size"]
        record["bbox_size_ratio_scale_shift_cam_aligned"] = _safe_size_ratio(
            _bbox_size_size_arr(cam_bb), g_size,
        )
        record["scale_shift_cam_s"]                = float(s_cam) if np.isfinite(s_cam) else float("nan")
        record["scale_shift_cam_tz"]               = float(tz_cam) if np.isfinite(tz_cam) else float("nan")
        record["scale_shift_cam_translation_norm"] = float(abs(tz_cam)) if np.isfinite(tz_cam) else float("nan")
        record["scale_shift_cam_num_pairs"]        = int(n_pairs_cam)
        record["scale_shift_cam_failed"]           = 0.0 if ok_cam else 1.0
        record["scale_shift_cam_failure_reason"]   = None if ok_cam else reason_cam
        record["scale_shift_cam_residual_rmse"]    = cam_residual_rmse
        record["scale_shift_cam_residual_rel"]     = cam_residual_rel
        record["scale_shift_cam_normal_cond"]      = cam_cond
        plys["pred_scale_shift_cam_aligned"]       = pred_cam_s

        with _timer.time("chamfer_scale_shift_cam"):
            cf_cam = chamfer_and_fscore(
                pred_cam_s, gt_pc_s, gt_tree=gt_tree, workers=_kdtw,
                physical_thresholds=physical_thresholds,
                pred_acc_mask=pred_cam_gtvalid_s,
            )
        record.update(prefix_metrics(cf_cam, "scale_shift_cam_aligned_"))

    # ---- SEEN / UNSEEN 3D splits (paper-style; layout_masks = visibility) ----
    # Reuses the exact 2D convention: a valid layout pixel is "seen" when
    # layout_masks > 0.5 (structural surface directly visible) and "unseen"
    # otherwise (structural surface behind clutter). Overall metrics above are
    # left untouched; this block only ADDS {track}_{seen,unseen}_* keys.
    lm_S = _stack_per_frame(sample.get("layout_masks"), S)
    _t_seen_unseen_0 = time.perf_counter() if (lm_S is not None and _timer.enabled) else None
    if lm_S is not None:
        # Indentation preserved; we time this block via perf_counter + record()
        # below to avoid re-indenting the long body.
        conv = args.extrinsics_convention
        intr_pred_S = _stack_per_frame(sample_for_pred.get("intrinsics"), S)
        extr_pred_S = _stack_per_frame(sample_for_pred.get("extrinsics"), S)
        MIN_SPLIT_PTS = 3

        # GT-side seen/unseen structures (the per-pixel seen/unseen label map,
        # the GT split clouds and their KD-trees) depend ONLY on GT (gt_ld,
        # gt_dm, layout_masks, GT cameras), never on the prediction or the
        # postprocess method. Build them once per (scene, eval-space) and cache;
        # pred-side labeling + chamfer below stay per-method. sample_pointcloud
        # uses a local seeded RNG, so lifting the GT split build above the
        # per-method pred work is value-preserving.
        if _gc is not None and "seen_unseen" in _gc:
            _su = _gc["seen_unseen"]
            label_S = _su["label_S"]
            gt_lab = _su["gt_lab"]
            gt_split_cache = _su["gt_split_cache"]
        else:
            # ``gt_dm_bool`` / ``gt_valid_S`` were computed once near the top of
            # this function (identical definition); reuse them here so the
            # overall-metric mask and the seen/unseen labels share one source of
            # truth. ``gt_valid_S`` is exactly ``label_S != -1``.
            # 1 = seen, 0 = unseen, -1 = not a (valid) layout pixel.
            label_S = np.where(gt_valid_S, np.where(lm_S > 0.5, 1, 0), -1).astype(np.int16)
            intr_gt_S = _stack_per_frame(sample.get("intrinsics"), S)
            extr_gt_S = _stack_per_frame(sample.get("extrinsics"), S)
            # GT labeled cloud (keep == layout-depth validity) partitions exactly
            # into seen ∪ unseen.
            gt_pts_lab, gt_lab = _backproject_labeled(
                gt_ld, intr_gt_S, extr_gt_S, gt_dm_bool, label_S, S, conv,
            )
            gt_split_cache: dict[int, tuple[np.ndarray, object]] = {}
            for _split_name, _val in (("seen", 1), ("unseen", 0)):
                _g = gt_pts_lab[gt_lab == _val]
                if len(_g) >= MIN_SPLIT_PTS:
                    _g_s = sample_pointcloud(_g, args.max_points_per_scene, seed=0xF2 + _val)
                    gt_split_cache[_val] = (_g_s, build_kdtree(_g_s))
                else:
                    gt_split_cache[_val] = (np.zeros((0, 3), dtype=np.float32), None)
            if _gc is not None:
                _gc["seen_unseen"] = {"label_S": label_S, "gt_lab": gt_lab,
                                      "gt_split_cache": gt_split_cache}
        # (Phase 2C) Reuse the full unsampled raw pred cloud built earlier
        # (``pred_raw`` from ``build_scene_pointcloud_from_batch``) instead of
        # re-backprojecting. The keep mask is bitwise-identical between the
        # two paths (depth > 1e-6 AND ``pred_keep``), so the labels extracted
        # by the same per-frame mask are point-aligned with ``pred_raw``.
        _raw_lab_chunks: list[np.ndarray] = []
        for _s_idx in range(S):
            _d_raw = np.asarray(pred_ld[_s_idx], dtype=np.float32)
            # Mirror the backproject_depth_to_world finite-and-positive gate
            # so the label row count matches ``pred_raw`` exactly.
            _final = np.isfinite(_d_raw) & (_d_raw > 1e-6)
            if pred_keep is not None:
                _ks = np.asarray(pred_keep[_s_idx], dtype=bool)
                if _ks.shape == _final.shape:
                    _final = _final & _ks
            if _final.any():
                _raw_lab_chunks.append(
                    label_S[_s_idx].reshape(-1)[_final.reshape(-1)]
                )
        pred_lab = (np.concatenate(_raw_lab_chunks).astype(np.int16)
                    if _raw_lab_chunks
                    else np.zeros((0,), dtype=np.int16))
        pred_raw_pts_lab = pred_raw  # built above by build_scene_pointcloud_from_batch
        if len(pred_raw_pts_lab) != len(pred_lab):
            # Defensive fallback if shapes drift unexpectedly.
            pred_raw_pts_lab, pred_lab = _backproject_labeled(
                pred_ld, intr_pred_S, extr_pred_S, pred_keep, label_S, S, conv,
            )

        record["n_points_gt_overall"]   = int(len(gt_lab))
        record["n_points_gt_seen"]      = int((gt_lab == 1).sum())
        record["n_points_gt_unseen"]    = int((gt_lab == 0).sum())
        record["n_points_pred_overall"] = int(len(pred_lab))
        record["n_points_pred_seen"]    = int((pred_lab == 1).sum())
        record["n_points_pred_unseen"]  = int((pred_lab == 0).sum())

        # Per-track aligned labeled pred clouds (labels follow the pred pixels;
        # alignment never drops or reorders pixels).
        track_clouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if do_raw:
            track_clouds["raw"] = (pred_raw_pts_lab, pred_lab)
        if do_scale and args.scale_alignment == "median_depth":
            sc = compute_median_depth_scale(pred_ld, gt_ld, overlap_mask)
            if np.isfinite(sc):
                ps_lab, pl = _backproject_labeled(
                    pred_ld * float(sc), intr_pred_S, extr_pred_S,
                    pred_keep, label_S, S, conv,
                )
                track_clouds["scale_aligned"] = (ps_lab, pl)
        if do_scale_shift_cam and ok_cam and S == 1:
            # 1-view scale_shift_cam: ``pred_cam_aligned_full`` is the FULL
            # world cloud after pred_depth × s + world_shift. The keep mask
            # is identical to scale_aligned (since we scaled the same pred
            # depth and applied a constant translation), so labels row-align
            # via the per-view ``label_S`` reduced by the same finite-and-
            # positive gate.
            lab_chunks_cam: list[np.ndarray] = []
            for _s_idx in range(S):
                _d_scaled = (pred_ld[_s_idx] * float(s_cam)).astype(np.float32)
                _final = np.isfinite(_d_scaled) & (_d_scaled > 1e-6)
                if pred_keep is not None:
                    _ks = np.asarray(pred_keep[_s_idx], dtype=bool)
                    if _ks.shape == _final.shape:
                        _final = _final & _ks
                if _final.any():
                    lab_chunks_cam.append(
                        label_S[_s_idx].reshape(-1)[_final.reshape(-1)]
                    )
            pl_cam = (np.concatenate(lab_chunks_cam).astype(np.int16)
                      if lab_chunks_cam
                      else np.zeros((0,), dtype=np.int16))
            ps_lab_cam = pred_cam_aligned_full
            if len(ps_lab_cam) != len(pl_cam):
                # Defensive fallback if shapes drift.
                ps_lab_cam, pl_cam = _backproject_labeled(
                    pred_ld * float(s_cam), intr_pred_S, extr_pred_S,
                    pred_keep, label_S, S, conv,
                )
                if len(ps_lab_cam):
                    # Reapply the world-frame translation (R_c2w · (0,0,tz)).
                    ext_3x4_fb = np.asarray(extr_pred_S[0], dtype=np.float32)
                    if ext_3x4_fb.shape == (4, 4):
                        ext_3x4_fb = ext_3x4_fb[:3, :]
                    if conv == "w2c":
                        from vggt.utils.geometry import closed_form_inverse_se3
                        ext_4x4_fb = np.eye(4, dtype=np.float32)
                        ext_4x4_fb[:3, :] = ext_3x4_fb
                        c2w_fb = closed_form_inverse_se3(ext_4x4_fb[None])[0]
                        R_c2w_fb = c2w_fb[:3, :3]
                    else:
                        R_c2w_fb = ext_3x4_fb[:3, :3]
                    ws_fb = R_c2w_fb @ np.array(
                        [0.0, 0.0, float(tz_cam)], dtype=np.float32,
                    )
                    ps_lab_cam = (ps_lab_cam.astype(np.float64) + ws_fb).astype(np.float32)
            track_clouds["scale_shift_cam_aligned"] = (ps_lab_cam, pl_cam)

        for track, (pred_pts, pred_pt_lab) in track_clouds.items():
            for split_name, val in (("seen", 1), ("unseen", 0)):
                p = pred_pts[pred_pt_lab == val] if len(pred_pts) == len(pred_pt_lab) \
                    else np.zeros((0, 3), dtype=np.float32)
                g_s, g_tree = gt_split_cache[val]
                if len(p) < MIN_SPLIT_PTS or len(g_s) < MIN_SPLIT_PTS:
                    cf = chamfer_and_fscore(
                        np.zeros((0, 3), dtype=np.float32),
                        np.zeros((0, 3), dtype=np.float32),
                        physical_thresholds=physical_thresholds,
                    )
                else:
                    p_s = sample_pointcloud(p, args.max_points_per_scene, seed=0xF0 + val)
                    cf = chamfer_and_fscore(
                        p_s, g_s, gt_tree=g_tree, workers=_kdtw,
                        physical_thresholds=physical_thresholds,
                    )
                record.update(prefix_metrics(cf, f"{track}_{split_name}_"))

    if _t_seen_unseen_0 is not None:
        _timer.record("seen_unseen_splits", time.perf_counter() - _t_seen_unseen_0)

    return record, plys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Hydra config name, e.g. 'room_envelopes/e1b_uf12_layout_depth_only_original_regression'")
    ap.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Required for E1-E6; for E0 the pretrained model is OK.")
    _cli.add_split(ap, default="val")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument(
        "--all_unique_scenes", action="store_true",
        help="Evaluate every unique room in the split exactly once, deterministically. "
             "Disables inside_random sampling. --max_scenes/--max_batches still apply "
             "as upper-bound caps (useful for smoke tests) but no longer pad the count."
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Per-scene RNG seed base (used only with --all_unique_scenes)."
    )
    ap.add_argument("--num_views", type=int, default=None,
                    help="Optional override for # views per scene (else dataset default)")
    _cli.add_camera_mode(ap)
    _cli.add_max_points_per_scene(ap)
    ap.add_argument(
        "--kdtree_workers", type=int, default=-1,
        help="Workers forwarded to scipy.spatial.cKDTree.query for chamfer / "
             "F-score nearest-neighbour queries. -1 (default) uses all cores; "
             "set to 1 if you wrap this script in a per-scene multiprocessing "
             "pool to avoid CPU oversubscription.",
    )
    ap.add_argument("--save_debug_every", type=int, default=25,
                    help="Save predicted+GT PLY every N scenes (0 to disable). "
                         "Alias: --ply_every.")
    ap.add_argument("--ply_every", type=int, default=None,
                    help="Alias for --save_debug_every; takes precedence if set.")
    ap.add_argument("--save_raw_ply", dest="save_raw_ply",
                    action="store_true", default=True,
                    help="Save raw pred + GT PLYs at the chosen cadence (default on).")
    ap.add_argument("--no_save_raw_ply", dest="save_raw_ply", action="store_false")
    ap.add_argument("--save_aligned_ply", dest="save_aligned_ply",
                    action="store_true", default=True,
                    help="Save scale-aligned and/or sim3-aligned pred PLYs (default on).")
    ap.add_argument("--no_save_aligned_ply", dest="save_aligned_ply", action="store_false")
    ap.add_argument("--no_save_ply", action="store_true",
                    help="Disable all PLY output (overrides --save_raw_ply / --save_aligned_ply).")
    ap.add_argument("--eval_2d_only", action="store_true")
    ap.add_argument("--eval_3d_only", action="store_true")
    _cli.add_use_depth_as_layout(ap)
    _cli.add_device(ap)
    _cli.add_extrinsics_convention(ap)
    # ----- alignment / scale-aligned 3D metrics -----
    ap.add_argument(
        "--alignment", default="auto",
        choices=("auto", "none", "scale", "sim3", "scale_shift", "scale_shift_cam",
                 "all", "all_cam"),
        help=(
            "Which alignment tracks to compute for 3D metrics. "
            "'auto' (default): camera_mode=gt → raw + scale; camera_mode=pred → raw + sim3 "
            "(the N-view orchestrator additionally adds scale_shift for 1-view manifests). "
            "'none' → raw only. 'scale' → raw + scale-only. "
            "'sim3' → raw + Sim(3). "
            "'scale_shift' (alias 'scale_shift_cam') → raw + camera-frame "
            "Room-Envelopes/LaRI scale_shift (joint xyz-scale + 1-D z-shift LSQ on "
            "the camera-local point map; 1-view only). "
            "'all' → raw + scale + sim3. "
            "'all_cam' → raw + scale + scale_shift_cam side-by-side "
            "(headline=scale_shift_cam)."
        ),
    )
    ap.add_argument(
        "--scale_alignment", default="median_depth",
        choices=("median_depth", "pointcloud_rms"),
        help=(
            "How to fit the scale-only alignment. 'median_depth' (default): "
            "scale_factor = median(gt_depth / pred_depth) over per-pixel overlap, "
            "applied to predicted depth maps before reprojection. "
            "'pointcloud_rms': fall back to centroid+RMS scaling on the fused cloud "
            "(only when per-frame overlap is unavailable)."
        ),
    )
    # ----- Profiling -----
    ap.add_argument(
        "--profile", action="store_true",
        help="Record per-stage wall-clock timings for the scene loop and "
             "write a timing_summary.json next to metrics_summary.json. "
             "Adds negligible overhead when disabled (default).",
    )
    # ----- VGGT-scene-normalized eval space -----
    ap.add_argument(
        "--eval_space", default="metric",
        choices=("metric", "vggt_scene", "both"),
        help=(
            "Which space to evaluate in. 'metric' (default): existing behavior "
            "- GT layout depths and extrinsics are raw metric metres / world frame. "
            "'vggt_scene': normalize each sample with the same helper used by "
            "training/trainer.py:_process_batch (per-sample mean valid-point "
            "distance to first-camera origin) BEFORE computing GT-side 2D and 3D "
            "metrics; predictions are passed through unchanged. Use this for "
            "checkpoints trained with the trainer-side normalization (E1-E7), "
            "raw 3D metrics become meaningful because pred and GT live in the "
            "same scene-normalized frame. 'both': run both tracks; metric-space "
            "keys keep their existing names; normalized-space keys are emitted "
            "with a 'vggt_scene_' prefix."
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    (do_raw, do_scale, do_sim3, do_scale_shift_cam,
     headline_alignment) = _resolve_alignment_tracks(
        args.alignment, args.camera_mode,
    )
    # PLY cadence: --ply_every wins; otherwise fall back to --save_debug_every.
    ply_every = args.ply_every if args.ply_every is not None else args.save_debug_every
    if args.no_save_ply:
        ply_every = 0

    if args.camera_mode == "pred":
        print(
            "[camera_mode=pred] decoding pose_enc → cameras and Umeyama-aligning "
            "predicted world frame to GT world frame for the Sim(3) metric track."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "pointclouds"

    import torch
    from hydra.utils import instantiate

    cfg, model, device = _load_model_and_cfg(args)

    split_cfg = _select_split(cfg, args.split)

    # Bypass the DynamicTorchDataset wrapper (which requires DDP init for its
    # DistributedSampler) and directly build the inner ComposedDataset using
    # the wrapper's `common_config`. This mirrors what DynamicTorchDataset
    # does at training/data/dynamic_dataloader.py:45 but skips the sampler.
    inner_ds = instantiate(
        split_cfg.dataset,
        common_config=split_cfg.common_config,
        _recursive_=False,
    )

    n_unique = None
    if args.all_unique_scenes:
        n_unique = _enable_unique_scene_mode(inner_ds)
        print(f"[data] all_unique_scenes=True: {n_unique} unique rooms in split")

    # Mask mode resolution
    has_mask_head = bool(getattr(cfg.model, "enable_layout_mask", False))
    has_normal_head = bool(getattr(cfg.model, "enable_layout_normal", False))
    has_layout_head = bool(getattr(cfg.model, "enable_layout_depth", False))

    if (not has_layout_head) and (not args.use_depth_as_layout):
        print(
            "[ERROR] cfg.model.enable_layout_depth=False and --use_depth_as_layout was not "
            "passed. Enable the layout depth head in the config or pass --use_depth_as_layout "
            "for the E0 vanilla baseline.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("=" * 70)
    print("Room-envelope reconstruction evaluation")
    print(f"  config            : {args.config}")
    print(f"  checkpoint        : {args.checkpoint}")
    print(f"  split             : {args.split}")
    print(f"  camera_mode       : {args.camera_mode}")
    print(f"  alignment         : {args.alignment}  → "
          f"raw={do_raw} scale={do_scale} sim3={do_sim3} "
          f"scale_shift_cam={do_scale_shift_cam}  "
          f"headline={headline_alignment}")
    print(f"  scale_alignment   : {args.scale_alignment}")
    print(f"  use_depth_as_layout: {args.use_depth_as_layout}")
    print(f"  has_mask_head     : {has_mask_head}")
    print(f"  has_normal_head   : {has_normal_head}")
    print(f"  has_layout_head   : {has_layout_head}")
    print(f"  device            : {device}")
    print(f"  max_batches       : {args.max_batches}")
    print(f"  max_scenes        : {args.max_scenes}")
    print(f"  all_unique_scenes : {args.all_unique_scenes}"
          + (f"  (n_unique={n_unique})" if args.all_unique_scenes else ""))
    print(f"  output_dir        : {out_dir}")
    print(f"  ply_every         : {ply_every}  "
          f"(raw={args.save_raw_ply}, aligned={args.save_aligned_ply}, "
          f"no_save_ply={args.no_save_ply})")
    print("=" * 70)

    # Resolve a fixed view count for the (idx, num_views, aspect) tuple-index
    # protocol used by ComposedDataset / TupleConcatDataset.
    if args.num_views is not None:
        fixed_views = args.num_views
    else:
        # Use a conservative default: middle of the val img_nums range.
        try:
            rng = list(split_cfg.common_config.img_nums)
            fixed_views = max(2, min(8, (rng[0] + rng[1]) // 2))
        except Exception:
            fixed_views = 2

    # Determine iteration count
    if args.all_unique_scenes:
        # n_unique was set above; len(inner_ds) now also reports this thanks to
        # _enable_unique_scene_mode rewriting child.len_train.
        n_total = n_unique
    else:
        try:
            n_total = len(inner_ds)
        except TypeError:
            n_total = 10_000
    if args.max_scenes is not None:
        n_total = min(n_total, args.max_scenes)
    if args.max_batches is not None:
        n_total = min(n_total, args.max_batches)
    print(f"[data] iterating {n_total} scenes with num_views={fixed_views}")

    # Loop over scenes
    per_scene_records: list[dict] = []
    n_pred_pts_total = 0
    n_gt_pts_total = 0
    n_views_total = 0
    n_evaluated = 0
    t_start = time.time()

    timer = StageTimer(enabled=bool(args.profile))

    if args.all_unique_scenes:
        import random as _random_mod

    for i in range(n_total):
        if args.all_unique_scenes:
            # Reproducible camera-within-room sampling
            # (the dataset loader's get_data uses np.random.choice).
            np.random.seed(args.seed + i)
            _random_mod.seed(args.seed + i)
        # ComposedDataset.__getitem__ takes a (sample_idx, num_images, aspect)
        # tuple, see training/data/composed_dataset.py:85.
        try:
            with timer.time("sample_load"):
                try:
                    sample = inner_ds[(i, fixed_views, 1.0)]
                except Exception:
                    sample = inner_ds[i]
        except Exception as e:
            print(f"[scene {i}] dataset error: {e}; skipping")
            continue

        imgs = sample.get("images")
        if imgs is None:
            continue

        # Convert images to model input (handles numpy lists and torch tensors).
        with timer.time("to_device"):
            imgs_t = _to_image_tensor(sample, device)  # (1, S, 3, H, W)
            K_t, E_t = cameras_from_sample(sample, device=device)

        # Route GT cameras through the model when OCA is enabled. For non-OCA
        # checkpoints ``forward_model`` is bit-equivalent to ``model(imgs_t)``.
        with timer.time("forward"):
            with torch.no_grad():
                preds = forward_model(model, imgs_t, intrinsics=K_t, extrinsics=E_t)
            # Forward call returns once GPU has produced tensors; subsequent
            # `.cpu().numpy()` in the 3D metric path will force a sync, so the
            # next stage absorbs any remaining latency.

        # Strip batch dim from per-frame predictions for the helpers.
        preds_one = _strip_batch_dim(preds)

        S = imgs_t.shape[1]
        n_views_total += S

        record: dict = {
            "scene_idx": i,
            "seq_name": str(sample.get("seq_name", f"scene_{i:04d}")),
            "n_views": int(S),
        }

        # ---- Decide which eval-space passes to run ----
        # `passes` is a list of (sample, key_prefix, ply_prefix, scene_scale)
        # tuples. ``scene_scale`` is None for the metric pass and equals the
        # per-sample ``vggt_scene_scale`` for the vggt_scene pass; it is
        # forwarded to ``_compute_3d_metrics_for_scene`` to compute
        # physical-equivalent F-score thresholds for the vggt_scene track.
        # For "metric": one pass with the raw sample, no key prefix.
        # For "vggt_scene": one pass with the normalized sample, "vggt_scene_" key prefix.
        # For "both": both passes, in that order.
        passes: list[tuple[dict, str, str, float | None]] = []
        if args.eval_space in ("metric", "both"):
            passes.append((sample, "", "", None))
        if args.eval_space in ("vggt_scene", "both"):
            try:
                normalized_sample, norm_info = normalize_sample_vggt_scene(sample)
                record["vggt_scene_scale"]              = norm_info["vggt_scene_scale"]
                record["vggt_scene_valid_point_count"]  = norm_info["valid_point_count"]
                record["vggt_scene_norm_source"]        = norm_info["normalization_source"]
                passes.append((
                    normalized_sample, "vggt_scene_", "vggt_scene_",
                    norm_info["vggt_scene_scale"],
                ))
            except Exception as e:
                print(f"[scene {i}] VGGT-scene normalization failed: {e}; "
                      f"skipping vggt_scene pass")
                record["vggt_scene_error"] = str(e)

        save_ply_now = (
            not args.no_save_ply
            and ply_every > 0
            and (n_evaluated % ply_every == 0)
        )

        # Track whether any 3D pass succeeded so we update aggregate counters
        # and stamp `alignment_used` exactly once.
        any_3d_ok = False

        for sample_active, key_prefix, ply_prefix, scene_scale_active in passes:
            # ---- 2D metrics ----
            if not args.eval_3d_only:
                try:
                    m2d = _compute_2d_metrics_for_scene(
                        sample_active, preds_one,
                        use_depth_as_layout=args.use_depth_as_layout,
                        has_mask_head=has_mask_head,
                        has_normal_head=has_normal_head,
                    )
                    if key_prefix:
                        for k, v in m2d.items():
                            record[f"{key_prefix}{k}"] = v
                    else:
                        record.update(m2d)
                except KeyError as e:
                    print(f"[scene {i}] 2D metrics ({key_prefix or 'metric'}) skipped: {e}")

            # ---- 3D metrics ----
            if not args.eval_2d_only:
                try:
                    m3d, plys = _compute_3d_metrics_for_scene(
                        sample=sample_active,
                        preds_one=preds_one,
                        args=args,
                        image_hw=(imgs_t.shape[-2], imgs_t.shape[-1]),
                        do_raw=do_raw,
                        do_scale=do_scale,
                        do_sim3=do_sim3,
                        do_scale_shift_cam=do_scale_shift_cam,
                        timer=timer,
                        vggt_scene_scale=scene_scale_active,
                    )
                    if key_prefix:
                        for k, v in m3d.items():
                            record[f"{key_prefix}{k}"] = v
                    else:
                        record.update(m3d)
                    if not any_3d_ok:
                        record["alignment_used"]       = headline_alignment
                        record["scale_alignment_mode"] = args.scale_alignment
                        n_pred_pts_total += int(m3d.get("n_pred_points", 0))
                        n_gt_pts_total   += int(m3d.get("n_gt_points",   0))
                        any_3d_ok = True

                    # PLY saving for this pass.
                    if save_ply_now:
                        with timer.time("ply_write"):
                            ply_dir.mkdir(parents=True, exist_ok=True)
                            if args.save_raw_ply:
                                if "gt" in plys:
                                    write_ply(
                                        str(ply_dir / f"scene_{i:04d}_{ply_prefix}gt.ply"),
                                        plys["gt"],
                                    )
                                if "pred_raw" in plys:
                                    write_ply(
                                        str(ply_dir / f"scene_{i:04d}_{ply_prefix}pred_raw.ply"),
                                        plys["pred_raw"],
                                    )
                            if args.save_aligned_ply:
                                if "pred_scale_aligned" in plys:
                                    write_ply(
                                        str(ply_dir / f"scene_{i:04d}_{ply_prefix}pred_scale_aligned.ply"),
                                        plys["pred_scale_aligned"],
                                    )
                                if "pred_sim3_aligned" in plys:
                                    write_ply(
                                        str(ply_dir / f"scene_{i:04d}_{ply_prefix}pred_sim3_aligned.ply"),
                                        plys["pred_sim3_aligned"],
                                    )
                except Exception as e:
                    print(f"[scene {i}] 3D metrics ({key_prefix or 'metric'}) error: {e}")
                    record[f"{key_prefix}error_3d"] = str(e)

        per_scene_records.append(record)
        n_evaluated += 1

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{n_total}] scenes done, {elapsed:.1f}s elapsed")

    # ---- Aggregate ----
    print(f"\n[summary] evaluated {n_evaluated} scenes, "
          f"avg views/scene={n_views_total / max(n_evaluated, 1):.2f}, "
          f"total pred points={n_pred_pts_total}, total gt points={n_gt_pts_total}")

    agg = _aggregate(per_scene_records)

    # Build summary in the requested layout (with NaN-safe lookups)
    def g(k, default=float("nan")):
        return agg.get(k, default)

    # Build nested + flat 3D blocks. Use prefixed keys produced by the new
    # tracks; backward-compat 'Chamfer' / 'FScore_*' mirror the *headline*
    # alignment so old aggregators continue to read meaningful numbers.
    def _track(prefix: str) -> dict:
        block = {
            # Back-compat (sum convention), historical headline key.
            "chamfer_l1":        g(f"{prefix}chamfer_l1"),
            "chamfer_l2":        g(f"{prefix}chamfer_l2"),
            # Explicit sum/mean variants. ``chamfer_l1`` aliases ``_sum`` for
            # back-compat; prefer ``_mean`` for paper-style comparisons.
            "chamfer_l1_sum":    g(f"{prefix}chamfer_l1_sum"),
            "chamfer_l1_mean":   g(f"{prefix}chamfer_l1_mean"),
            "chamfer_l2_sum":    g(f"{prefix}chamfer_l2_sum"),
            "chamfer_l2_mean":   g(f"{prefix}chamfer_l2_mean"),
            "accuracy_mean":     g(f"{prefix}accuracy_mean"),
            "completeness_mean": g(f"{prefix}completeness_mean"),
            "fscore_0.05":       g(f"{prefix}fscore_0.05"),
            "fscore_0.10":       g(f"{prefix}fscore_0.10"),
            "fscore_0.20":       g(f"{prefix}fscore_0.20"),
            "precision_0.10":    g(f"{prefix}precision_0.10"),
            "recall_0.10":       g(f"{prefix}recall_0.10"),
        }
        # Seen / unseen splits (present only when layout_masks was available).
        for split in ("seen", "unseen"):
            for k in ("chamfer_l1", "chamfer_l2",
                      "chamfer_l1_sum", "chamfer_l1_mean",
                      "chamfer_l2_sum", "chamfer_l2_mean",
                      "fscore_0.05", "fscore_0.10",
                      "fscore_0.20", "precision_0.10", "recall_0.10"):
                block[f"{split}_{k}"] = g(f"{prefix}{split}_{k}")
        return block

    raw_block = _track("raw_")
    scale_block = _track("scale_aligned_")
    # NOTE: the ``_aggregate(per_scene_records)`` helper computes
    # ``np.nanmean`` (see _common.py::aggregate). Several "median" keys
    # below were actually means despite their name, they are kept for
    # backward compatibility (existing JSON consumers read these names)
    # but a correctly-named ``_mean`` alias is emitted alongside each one.
    # Prefer reading the ``_mean`` key in new code.
    scale_block["scale_alignment_factor_median"] = g("scale_alignment_factor")
    scale_block["scale_alignment_factor_mean"]   = g("scale_alignment_factor")
    sim3_block = _track("sim3_aligned_")
    sim3_block["sim3_scale_median"]              = g("sim3_scale")
    sim3_block["sim3_scale_mean"]                = g("sim3_scale")
    sim3_block["sim3_rotation_det_median"]       = g("sim3_rotation_det")
    sim3_block["sim3_rotation_det_mean"]         = g("sim3_rotation_det")
    sim3_block["sim3_translation_norm_median"]   = g("sim3_translation_norm")
    sim3_block["sim3_translation_norm_mean"]     = g("sim3_translation_norm")

    headline_prefix = (
        "scale_aligned_" if headline_alignment == "scale"
        else ("sim3_aligned_" if headline_alignment == "sim3"
              else ("scale_shift_cam_aligned_" if headline_alignment == "scale_shift_cam"
                    else "raw_"))
    )
    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "split": args.split,
        "camera_mode": args.camera_mode,
        "alignment": args.alignment,
        "scale_alignment": args.scale_alignment,
        "headline_alignment": headline_alignment,
        "eval_space": args.eval_space,
        "use_depth_as_layout": args.use_depth_as_layout,
        "num_scenes": n_evaluated,
        "num_views_avg": float(n_views_total / max(n_evaluated, 1)),
        "2d": {
            "AbsRel_all": g("absrel_all"),
            "AbsRel_visible": g("absrel_visible"),
            "AbsRel_occluded": g("absrel_occluded"),
            "RMSE_all": g("rmse_all"),
            "RMSE_visible": g("rmse_visible"),
            "RMSE_occluded": g("rmse_occluded"),
            "LogRMSE_all": g("log_rmse_all"),
            "Delta1_all": g("delta1_all"),
            "Delta1_visible": g("delta1_visible"),
            "Delta1_occluded": g("delta1_occluded"),
            "Delta2_all": g("delta2_all"),
            "Delta3_all": g("delta3_all"),
            "SILog_all": g("silog_all"),
            "Mask_IoU": g("mask_iou"),
            "Mask_F1": g("mask_f1"),
            "Mask_Precision": g("mask_precision"),
            "Mask_Recall": g("mask_recall"),
            "Normal_MeanAngErr": g("normal_mean_deg"),
            "Normal_MedianAngErr": g("normal_median_deg"),
            "Normal_PctUnder11_25": g("normal_pct_under_11_25"),
            "Normal_PctUnder22_5": g("normal_pct_under_22_5"),
            "Normal_PctUnder30": g("normal_pct_under_30"),
        },
        "3d": {
            "raw":                raw_block,
            "scale_aligned":      scale_block,
            "sim3_aligned":       sim3_block,
            # Backward-compat headline keys (mirror the chosen alignment).
            "Chamfer":            g(f"{headline_prefix}chamfer_l1"),
            "ChamferL2":          g(f"{headline_prefix}chamfer_l2"),
            "Accuracy":           g(f"{headline_prefix}accuracy_mean"),
            "Completeness":       g(f"{headline_prefix}completeness_mean"),
            "FScore_0.05":        g(f"{headline_prefix}fscore_0.05"),
            "FScore_0.10":        g(f"{headline_prefix}fscore_0.10"),
            "FScore_0.20":        g(f"{headline_prefix}fscore_0.20"),
            "Precision_0.10":     g(f"{headline_prefix}precision_0.10"),
            "Recall_0.10":        g(f"{headline_prefix}recall_0.10"),
        },
        "diagnostics": {
            "median_pred_depth":      g("median_pred_depth"),
            "median_gt_depth":        g("median_gt_depth"),
            "median_pred_gt_ratio":   g("median_pred_gt_ratio"),
            "median_gt_pred_scale":   g("median_gt_pred_scale"),
            "scale_alignment_factor": g("scale_alignment_factor"),
            "sim3_scale":             g("sim3_scale"),
            "sim3_rotation_det":      g("sim3_rotation_det"),
            "sim3_translation_norm":  g("sim3_translation_norm"),
            "n_points_gt_overall":    g("n_points_gt_overall"),
            "n_points_gt_seen":       g("n_points_gt_seen"),
            "n_points_gt_unseen":     g("n_points_gt_unseen"),
            "n_points_pred_overall":  g("n_points_pred_overall"),
            "n_points_pred_seen":     g("n_points_pred_seen"),
            "n_points_pred_unseen":   g("n_points_pred_unseen"),
        },
    }

    # Head-aware: emit pose/intrinsics metrics whenever a camera head produced
    # them (any scene recorded a ``pose_*`` key), independent of --camera-mode.
    if any(k.startswith("pose_") for k in agg):
        summary["pose"] = {
            "rot_err_deg_mean":     g("pose_rot_err_deg_mean"),
            "rot_err_deg_median":   g("pose_rot_err_deg_median"),
            "rot_err_deg_p95":      g("pose_rot_err_deg_p95"),
            "trans_err_raw_m_mean": g("pose_trans_err_raw_m_mean"),
            "trans_err_raw_m_median": g("pose_trans_err_raw_m_median"),
            "ate_m":                g("pose_ate_m"),
            "sim3_scale":           g("pose_sim3_scale"),
            "focal_err_px_fx_mean": g("pose_focal_err_px_fx_mean"),
            "focal_err_px_fy_mean": g("pose_focal_err_px_fy_mean"),
            "fov_h_err_deg_mean":   g("pose_fov_h_err_deg_mean"),
            "fov_v_err_deg_mean":   g("pose_fov_v_err_deg_mean"),
        }

    # ---- VGGT-scene-normalized metrics block (when computed) ----
    if args.eval_space in ("vggt_scene", "both"):
        def vg(k, default=float("nan")):
            return agg.get(f"vggt_scene_{k}", default)

        def _vggt_track(prefix: str) -> dict:
            block = {
                # vggt_scene Chamfer is reported in normalized-space units.
                # ``chamfer_l1`` aliases the sum form (back-compat); use
                # ``chamfer_l1_mean`` for paper-style 0.5·(acc+com).
                "chamfer_l1":        vg(f"{prefix}chamfer_l1"),
                "chamfer_l2":        vg(f"{prefix}chamfer_l2"),
                "chamfer_l1_sum":    vg(f"{prefix}chamfer_l1_sum"),
                "chamfer_l1_mean":   vg(f"{prefix}chamfer_l1_mean"),
                "chamfer_l2_sum":    vg(f"{prefix}chamfer_l2_sum"),
                "chamfer_l2_mean":   vg(f"{prefix}chamfer_l2_mean"),
                "accuracy_mean":     vg(f"{prefix}accuracy_mean"),
                "completeness_mean": vg(f"{prefix}completeness_mean"),
                # Default F-score thresholds are unitless in vggt_scene space
                # (they correspond to ``threshold * avg_scale`` metres physical,
                # typically ~1-2 m at threshold 0.20, NOT 20 cm).
                "fscore_0.05":       vg(f"{prefix}fscore_0.05"),
                "fscore_0.10":       vg(f"{prefix}fscore_0.10"),
                "fscore_0.20":       vg(f"{prefix}fscore_0.20"),
                "precision_0.10":    vg(f"{prefix}precision_0.10"),
                "recall_0.10":       vg(f"{prefix}recall_0.10"),
                # Physical-equivalent F-score: threshold rescaled by
                # ``1 / vggt_scene_scale`` so the gate is exactly
                # "physical distance < {0.05, 0.10, 0.20} m". Use these for
                # direct comparison against metric-track F-scores.
                "fscore_physical_0.05m":    vg(f"{prefix}fscore_physical_0.05m"),
                "fscore_physical_0.10m":    vg(f"{prefix}fscore_physical_0.10m"),
                "fscore_physical_0.20m":    vg(f"{prefix}fscore_physical_0.20m"),
                "precision_physical_0.10m": vg(f"{prefix}precision_physical_0.10m"),
                "recall_physical_0.10m":    vg(f"{prefix}recall_physical_0.10m"),
            }
            for split in ("seen", "unseen"):
                for k in ("chamfer_l1", "chamfer_l2",
                          "chamfer_l1_sum", "chamfer_l1_mean",
                          "chamfer_l2_sum", "chamfer_l2_mean",
                          "fscore_0.05", "fscore_0.10",
                          "fscore_0.20", "precision_0.10", "recall_0.10",
                          "fscore_physical_0.05m", "fscore_physical_0.10m",
                          "fscore_physical_0.20m"):
                    block[f"{split}_{k}"] = vg(f"{prefix}{split}_{k}")
            return block

        summary["vggt_scene"] = {
            "2d": {
                "AbsRel_all":         vg("absrel_all"),
                "AbsRel_visible":     vg("absrel_visible"),
                "AbsRel_occluded":    vg("absrel_occluded"),
                "RMSE_all":           vg("rmse_all"),
                "Delta1_all":         vg("delta1_all"),
                "Delta1_occluded":    vg("delta1_occluded"),
                "SILog_all":          vg("silog_all"),
                "Mask_IoU":           vg("mask_iou"),
                "Mask_F1":            vg("mask_f1"),
                "Normal_MeanAngErr":  vg("normal_mean_deg"),
            },
            "3d": {
                "raw":                _vggt_track("raw_"),
                "scale_aligned":      _vggt_track("scale_aligned_"),
                "sim3_aligned":       _vggt_track("sim3_aligned_"),
            },
            "diagnostics": {
                "scene_scale":            agg.get("vggt_scene_scale"),
                "median_pred_depth":      vg("median_pred_depth"),
                "median_gt_depth":        vg("median_gt_depth"),
                "median_pred_gt_ratio":   vg("median_pred_gt_ratio"),
                "median_gt_pred_scale":   vg("median_gt_pred_scale"),
                "scale_alignment_factor": vg("scale_alignment_factor"),
            },
        }

    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {summary_path}")
    if ply_every > 0 and ply_dir.exists():
        n_plys = len(list(ply_dir.glob('*.ply')))
        print(f"Saved: {n_plys} PLY debug files → {ply_dir}")

    if timer.enabled:
        timing_path = out_dir / "timing_summary.json"
        timer.dump_json(timing_path)
        print(f"Saved: {timing_path}")
        print("\n[profile] Per-stage wall-clock timings (sorted by total):")
        timer.print_table(prefix="  ")


if __name__ == "__main__":
    main()
