"""Clean Weights & Biases qualitative logging for room-envelope eval / training.

This module is intentionally SEPARATE from
``training/train_utils/wandb_logger.py``. The legacy module emits side-by-side
GT|PRED, overlays, FP/FN error maps, depth |pred-gt| error heatmaps and angular
error visuals. This module emits **clean** standalone images only:

  - RGB
  - predicted layout depth (colormapped)
  - GT layout depth (colormapped, when available)
  - predicted layout mask (sigmoid probability, greyscale)
  - GT layout mask (greyscale)
  - predicted normals (RGB-mapped)
  - GT normals (RGB-mapped)
  - optional predicted metric depth (colormapped)

Plus W&B ``Object3D`` point-cloud logging:

  - predicted scene reconstruction (RGB-colored if RGB is available)
  - GT scene reconstruction when ``log_gt_3d`` is True and GT layout_depth+masks
    are available

Subsampling to ``max_points_preview`` keeps the W&B preview cheap.

All public functions are safe to call from every DDP rank: they silently no-op
on non-rank-0 processes and when wandb is not installed / no active run is
present. Missing GT, missing prediction heads, or shape surprises are caught
and logged at DEBUG.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

# Reuse the *clean* helpers from the legacy logger. We do NOT import overlay /
# error / side-by-side helpers here.
from train_utils.wandb_logger import (  # type: ignore  # noqa: E402
    _apply_colormap,
    prepare_binary_mask_for_vis,
    prepare_depth_for_vis,
    prepare_normal_for_vis,
    prepare_rgb_for_vis,
)

# Make the geometry helpers (training/geometry) and vggt utils importable both
# from training-time runs (cwd = training/) and offline eval (cwd = repo root).
# Be defensive in case this module is imported from another cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_TRAINING_DIR)
for _p in (_REPO_ROOT, _TRAINING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle helpers (mirror wandb_logger.py, single source of truth)
# ---------------------------------------------------------------------------

def _is_main_process() -> bool:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _wandb():
    """Return the wandb module if installed AND a run is active, else None."""
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    if getattr(wandb, "run", None) is None:
        return None
    return wandb


# ---------------------------------------------------------------------------
# Tensor utilities, accept torch tensors, numpy arrays, lists, or None
# ---------------------------------------------------------------------------

def _to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    if isinstance(x, list):
        try:
            return np.asarray(x)
        except Exception:
            return None
    try:
        return np.asarray(x)
    except Exception:
        return None


def _index(arr, idx):
    """Index along axis 0 from list or array."""
    if arr is None:
        return None
    if isinstance(arr, list):
        return arr[idx]
    return arr[idx]


def _squeeze_last1(a: np.ndarray) -> np.ndarray:
    """If the trailing axis has size 1, drop it."""
    if a.ndim >= 1 and a.shape[-1] == 1:
        return a[..., 0]
    return a


def _looks_batched(batch: Mapping, predictions: Mapping) -> bool:
    """Detect whether ``batch`` is B-leading (training) or S-leading (eval).

    Heuristic: ``images`` shape decides. (B,S,3,H,W) → batched.
    (S,3,H,W) or (S,H,W,3) → per-scene.
    """
    images = batch.get("images")
    if images is None:
        # Fall back to predictions
        pred_ld = predictions.get("layout_depth")
        if pred_ld is None:
            pred_ld = predictions.get("depth")
        if pred_ld is None:
            return False
        return pred_ld.ndim >= 5

    if hasattr(images, "ndim"):
        return images.ndim >= 5
    return False


def _scene_view(d: Mapping, b: Optional[int]) -> dict:
    """Return a per-scene view of ``d`` by indexing axis 0 if batched."""
    if b is None:
        return dict(d)
    out: dict = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        if torch.is_tensor(v):
            if v.ndim >= 1 and v.shape[0] > b:
                out[k] = v[b]
            else:
                out[k] = v
        elif isinstance(v, np.ndarray):
            if v.ndim >= 1 and v.shape[0] > b:
                out[k] = v[b]
            else:
                out[k] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Clean 2D panel collection (NO overlays, NO errors)
# ---------------------------------------------------------------------------

def _rgb_uint8_for_view(images, s: int) -> Optional[np.ndarray]:
    """Return (H, W, 3) uint8 for view ``s``.

    Handles (S, 3, H, W) float in [0,1], (S, H, W, 3) uint8/float in [0,1],
    or torch tensors of either layout.
    """
    if images is None:
        return None
    try:
        if torch.is_tensor(images):
            t = images[s]
            if t.ndim == 3 and t.shape[0] == 3:
                return prepare_rgb_for_vis(t)
            if t.ndim == 3 and t.shape[-1] == 3:
                f = t.detach().cpu().float()
                if f.max() > 1.5:
                    f = f / 255.0
                f = f.clamp(0.0, 1.0)
                return (f.numpy() * 255).astype(np.uint8)
        else:
            arr = np.asarray(images[s])
            if arr.ndim == 3 and arr.shape[0] == 3:
                t = torch.from_numpy(arr.astype(np.float32))
                return prepare_rgb_for_vis(t)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                if arr.dtype == np.uint8:
                    return arr
                f = arr.astype(np.float32)
                if f.max() > 1.5:
                    f = f / 255.0
                return (np.clip(f, 0, 1) * 255).astype(np.uint8)
    except Exception as exc:
        log.debug(f"_rgb_uint8_for_view failed: {exc}")
    return None


def _depth_to_clean_color(depth, mask=None) -> Optional[np.ndarray]:
    """(H, W) → uint8 RGB via robust percentile normalization + turbo colormap.

    Returns None if the input is unusable.
    """
    if depth is None:
        return None
    try:
        t = depth if torch.is_tensor(depth) else torch.as_tensor(np.asarray(depth))
        if t.ndim == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.ndim != 2:
            return None
        m_t = None
        if mask is not None:
            m_t = mask if torch.is_tensor(mask) else torch.as_tensor(np.asarray(mask))
            if m_t.ndim == 3 and m_t.shape[-1] == 1:
                m_t = m_t.squeeze(-1)
        norm = prepare_depth_for_vis(t, mask=m_t)
        return _apply_colormap(norm)
    except Exception as exc:
        log.debug(f"_depth_to_clean_color failed: {exc}")
        return None


def _binary_mask_clean(mask) -> Optional[np.ndarray]:
    if mask is None:
        return None
    try:
        t = mask if torch.is_tensor(mask) else torch.as_tensor(np.asarray(mask))
        if t.dtype != torch.float32:
            t = t.float()
        return prepare_binary_mask_for_vis(t)
    except Exception as exc:
        log.debug(f"_binary_mask_clean failed: {exc}")
        return None


def _normal_clean(normal) -> Optional[np.ndarray]:
    if normal is None:
        return None
    try:
        t = normal if torch.is_tensor(normal) else torch.as_tensor(np.asarray(normal))
        return prepare_normal_for_vis(t)
    except Exception as exc:
        log.debug(f"_normal_clean failed: {exc}")
        return None


def _collect_clean_panels(scene: Mapping,
                          preds: Mapping,
                          s: int,
                          use_depth_as_layout: bool = False) -> dict:
    """Build all clean per-view panels for view ``s`` of a per-scene dict.

    Returns a dict ``{name: (H,W,3) uint8 or (H,W) uint8}`` containing only
    the panels we could produce (missing GT does not raise).
    """
    panels: dict = {}

    # RGB
    rgb = _rgb_uint8_for_view(scene.get("images"), s)
    if rgb is not None:
        panels["rgb"] = rgb

    # Predicted layout depth (fall back to predicted metric depth on request)
    pred_ld = preds.get("layout_depth")
    pred_depth_field = "layout_depth"
    if pred_ld is None and use_depth_as_layout:
        pred_ld = preds.get("depth")
        pred_depth_field = "depth"
    if pred_ld is not None:
        try:
            pred_ld_s = pred_ld[s]
            img = _depth_to_clean_color(pred_ld_s)
            if img is not None:
                panels[f"pred_{pred_depth_field}"] = img
        except Exception as exc:
            log.debug(f"pred layout depth panel failed: {exc}")

    # Predicted metric depth (separate from layout depth), only when both heads
    # exist and we are not already using depth-as-layout.
    if not use_depth_as_layout:
        pred_d = preds.get("depth")
        if pred_d is not None and pred_ld is not None and pred_d is not pred_ld:
            try:
                img = _depth_to_clean_color(pred_d[s])
                if img is not None:
                    panels["pred_depth"] = img
            except Exception as exc:
                log.debug(f"pred metric depth panel failed: {exc}")

    # GT layout depth
    gt_ld = scene.get("layout_depths")
    gt_ldm = scene.get("layout_depth_masks")
    if gt_ld is not None:
        try:
            m = gt_ldm[s] if gt_ldm is not None else None
            img = _depth_to_clean_color(gt_ld[s], mask=m)
            if img is not None:
                panels["gt_layout_depth"] = img
        except Exception as exc:
            log.debug(f"gt layout depth panel failed: {exc}")

    # GT metric depth (useful when the head predicts it; non-fatal otherwise)
    gt_d = scene.get("depths")
    if gt_d is not None:
        try:
            img = _depth_to_clean_color(gt_d[s])
            if img is not None:
                panels["gt_depth"] = img
        except Exception as exc:
            log.debug(f"gt metric depth panel failed: {exc}")

    # Predicted layout mask (sigmoid → greyscale)
    pred_ml = preds.get("layout_mask_logits")
    if pred_ml is not None:
        try:
            t = pred_ml[s]
            if torch.is_tensor(t):
                if t.ndim == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)
                prob = torch.sigmoid(t.detach().cpu().float())
            else:
                arr = np.asarray(t)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                prob = torch.from_numpy(1.0 / (1.0 + np.exp(-arr)))
            img = _binary_mask_clean(prob)
            if img is not None:
                panels["pred_layout_mask"] = img
        except Exception as exc:
            log.debug(f"pred layout mask panel failed: {exc}")

    # GT layout mask
    gt_lm = scene.get("layout_masks")
    if gt_lm is not None:
        try:
            img = _binary_mask_clean(gt_lm[s])
            if img is not None:
                panels["gt_layout_mask"] = img
        except Exception as exc:
            log.debug(f"gt layout mask panel failed: {exc}")

    # Predicted normals (accept multiple alias keys)
    _PRED_NORMAL_KEYS = ("layout_normal_pred", "pred_layout_normals",
                         "layout_normals_pred", "layout_normal")
    pred_n = None
    for k in _PRED_NORMAL_KEYS:
        v = preds.get(k)
        if v is not None:
            pred_n = v
            break
    if pred_n is not None:
        try:
            img = _normal_clean(pred_n[s])
            if img is not None:
                panels["pred_layout_normal"] = img
        except Exception as exc:
            log.debug(f"pred layout normal panel failed: {exc}")

    # GT normals
    _GT_NORMAL_KEYS = ("layout_normals", "layout_normal_maps", "layout_normal")
    gt_n = None
    for k in _GT_NORMAL_KEYS:
        v = scene.get(k)
        if v is not None and v is not pred_n:
            gt_n = v
            break
    if gt_n is not None:
        try:
            img = _normal_clean(gt_n[s])
            if img is not None:
                panels["gt_layout_normal"] = img
        except Exception as exc:
            log.debug(f"gt layout normal panel failed: {exc}")

    return panels


# ---------------------------------------------------------------------------
# 3D point cloud collection (RGB-colored)
# ---------------------------------------------------------------------------

def _depth_to_world_points_np(depth: np.ndarray,
                              intr: np.ndarray,
                              extr: np.ndarray,
                              mask: np.ndarray,
                              extrinsics_convention: str = "w2c") -> tuple[np.ndarray, np.ndarray]:
    """Unproject one view to world-frame points; return (pts_world, pixel_idx).

    ``pixel_idx`` is the flat-index of kept pixels (so the caller can sample
    matching RGB).
    """
    # Unproject inline (same math as the eval pipeline's
    # backproject_depth_to_world) while also exposing which pixels survived, so
    # the caller can sample matching RGB without re-running the math.
    H, W = depth.shape
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])

    if mask is None:
        mask = depth > 1e-6
    else:
        mask = (mask.astype(bool)) & (depth > 1e-6)

    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    flat_keep = np.flatnonzero(mask.reshape(-1))
    vs, us = np.divmod(flat_keep, W)
    z = depth.reshape(-1)[flat_keep].astype(np.float32)
    x = (us.astype(np.float32) - cx) / fx * z
    y = (vs.astype(np.float32) - cy) / fy * z
    cam_pts = np.stack([x, y, z], axis=-1)  # (N, 3)

    extr_arr = np.asarray(extr, dtype=np.float32)
    if extr_arr.shape == (4, 4):
        extr_3x4 = extr_arr[:3, :]
    elif extr_arr.shape == (3, 4):
        extr_3x4 = extr_arr
    else:
        raise ValueError(
            f"Expected extrinsics of shape (3,4) or (4,4); got {extr_arr.shape}"
        )

    if extrinsics_convention == "w2c":
        from vggt.utils.geometry import closed_form_inverse_se3  # type: ignore
        ext_4x4 = np.eye(4, dtype=np.float32)
        ext_4x4[:3, :] = extr_3x4
        c2w = closed_form_inverse_se3(ext_4x4[None])[0]
        R = c2w[:3, :3]
        t = c2w[:3, 3]
    else:  # "c2w", extrinsics already camera-to-world
        R = extr_3x4[:3, :3]
        t = extr_3x4[:3, 3]

    world = cam_pts @ R.T + t
    return world.astype(np.float32), flat_keep


def _build_colored_pointcloud(scene: Mapping,
                              preds: Mapping,
                              mode: str,
                              use_depth_as_layout: bool = False,
                              mask_threshold: float = 0.5) -> Optional[np.ndarray]:
    """Build an (N, 6) [x, y, z, r, g, b] world-frame point cloud.

    Mirrors ``evaluations/src/3d/pointcloud.build_scene_pointcloud_from_batch``
    but also samples per-pixel RGB so the cloud is colored.

    Args:
        mode: "pred" or "gt".

    Returns None on missing inputs or failure (do not crash callers).
    """
    if mode not in ("pred", "gt"):
        raise ValueError(f"mode must be 'pred' or 'gt'; got {mode!r}")

    intr = _to_np(scene.get("intrinsics"))
    extr = _to_np(scene.get("extrinsics"))
    if intr is None or extr is None:
        log.debug("3D build skipped: missing intrinsics/extrinsics")
        return None

    images = scene.get("images")  # for RGB sampling
    if images is not None and torch.is_tensor(images):
        images_np = images.detach().cpu().float().numpy()
    else:
        images_np = _to_np(images)

    S = extr.shape[0] if hasattr(extr, "shape") and extr.ndim >= 2 else len(extr)

    # Resolve depth source
    if mode == "gt":
        depth_src = _to_np(scene.get("layout_depths"))
        valid_src = _to_np(scene.get("layout_depth_masks"))
        if depth_src is None:
            return None
    else:
        depth_src = _to_np(preds.get("layout_depth"))
        if depth_src is None and use_depth_as_layout:
            depth_src = _to_np(preds.get("depth"))
        if depth_src is None:
            return None
        valid_src = None  # only the depth-validity gate for preds

    if depth_src.ndim == 4 and depth_src.shape[-1] == 1:
        depth_src = depth_src[..., 0]

    # Resolve pred mask (only used when mode=="pred")
    pred_mask = None
    if mode == "pred":
        ml = _to_np(preds.get("layout_mask_logits"))
        if ml is not None:
            if ml.ndim == 4 and ml.shape[1] == 1:
                ml = ml[:, 0]
            pred_mask = 1.0 / (1.0 + np.exp(-ml))  # sigmoid

    chunks: list = []
    for s in range(S):
        try:
            depth_s = np.asarray(depth_src[s], dtype=np.float32)
            intr_s = np.asarray(intr[s])
            extr_s = np.asarray(extr[s])

            keep = depth_s > 1e-6
            if mode == "gt" and valid_src is not None:
                keep = keep & np.asarray(valid_src[s]).astype(bool)
            if mode == "pred" and pred_mask is not None:
                keep = keep & (pred_mask[s] > mask_threshold)

            pts, flat_idx = _depth_to_world_points_np(depth_s, intr_s, extr_s, keep)
            if len(pts) == 0:
                continue

            # Sample RGB for the same pixels
            rgb_view = _rgb_uint8_for_view(images, s) if images is not None else None
            if rgb_view is None and images_np is not None:
                # Final fallback: synthesise from numpy stash
                try:
                    arr = images_np[s]
                    if arr.ndim == 3 and arr.shape[0] == 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.dtype != np.uint8:
                        f = arr.astype(np.float32)
                        if f.max() > 1.5:
                            f = f / 255.0
                        arr = (np.clip(f, 0, 1) * 255).astype(np.uint8)
                    rgb_view = arr
                except Exception:
                    rgb_view = None

            if rgb_view is not None and rgb_view.shape[:2] == depth_s.shape:
                colors = rgb_view.reshape(-1, 3)[flat_idx]
            else:
                # No RGB available → fall back to a flat colour per cloud type
                fill = np.array(
                    [180, 60, 60] if mode == "pred" else [60, 160, 80],
                    dtype=np.uint8,
                )
                colors = np.tile(fill, (len(pts), 1))

            chunks.append(np.concatenate([pts, colors.astype(np.float32)], axis=1))
        except Exception as exc:
            log.debug(f"view {s} 3D build failed: {exc}")
            continue

    if not chunks:
        return None
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _subsample(pts_rgb: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(pts_rgb) <= max_points:
        return pts_rgb
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts_rgb), int(max_points), replace=False)
    return pts_rgb[idx]


def _make_object3d(pts_rgb: np.ndarray, wb_module) -> Optional[Any]:
    """Construct wandb.Object3D with the most robust schema available.

    Tries (N,6) xyz+rgb first (supported in wandb ≥ 0.10ish). Falls back to
    (N,3) on any schema rejection.
    """
    try:
        return wb_module.Object3D(pts_rgb.astype(np.float32))
    except Exception as exc_rgb:
        log.debug(f"Object3D (N,6) failed: {exc_rgb}; falling back to (N,3)")
        try:
            return wb_module.Object3D(pts_rgb[:, :3].astype(np.float32))
        except Exception as exc_xyz:
            log.debug(f"Object3D (N,3) failed: {exc_xyz}")
            return None


def _save_ply(path: str, pts_rgb: np.ndarray) -> None:
    """Write a coloured PLY via the geometry helper, with an inline fallback."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        from geometry.room_envelope_geometry import save_point_cloud_ply  # type: ignore
        if pts_rgb.shape[1] >= 6:
            save_point_cloud_ply(pts_rgb[:, :3].astype(np.float32), path,
                                 colors=pts_rgb[:, 3:6].astype(np.uint8))
        else:
            save_point_cloud_ply(pts_rgb.astype(np.float32), path)
        return
    except Exception:
        pass
    # Inline ASCII PLY fallback
    pts = pts_rgb[:, :3].astype(np.float32)
    has_color = pts_rgb.shape[1] >= 6
    colors = pts_rgb[:, 3:6].astype(np.uint8) if has_color else None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(pts)):
            x, y, z = pts[i]
            if has_color:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def log_clean_visuals(
    *,
    batch: Mapping,
    predictions: Mapping,
    phase: str,
    wandb_step: Optional[int] = None,
    commit: Optional[bool] = None,
    step: Optional[int] = None,         # backward-compat alias for ``wandb_step``
    epoch: Optional[int] = None,
    local_step: Optional[int] = None,
    scene_index: Optional[int] = None,
    max_samples: int = 2,
    view_indices: Optional[Sequence[int]] = None,
    log_2d: bool = True,
    log_3d: bool = True,
    max_points_preview: int = 50_000,
    log_gt_3d: bool = True,
    save_full_pointcloud_dir: Optional[str] = None,
    use_depth_as_layout: bool = False,
    tag: Optional[str] = None,
) -> None:
    """Emit clean qualitative 2D + 3D visuals to W&B.

    Safe to call from any rank; non-rank-0 and missing-wandb cases no-op.

    W&B requires the explicit ``step=`` passed to ``wandb.log`` to be
    monotonically increasing across the *whole* run. For validation logs
    (whose local val step lags the train step) we therefore stage with
    ``commit=False`` instead of passing a smaller step explicitly, the staged
    data rides along on the next train-side commit. The legacy ``step`` kwarg
    is kept as a thin alias for ``wandb_step`` so older callers (e.g. offline
    eval that passes a monotonic scene index) don't need to change.

    Args:
        batch: per-batch (B-leading) tensors during training, or per-scene
            (S-leading) dict during offline eval. The function auto-detects.
        predictions: model output dict, same layout convention as ``batch``.
        phase: tag for the W&B key namespace, e.g. "train", "val", "eval".
        wandb_step: the actual step passed to ``wandb.log``. **Must be
            monotonic across the run.** ``None`` means "stage, don't commit".
        commit: explicit override. ``False`` ⇒ stage (no ``step=``);
            ``True`` (or ``None`` with a non-None ``wandb_step``) ⇒ commit.
        step: legacy alias for ``wandb_step`` (back-compat). Ignored if
            ``wandb_step`` is also passed.
        epoch: epoch tag, logged as a scalar and included in captions.
        local_step: phase-local step counter (e.g. ``self.steps["val"]``),
            logged as a scalar for diagnostics. NEVER passed as ``wandb_step``.
        scene_index: per-eval-run scene index, logged as a scalar.
        max_samples: cap on batch items to log when ``batch`` is B-leading.
        view_indices: which view(s) per scene to image-log. Defaults to ``[0]``.
        log_2d: emit RGB / depth / mask / normal panels.
        log_3d: emit ``wandb.Object3D`` for the predicted point cloud (and GT
            cloud when ``log_gt_3d``).
        max_points_preview: subsample target for W&B Object3D preview.
        log_gt_3d: also build + log GT 3D cloud when GT depth is available.
        save_full_pointcloud_dir: if set, write full-resolution PLY files into
            this directory; the W&B preview is still subsampled.
        use_depth_as_layout: forward to point-cloud construction; needed for E0.
        tag: optional sub-tag (e.g. seq_name) appended to log keys.
    """
    if not _is_main_process():
        return
    wb = _wandb()
    if wb is None:
        return

    # Resolve the back-compat alias. ``wandb_step`` wins if both are set.
    if wandb_step is None and step is not None:
        wandb_step = step

    view_indices = list(view_indices) if view_indices else [0]

    batched = _looks_batched(batch, predictions)
    n_to_log = max_samples if batched else 1

    logs: dict = {}

    # Phase-level metadata scalars, visible in W&B charts even when we
    # stage (commit=False). NEVER push the val-local step as the run step,
    # but expose it here so the user can read it.
    meta_scope = f"Clean/{phase}"
    if epoch is not None:
        logs[f"{meta_scope}/epoch"] = int(epoch)
    if local_step is not None:
        logs[f"{meta_scope}/local_step"] = int(local_step)
    if scene_index is not None:
        logs[f"{meta_scope}/scene_index"] = int(scene_index)
    if wandb_step is not None:
        logs[f"{meta_scope}/global_step"] = int(wandb_step)

    for b in range(n_to_log):
        b_idx = b if batched else None
        scene = _scene_view(batch, b_idx)
        preds = _scene_view(predictions, b_idx)

        # Disambiguate the W&B key namespace
        scope_parts = ["Clean", phase]
        if tag is not None:
            scope_parts.append(str(tag))
        elif batched:
            scope_parts.append(f"b{b}")
        scope = "/".join(scope_parts)

        caption_bits = []
        if epoch is not None:
            caption_bits.append(f"ep={epoch}")
        if local_step is not None:
            caption_bits.append(f"step={local_step}")
        if wandb_step is not None:
            caption_bits.append(f"gs={wandb_step}")
        if tag is not None:
            caption_bits.append(f"scene={tag}")
        elif batched:
            caption_bits.append(f"b={b}")
        caption_prefix = " | ".join(caption_bits) if caption_bits else phase

        # ---- 2D panels ----------------------------------------------------
        if log_2d:
            for s in view_indices:
                try:
                    panels = _collect_clean_panels(
                        scene, preds, s,
                        use_depth_as_layout=use_depth_as_layout,
                    )
                except Exception as exc:
                    log.debug(f"clean panel collection failed: {exc}")
                    panels = {}
                for name, img in panels.items():
                    key = f"{scope}/v{s}/{name}"
                    try:
                        logs.setdefault(key, []).append(
                            wb.Image(img, caption=f"{caption_prefix} | view={s} | {name}")
                        )
                    except Exception as exc:
                        log.debug(f"wb.Image construction failed for {name}: {exc}")

        # ---- 3D point clouds ---------------------------------------------
        if log_3d:
            try:
                pred_pts = _build_colored_pointcloud(
                    scene, preds, mode="pred",
                    use_depth_as_layout=use_depth_as_layout,
                )
            except Exception as exc:
                log.debug(f"pred 3D build failed: {exc}")
                pred_pts = None
            if pred_pts is not None and len(pred_pts) > 0:
                preview = _subsample(pred_pts, max_points_preview)
                obj = _make_object3d(preview, wb)
                if obj is not None:
                    logs[f"{scope}/recon_pred"] = obj
                if save_full_pointcloud_dir:
                    try:
                        suffix = str(tag) if tag is not None else f"b{b}"
                        path = os.path.join(save_full_pointcloud_dir,
                                            f"{suffix}_pred.ply")
                        _save_ply(path, pred_pts)
                    except Exception as exc:
                        log.debug(f"PLY save (pred) failed: {exc}")

            if log_gt_3d:
                try:
                    gt_pts = _build_colored_pointcloud(scene, preds, mode="gt")
                except Exception as exc:
                    log.debug(f"gt 3D build failed: {exc}")
                    gt_pts = None
                if gt_pts is not None and len(gt_pts) > 0:
                    preview = _subsample(gt_pts, max_points_preview)
                    obj = _make_object3d(preview, wb)
                    if obj is not None:
                        logs[f"{scope}/recon_gt"] = obj
                    if save_full_pointcloud_dir:
                        try:
                            suffix = str(tag) if tag is not None else f"b{b}"
                            path = os.path.join(save_full_pointcloud_dir,
                                                f"{suffix}_gt.ply")
                            _save_ply(path, gt_pts)
                        except Exception as exc:
                            log.debug(f"PLY save (gt) failed: {exc}")

    if not logs:
        return
    # Decide whether to commit or stage. Staging (commit=False) is required
    # whenever the caller's natural step counter lags the run-global step
    # (the val-loop case). The legacy ``step=...`` argument is preserved by
    # routing through ``wandb_step`` above.
    do_stage = (commit is False) or (commit is None and wandb_step is None)
    try:
        if do_stage:
            wb.log(logs, commit=False)
        elif wandb_step is None:
            wb.log(logs)
        else:
            wb.log(logs, step=int(wandb_step))
    except Exception as exc:
        log.debug(f"wandb.log (clean) failed: {exc}")
