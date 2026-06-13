"""Point-cloud construction: depth unprojection + multi-view fusion.

Camera convention notes live in the original reconstruction_utils docstring;
extrinsics are (3,4) camera-from-world OpenCV matrices.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Path setup (training/ on sys.path) is performed by common/_paths.py,
# imported by the entry-point runner before this module loads.
from geometry.room_envelope_geometry import (
    unproject_depth_to_camera_points,
    transform_camera_to_world,
)


def _to_3x4(extrinsics: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsics, dtype=np.float32)
    if e.shape == (4, 4):
        return e[:3, :].copy()
    if e.shape == (3, 4):
        return e.copy()
    raise ValueError(f"Expected extrinsics of shape (3,4) or (4,4); got {e.shape}")


def backproject_depth_to_world(depth: np.ndarray,
                               intrinsics: np.ndarray,
                               extrinsics: np.ndarray,
                               mask: np.ndarray | None = None,
                               extrinsics_convention: str = "w2c") -> np.ndarray:
    """Backproject a single-view depth map to a flat (N, 3) world-frame point cloud.

    Args:
        depth:       (H, W) float depth in metres.
        intrinsics:  (3, 3) OpenCV intrinsic matrix.
        extrinsics:  (3, 4) or (4, 4) camera transform. Convention selectable
            via ``extrinsics_convention`` (default ``"w2c"`` matches the
            dataset loader).
        mask:        optional (H, W) bool, keep only ``True`` pixels in
            addition to ``depth > 1e-6``.
        extrinsics_convention: ``"w2c"`` (camera-from-world, dataset default)
            or ``"c2w"`` (world-from-camera, e.g. some external annotations).

    Returns:
        (N, 3) float32 world points.
    """
    ext = _to_3x4(extrinsics)
    intr = np.asarray(intrinsics, dtype=np.float32)
    cam_pts = unproject_depth_to_camera_points(depth.astype(np.float32), intr)

    if extrinsics_convention == "w2c":
        world_pts = transform_camera_to_world(cam_pts, ext)
    elif extrinsics_convention == "c2w":
        # Treat ext directly as world-from-camera: X_world = R @ X_cam + t.
        R = ext[:3, :3]
        t = ext[:3, 3]
        world_pts = (cam_pts.reshape(-1, 3) @ R.T + t).reshape(cam_pts.shape).astype(np.float32)
    else:
        raise ValueError(f"extrinsics_convention must be 'w2c' or 'c2w'; got {extrinsics_convention!r}")

    # `depth > 1e-6` rejects NaN (NaN > x is False), but *not* +inf, and
    # the layout-depth head's ``exp(x)`` activation can overflow there.
    # ``np.isfinite`` filters NaN and ±inf together so downstream KDTree
    # queries never see points-at-infinity.
    valid = np.isfinite(depth) & (depth > 1e-6)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    return world_pts.reshape(-1, 3)[valid.reshape(-1)]


def _maybe_array(x):
    """Convert a list-of-arrays or array to an indexable container."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return x
    # torch tensor
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _per_frame(arr, s: int):
    """Index frame ``s`` from a list of arrays or a stacked array."""
    if isinstance(arr, list):
        return np.asarray(arr[s])
    return np.asarray(arr[s])


def build_scene_pointcloud_from_batch(batch: dict,
                                      predictions: dict,
                                      mode: str = "pred",
                                      use_depth_as_layout: bool = False,
                                      extrinsics_convention: str = "w2c") -> np.ndarray:
    """Build a single-scene point cloud (concatenated across views).

    Args:
        batch:       per-scene dict from the dataset loader. Expects keys
            ``layout_depths``, ``layout_depth_masks``, ``layout_masks``
            (optional), ``intrinsics``, ``extrinsics``. Per-frame entries can
            be lists or stacked arrays.
        predictions: per-scene dict from model inference. Keys honoured:
            ``layout_depth`` (S,H,W) or (S,H,W,1); ``depth`` (used if
            ``use_depth_as_layout`` is True and no layout_depth); and
            ``layout_mask_logits`` (S,1,H,W).
        mode:        ``"pred"`` to build the predicted cloud, ``"gt"`` for the
            GT cloud. The predicted cloud is gated by **depth validity only**
            (never by a layout mask); the GT cloud is additionally gated by
            ``layout_depth_masks`` (depth validity).
        use_depth_as_layout: if True and the model has no ``layout_depth``,
            use ``predictions["depth"]`` as the layout-depth proxy. Used for
            the E0 vanilla-VGGT baseline.
        extrinsics_convention: see :func:`backproject_depth_to_world`.

    Returns:
        (N, 3) float32 concatenated world-frame points.
    """
    if mode not in ("pred", "gt"):
        raise ValueError(f"mode must be 'pred' or 'gt'; got {mode!r}")

    intrinsics = _maybe_array(batch.get("intrinsics"))
    extrinsics = _maybe_array(batch.get("extrinsics"))
    if intrinsics is None or extrinsics is None:
        raise KeyError("batch missing 'intrinsics' or 'extrinsics', required for backprojection")

    S = len(extrinsics) if isinstance(extrinsics, list) else extrinsics.shape[0]

    # Pull depth source
    if mode == "gt":
        depth_src = _maybe_array(batch["layout_depths"])
        valid_src = _maybe_array(batch.get("layout_depth_masks"))
    else:
        if "layout_depth" in predictions:
            ld = predictions["layout_depth"]
            # (S,H,W,1) → (S,H,W); torch tensor or numpy
            if hasattr(ld, "detach"):
                ld = ld.detach().cpu().numpy()
            ld = np.asarray(ld)
            if ld.ndim == 4 and ld.shape[-1] == 1:
                ld = ld[..., 0]
            depth_src = ld
        elif use_depth_as_layout and "depth" in predictions:
            d = predictions["depth"]
            if hasattr(d, "detach"):
                d = d.detach().cpu().numpy()
            d = np.asarray(d)
            if d.ndim == 4 and d.shape[-1] == 1:
                d = d[..., 0]
            depth_src = d
        else:
            raise KeyError(
                "predictions has neither 'layout_depth' nor a usable 'depth' "
                "(set use_depth_as_layout=True for E0 vanilla baseline)."
            )
        valid_src = None  # depth-validity mask only for predictions

    chunks = []
    for s in range(S):
        depth_s = _per_frame(depth_src, s).astype(np.float32)
        intr_s = _per_frame(intrinsics, s)
        extr_s = _per_frame(extrinsics, s)

        # Match the np.isfinite gate in backproject_depth_to_world so the
        # per-frame keep mask and the per-pixel valid mask agree (otherwise
        # the labeled-cloud row counts in seen/unseen reuse would drift
        # against the backproject_depth_to_world output).
        keep = np.isfinite(depth_s) & (depth_s > 1e-6)
        if mode == "gt" and valid_src is not None:
            keep = keep & _per_frame(valid_src, s).astype(bool)

        pts = backproject_depth_to_world(
            depth_s, intr_s, extr_s, mask=keep,
            extrinsics_convention=extrinsics_convention,
        )
        if len(pts):
            chunks.append(pts)

    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def sample_pointcloud(points: np.ndarray, max_points: int = 50_000,
                      seed: int = 0) -> np.ndarray:
    """Random subsample to at most ``max_points`` points (no-op if N<=max)."""
    pts = np.asarray(points)
    N = len(pts)
    if N <= max_points or max_points <= 0:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, max_points, replace=False)
    return pts[idx]


def sample_pointcloud_with_companion(points: np.ndarray,
                                     companion: np.ndarray,
                                     max_points: int = 50_000,
                                     seed: int = 0):
    """Like :func:`sample_pointcloud`, but co-samples a per-point ``companion``
    array (e.g. a label / validity mask) with the *identical* indices.

    The ``points`` half is byte-identical to ``sample_pointcloud(points,
    max_points, seed)`` (same RNG call sequence), so existing point samples are
    unchanged; ``companion[idx]`` stays row-aligned with the returned points.

    Returns ``(points_s, companion_s)``.
    """
    pts = np.asarray(points)
    comp = np.asarray(companion)
    N = len(pts)
    if N <= max_points or max_points <= 0:
        return pts, comp
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, max_points, replace=False)
    return pts[idx], comp[idx]


def pred_cloud_gtvalid_mask(depth: np.ndarray,
                            gt_valid: np.ndarray | None,
                            keep: np.ndarray | None = None) -> np.ndarray | None:
    """Per-point GT-layout-depth-validity mask, row-aligned with a predicted cloud.

    The predicted cloud (raw / scale / sim3 / scale_shift_cam) is produced by
    backprojecting ``depth`` with the gate ``np.isfinite(d) & (d > 1e-6)``
    (optionally ``& keep``), per frame, row-major, concatenated across frames,
    exactly the gate :func:`backproject_depth_to_world` /
    :func:`build_scene_pointcloud_from_batch` apply internally. Applying that
    same gate to each frame's ``gt_valid`` slice yields a boolean array aligned
    1:1 with the cloud's points.

    Args:
        depth:    (S, H, W) depth used to build the cloud for this track. **Pass
            the SAME (scaled) depth** the cloud was built from, ``pred_ld`` for
            the raw track, ``pred_ld * scale`` for an aligned track, because the
            ``> 1e-6`` gate is not invariant to scaling near the threshold.
        gt_valid: (S, H, W) bool, True where GT layout depth is valid
            (``gt_layout_depth > 1e-6`` AND ``layout_depth_masks``). ``None``
            disables masking (returns ``None``).
        keep:     optional (S, H, W) bool keep-mask applied in addition to the
            depth-validity gate (mirrors the ``mask`` arg of the backprojector).

    Returns:
        1-D bool array (``True`` = pred point's source pixel has valid GT layout
        depth), or ``None`` when ``gt_valid is None``.
    """
    if gt_valid is None:
        return None
    depth = np.asarray(depth)
    gt_valid = np.asarray(gt_valid)
    S = depth.shape[0]
    chunks: list[np.ndarray] = []
    for s in range(S):
        d = np.asarray(depth[s], dtype=np.float32)
        final = np.isfinite(d) & (d > 1e-6)
        if keep is not None:
            ks = np.asarray(keep[s], dtype=bool)
            if ks.shape == final.shape:
                final = final & ks
        if final.any():
            chunks.append(np.asarray(gt_valid[s]).reshape(-1)[final.reshape(-1)])
    if not chunks:
        return np.zeros((0,), dtype=bool)
    return np.concatenate(chunks).astype(bool)


def build_kdtree(points: np.ndarray):
    """Build a ``scipy.spatial.cKDTree`` over ``points``; ``None`` if empty
    or scipy is unavailable.

    Exposed so callers can build a tree once per scene and reuse it across
    multiple :func:`chamfer_and_fscore` calls (the same GT cloud is queried
    in raw / scale / sim3 / scale_shift_cam alignment tracks).
    """
    if points is None or len(points) == 0:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    return cKDTree(points)

